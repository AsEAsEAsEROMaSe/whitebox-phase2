"""С119-б/05.09 — НОСІЙ ДЛЯ ПАРНИХ КУМУЛЯНТІВ: «ДІАГОНАЛЬНЕ ПОРОДЖЕННЯ + ЛАНЦЮГ U» (офіційні mini, моменти keenanpepper N=1e8)

Повний оракул (істинні κ₂₁ + γ₄ + κ₂₂/κ₃₁, усі ПАРНІ) дає 0.0050% — рівень #1. Питання: чим ці парні блоки НЕСТИ.
Модель: джерело на шарі g = власні кумулянти нейронів h_g (κ₃(h_a), κ₄(h_a) — істинні з post_m3/m4 = оракул породження),
перенос лінійним відгуком: U^{g→l} = (U^{g→l−1} ∘ Φ_{l−1}) @ W_l, U^{g→g+1} = W_{g+1}. Тоді
   κ₂₁(p_l)_ij += Σ_a κ₃(h_a) U_ai² U_aj      κ₂₂ += Σ_a κ₄ U_ai² U_aj²      κ₃₁ += Σ_a κ₄ U_ai³ U_aj      γ₄-чисельник += Σ_a κ₄ U_ai⁴
   (+ MODE=hub: гаусів хаб (2,1,1) з руками v_a = c₁∘ρ_a, як AMAX=1 у rec118h; MODE=both — сума)
Рангу 1024 у розгортці (i;j), «ізотропна» — саме форма зміряного дефіциту (kdir: поза топ-64 ×1.50, FITW хабами R²<0).
Ціна носія: 1 матмул на ланцюг + по 1 на блок, на пару (g,l): ~0.1–0.2 B fp32 без обрізання.
ПЕРЕДБАЧЕННЯ: R² моделі diag проти істинної κ₂₁ поза топ-64 > 0.5 (шанс 45%); κ₂₂ R² > 0.8 (55%) — бо κ₄(h_a) O(1), парні O(ρ²).
Пломба: MODE=hub мусить дати R²(κ₂₁) ≈ 0.9 як у rec118h AMAX=1 (незалежна реалізація тієї ж моделі).
Вихід: /home/trim/wblogs/kp119d_<idx>_<mode>.npz з P3f_l, k22_l, k31_l (нормовані), g4_l — для rec118h KBANK/K4B.
"""
import os, sys, time, glob
import numpy as np, pyarrow.parquet as pq
from math import erf
D, NW = 16, 1024; OUT = "/home/trim/wblogs"; T0 = time.time(); SQ2PI = np.sqrt(2*np.pi)
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
MODE = os.environ.get("MODE", "diag"); NETS = [int(x) for x in os.environ.get("NETS", "0").split(",")]
PHI = os.environ.get("PHI", "true")     # true: Φ з істинного t; (пізніше: our)
def phi(t): return np.exp(-0.5*t*t)/SQ2PI
_erf = np.vectorize(erf)
def Phi(t): return 0.5*(1.0 + _erf(t/np.sqrt(2)))
def He(n, t):
    a, b = np.ones_like(t), t
    if n == 0: return a
    for k in range(1, n): a, b = b, t*b - k*a
    return b
def hcoef(s, t, N):
    ph = phi(t); c = [s*(ph + t*Phi(t)), s*Phi(t)]; f = 1.0
    for n in range(2, N+1): f *= n; c.append(s*((-1)**n)*He(n-2, t)*ph/f)
    return c
def r2(M, T): return 1 - np.linalg.norm(M - T)**2/np.linalg.norm(T)**2
def load(i):
    kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); nm = str(kp["mlp_name"]); W = None
    for f in sorted(glob.glob("/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data/mini-*.parquet")):
        t = pq.read_table(f, columns=["mlp_name"]); nms = [x.as_py() for x in t["mlp_name"]]
        if nm in nms:
            W = np.asarray(pq.read_table(f, columns=["weights"])["weights"][nms.index(nm)].as_py(), np.float64).reshape(D, NW, NW); break
    return nm, W, kp
def true_pair(kp, l):
    mu = kp["pre_mean"]; m3 = kp["pre_m3"]; a = mu[l]
    M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64)
    M31 = kp["pre_M31"][l].astype(np.float64); M22 = kp["pre_M22"][l].astype(np.float64); E2 = np.diag(M11)
    k21 = M21 - a[None, :]*E2[:, None] - 2*a[:, None]*M11 + 2*(a*a)[:, None]*a[None, :]
    C11 = M11 - np.outer(a, a)
    mu31 = M31 - a[None, :]*m3[l][:, None] - 3*a[:, None]*M21 + 3*(a[:, None]*a[None, :])*E2[:, None] + 3*(a*a)[:, None]*M11 - 3*(a**3)[:, None]*a[None, :]
    mu22 = M22 - 2*a[None, :]*M21 - 2*a[:, None]*M21.T + (a*a)[None, :]*E2[:, None] + (a*a)[:, None]*E2[None, :] + 4*(a[:, None]*a[None, :])*M11 - 3*(a*a)[:, None]*(a*a)[None, :]
    k22 = mu22 - np.outer(E2 - a*a, E2 - a*a) - 2*C11*C11
    k31 = mu31 - 3*(E2 - a*a)[:, None]*C11
    return k21, k22, k31, C11
for i in NETS:
    nm, W, kp = load(i)
    mu = kp["pre_mean"]; sd = np.sqrt(kp["pre_m2"] - mu*mu); t = mu/sd
    hm = kp["mean"]; hm2 = kp["post_m2"]; hm3 = kp["post_m3"]; hm4 = kp["post_m4"]
    hv = hm2 - hm*hm; k3h = hm3 - 3*hm*hm2 + 2*hm**3; k4h = hm4 - 4*hm*hm3 + 6*hm*hm*hm2 - 3*hm**4 - 3*hv*hv
    lg("%s (%d): κ₃(h) mean %.3f · κ₄(h) mean %.3f на l=0 / %.3f %.3f на l=8" % (nm, i, k3h[0].mean(), k4h[0].mean(), k3h[8].mean(), k4h[8].mean()))
    chains = {}   # g -> U^{g->l}
    vch = {}      # g -> V^{g->l} (хабові руки)
    out = {}; res = []
    for l in range(1, D):
        Ph = Phi(t[l-1])
        # оновити ланцюги: g < l-1 переносяться через Φ_{l-1} і W_l; нове джерело g = l-1: U = W_l
        for g in list(chains):
            chains[g] = (chains[g]*Ph[None, :]) @ W[l]
            if g in vch:
                for key in vch[g]: vch[g][key] = (vch[g][key]*Ph[None, :]) @ W[l]
        chains[l-1] = W[l].copy()
        if MODE != "diag":
            s_ = sd[l-1]; c = hcoef(s_, t[l-1], 3); rho = (kp["pre_M11"][l-1].astype(np.float64) - np.outer(mu[l-1], mu[l-1]))/np.outer(s_, s_)
            vch[l-1] = {"V": (c[1][None, :]*rho) @ W[l]}          # рядок a: v_a = c₁∘ρ_a, перенесено W_l
            if "2" in MODE: vch[l-1]["V2"] = (c[2][None, :]*rho*rho) @ W[l]      # гаусова (3,1,2): рука c₂∘ρ_a²
            if "f" in MODE:                                                   # зворотний зв'язок: істинна κ₂₁(p_{l-1}) як рука
                k21g = true_pair(kp, l-1)[0] if l-1 >= 1 else np.zeros((NW, NW)); k21n = k21g/np.outer(s_*s_, s_)   # нормована κ₂₁(x): [i,j] = κ(x_i,x_i,x_j)
                vch[l-1]["Q"] = (c[2][None, :]*k21n.T) @ W[l]      # q_j(i) = c₂(i)·κ₂₁_ij  (рядок j)  — E[He₂He₂He₁] ∋ 2κ₂₁_ij ρ_jk, хаб j
                vch[l-1]["R"] = (c[1][None, :]*k21n) @ W[l]        # r_i(j) = c₁(j)·κ₂₁_ij  (рядок i)  — E[He₃He₁He₁] ∋ 3κ₂₁_ij ρ_ik, хаб i
        K21 = np.zeros((NW, NW)); K22 = np.zeros((NW, NW)); K31 = np.zeros((NW, NW)); G4 = np.zeros(NW)
        for g, U in chains.items():
            U2 = U*U
            if MODE in ("diag", "both"):
                K21 += U2.T @ (k3h[g][:, None]*U); K22 += U2.T @ (k4h[g][:, None]*U2); K31 += (U2*U).T @ (k4h[g][:, None]*U); G4 += (U2*U2).T @ k4h[g]
            if MODE != "diag":
                V = vch[g]["V"]; cg = hcoef(sd[g], t[g], 3); c2 = cg[2]; c3 = cg[3]
                K21 += 2*(2*(U*V).T @ (c2[:, None]*V) + (V*V).T @ (c2[:, None]*U))   # E[He₂He₁He₁] = 2ρρ ⟹ множник 2 (пломба масштабу 05.09)
                def sym3(w, A, B):   # хаб u з двома РІЗНИМИ руками A,B, вага w: K21 += Σ w[2 u_i A_i B_j + 2 u_i B_i A_j + 2 A_i B_i u_j]
                    return 2*((U*A).T @ (w[:, None]*B) + (U*B).T @ (w[:, None]*A) + (A*B).T @ (w[:, None]*U))
                if "2" in MODE: K21 += sym3(6*c3, V, vch[g]["V2"])
                if "f" in MODE: K21 += sym3(2*c2, vch[g]["Q"], V) + sym3(3*c3, vch[g]["R"], V)
        k21t, k22t, k31t, C11 = true_pair(kp, l); s_ = sd[l]
        _U, _S, _Vt = np.linalg.svd(k21t, full_matrices=False)
        P = lambda M: M - _U[:, :64] @ (_U[:, :64].T @ M @ _Vt[:64].T) @ _Vt[:64]
        g4t = np.diag(k22t)/s_**4
        line = "l=%2d κ₂₁: R² %.3f діаг %.3f поза-топ64 %.3f масштаб %.2f | κ₂₂: R² %.3f діаг(γ₄) %.3f | κ₃₁: R² %.3f | γ₄ corr %.3f" % (l,
            r2(K21, k21t), 1 - ((np.diag(K21) - np.diag(k21t))**2).sum()/(np.diag(k21t)**2).sum(), r2(P(K21), P(k21t)),
            (K21*k21t).sum()/(K21*K21).sum(), r2(K22, k22t), 1 - ((np.diag(K22) - np.diag(k22t))**2).sum()/(np.diag(k22t)**2).sum(),
            r2(K31, k31t), np.corrcoef(G4/s_**4, g4t)[0, 1])
        if l in (1, 2, 4, 8, 12, 15): lg("   " + line)
        res.append(line)
        out["P3f_%d" % l] = K21; out["k22_%d" % l] = K22/np.outer(s_**2, s_**2); out["k31_%d" % l] = K31/np.outer(s_**3, s_); out["g4_%d" % l] = G4/s_**4
    np.savez("%s/kp119d_%d_%s.npz" % (OUT, i, MODE), **out)
print("DONE_KP119D")
