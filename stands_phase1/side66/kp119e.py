"""С119-б/05.09 — УЗАГАЛЬНЕНА ХАБОВА МОДЕЛЬ κ₂₁ З ІСТИННИМ СТАНОМ: ГАУСОВІ ДІАГРАМИ + НЕГАУСІВ ЗВОРОТНИЙ ЗВ'ЯЗОК (κ₂₁ і κ₄)

Джерело = {H (рядки a: хаб-вектор над i), A, B (руки), w (вага на рядок)}; κ₂₁(p_l)[i,j] += Σ_a w_a[2H_i A_i B_j + 2H_i B_i A_j + 2A_i B_i H_j].
Перенос кожного рядка: X ← (X∘Φ_{l})@W_{l+1} (лінійний відгук ReLU). Стан (t,s,ρ,Φ,c_k) — ІСТИННИЙ з датасету keenanpepper (оракул).
Діаграми породження κ₃(h_l) з p_l (прапорці у MODE через '+'):
  g     E[He₂He₁He₁]=2ρρ            H=e_a  A=B=c₁∘ρ_a           w=2c₂(a)             ← база (kp119d hub, R² 0.97)
  312   E[He₃He₁He₂]=6ρρ²           H=e_a  A=c₁∘ρ_a B=c₂∘ρ_a²   w=6c₃(a)
  f21t  E[He₂He₂He₁]∋2κ₂₁ρ, E[He₃He₁He₁]∋3κ₂₁ρ з ІСТИННОЇ κ₂₁(p_l)   (оракул зворотного зв'язку)
  f21m  те саме з МОДЕЛЬНОЇ κ₂₁(p_l) (самоузгоджено — реальний носій)
  f4    E[He₂He₁He₁]∋κ₄(p_l)_iijk, де κ₄(p_l) = перенесена хаб-3-руки κ₄(h_g): E[He₃He₁He₁He₁]=6ρρρ, w=6c₃(a):
        κ₄(p)_iijk = Σ_a[12c₃ U_ai V_ai V_aj V_ak + 6c₃ V_ai²(U_aj V_ak + V_aj U_ak)] ⟹ джерела H=c₂∘(U∘V) A=B=c₁∘V w=12c₃ · H=c₂∘V² A=c₁∘U B=c₁∘V w=12c₃
Пломба: MODE=g мусить відтворити kp119d hub (R² 0.97, масштаб ~1.0). ПЕРЕДБАЧЕННЯ: f21 сам ⟹ масштаб 1.6 (від'ємний), f4 сам ⟹ масштаб <1 (додатний),
разом (g+312+f4+f21) ⟹ масштаб → 1.0 і R² поза топ-64 на l=15 ↑ з 0.35 (шанс R²>0.7: 40%).
"""
import os, sys, time, glob
import numpy as np, pyarrow.parquet as pq
from math import erf
D, NW = 16, 1024; OUT = "/home/trim/wblogs"; T0 = time.time(); SQ2PI = np.sqrt(2*np.pi)
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
MODE = os.environ.get("MODE", "g"); FL = set(MODE.split("+")); NETS = [int(x) for x in os.environ.get("NETS", "0").split(",")]
SAVE = int(os.environ.get("SAVE", "0")); R22 = int(os.environ.get("R22", "16"))
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
def true_k21(kp, l):
    a = kp["pre_mean"][l]; M11 = kp["pre_M11"][l].astype(np.float64); M21 = kp["pre_M21"][l].astype(np.float64); E2 = np.diag(M11)
    return M21 - a[None, :]*E2[:, None] - 2*a[:, None]*M11 + 2*(a*a)[:, None]*a[None, :]
from math import comb, factorial
def k22h(s_, t_, rho, N=6):
    """гаусове породження парної κ₄(h)_aabb = E[h̃_a²h̃_b²] − v_a v_b − 2C_ab²: h̃² = Σ_j d_j He_j, E[He_j(x_a)He_j(x_b)] = j!ρ^j"""
    c = hcoef(s_, t_, N); d = {}
    for k in range(1, N+1):
        for l_ in range(1, N+1):
            for m in range(0, min(k, l_)+1):
                j = k + l_ - 2*m; d[j] = d.get(j, 0) + c[k]*c[l_]*comb(k, m)*comb(l_, m)*factorial(m)
    C = sum(factorial(k)*np.outer(c[k], c[k])*rho**k for k in range(1, N+1))
    K = sum(factorial(j)*np.outer(d[j], d[j])*rho**j for j in d if j >= 1) - 2*C*C
    return K
def readout(src):
    H, A, B, w = src["H"], src["A"], src["B"], src["w"]
    if A is B:   # однакові руки: 3 розміщення хаба ⟹ 2H_iA_iA_j + A_i²H_j
        return 2*(H*A).T @ (w[:, None]*A) + (A*A).T @ (w[:, None]*H)
    return 2*((H*A).T @ (w[:, None]*B) + (H*B).T @ (w[:, None]*A) + (A*B).T @ (w[:, None]*H))   # різні руки: 6 розміщень
for i in NETS:
    nm, W, kp = load(i)
    mu = kp["pre_mean"]; sd = np.sqrt(kp["pre_m2"] - mu*mu); t = mu/sd
    srcs = []; out = {}
    K21 = np.zeros((NW, NW))   # κ₂₁(p_0) = 0 (p_0 гаусів)
    for l in range(0, D-1):
        # ---- породження на шарі l: h_l = ReLU(p_l); стан p_l істинний
        s_ = sd[l]; tl = t[l]; c = hcoef(s_, tl, 3); rho = (kp["pre_M11"][l].astype(np.float64) - np.outer(mu[l], mu[l]))/np.outer(s_, s_)
        Ph = Phi(tl); I = np.eye(NW); V0 = c[1][None, :]*rho
        new = [{"H": I, "A": V0, "B": V0, "w": 2*c[2]}]
        if "312" in FL: new.append({"H": I, "A": V0, "B": c[2][None, :]*rho*rho, "w": 6*c[3], "kind": "312"})
        if ("f21t" in FL or "f21m" in FL) and l >= 1:
            k21n = (true_k21(kp, l) if "f21t" in FL else K21)/np.outer(s_*s_, s_)
            new.append({"H": I, "A": c[2][None, :]*k21n.T, "B": V0, "w": 2*c[2], "kind": "f21a"})     # хаб j: q_j(i)=c₂(i)κ₂₁_ij
            new.append({"H": I, "A": c[1][None, :]*k21n, "B": V0, "w": 3*c[3], "kind": "f21b"})       # хаб i: r_i(j)=c₁(j)κ₂₁_ij
        if "f4" in FL:
            for sr in srcs:                       # κ₄(p_l) від хаб-3-руки κ₄(h_g) кожного старого джерела (U=sr["H"], V=sr["A"] у просторі p_l)
                if sr.get("kind") != "g": continue
                U, V, c3g = sr["H"], sr["A"], sr["c3"]
                cV = c[1][None, :]*V
                new.append({"H": c[2][None, :]*(U*V), "A": cV, "B": cV, "w": 12*c3g, "kind": "f4"})              # однакові руки (3 розміщення)
                new.append({"H": c[2][None, :]*(V*V), "A": c[1][None, :]*U, "B": c[1][None, :]*V.copy(), "w": 6*c3g, "kind": "f4"})   # різні руки (6 розміщень)
        if "f4d" in FL:                       # κ₄(p_l)_iijk від ВЛАСНОГО ексцесу κ₄(h_g) старих джерел (CP e⁴): H=c₂∘U², A=B=c₁∘U, w=κ₄(h_a)
            for sr in srcs:
                if sr.get("kind") != "g": continue
                U = sr["H"]; cU = c[1][None, :]*U
                new.append({"H": c[2][None, :]*(U*U), "A": cU, "B": cU, "w": sr["k4h"], "kind": "f4d"})
        if "f22" in FL:                       # κ₄(p_l)_iijk від ПАРНОЇ κ₄(h_g)_aabb (гаусове породження) старих джерел
            for sr in srcs:
                if sr.get("kind") != "g": continue
                U = sr["H"]; K22 = sr["k22"]
                G = (U*U).T @ K22                                       # G[i,b] = Σ_a U_ai² κ_ab  → хаб b, руки U_b,U_b
                cU = c[1][None, :]*U
                new.append({"H": (c[2][None, :]*G).T, "A": cU, "B": cU, "w": np.ones(NW), "kind": "f22"})
                for r in range(sr["k22r"].shape[1]):                     # середній член: κ≈Σ λ_r x_r x_rᵀ ⟹ Σ_i e_i⊗P_r[i]⊗P_r[i]·2λ_r
                    x = sr["k22r"][:, r]; P = U.T @ (x[:, None]*U); cP = c[1][None, :]*P
                    new.append({"H": I.copy(), "A": cP, "B": cP, "w": 2*sr["k22l"][r]*c[2], "kind": "f22"})
        hm = kp["mean"][l]; hm2 = kp["post_m2"][l]; hm3 = kp["post_m3"][l]; hm4 = kp["post_m4"][l]; hv = hm2 - hm*hm
        if "f22" in FL:
            K22 = k22h(s_, tl, rho); np.fill_diagonal(K22, 0.0)
            if l == 0:
                _k22d = np.diag(k22h(s_, tl, rho)); _hm = kp["mean"][0]; _h2 = kp["post_m2"][0]; _h3 = kp["post_m3"][0]; _h4 = kp["post_m4"][0]; _hv = _h2 - _hm*_hm
                _k4t = _h4 - 4*_hm*_h3 + 6*_hm*_hm*_h2 - 3*_hm**4 - 3*_hv*_hv
                lg("   🔒 l=0 (p гаусів): діаг κ₂₂(h) гаус проти істинного κ₄(h): rel %.2e" % (np.linalg.norm(_k22d - _k4t)/np.linalg.norm(_k4t)))
            lam, X = np.linalg.eigh(K22); idx = np.argsort(-np.abs(lam))[:R22]
            new[0]["k22"] = K22; new[0]["k22r"] = X[:, idx]; new[0]["k22l"] = lam[idx]
            if l in (0, 4, 8): lg("   l=%d ‖κ₂₂(h)‖ %.3e · топ-%d власних %.1f%% енергії" % (l, np.linalg.norm(K22), R22, 100*(lam[idx]**2).sum()/(lam**2).sum()))
        new[0]["kind"] = "g"; new[0]["c3"] = c[3]; new[0]["k4h"] = hm4 - 4*hm*hm3 + 6*hm*hm*hm2 - 3*hm**4 - 3*hv*hv   # істинний κ₄(h_l) понейронно
        # ---- перенос усіх джерел у p_{l+1}
        Wn = W[l+1]
        def push(sr, scale):
            same = sr["A"] is sr["B"]
            for key in ("H", "A"): sr[key] = ((sr[key]*scale[None, :]) if scale is not None else sr[key]) @ Wn
            sr["B"] = sr["A"] if same else (((sr["B"]*scale[None, :]) if scale is not None else sr["B"]) @ Wn)
        for sr in srcs: push(sr, Ph)
        for sr in new: push(sr, None)      # народжене в h_l: без Φ
        srcs += new
        Kk = {}
        for sr in srcs:
            kd = sr.get("kind") or "x"; Kk[kd] = Kk.get(kd, 0) + readout(sr)
        K21 = sum(Kk.values())
        L = l + 1; k21t = true_k21(kp, L); s1 = sd[L]
        if L in (4, 8, 12, 15):
            _U, _S, _Vt = np.linalg.svd(k21t, full_matrices=False)
            P = lambda M: M - _U[:, :64] @ (_U[:, :64].T @ M @ _Vt[:64].T) @ _Vt[:64]
            lg("   l=%2d κ₂₁: R² %.3f діаг %.3f поза-топ64 %.3f масштаб %.3f · джерел %d" % (L, r2(K21, k21t),
               1 - ((np.diag(K21) - np.diag(k21t))**2).sum()/(np.diag(k21t)**2).sum(), r2(P(K21), P(k21t)), (K21*k21t).sum()/(K21*K21).sum(), len(srcs)))
            if len(Kk) > 1:   # МНК по класах діаграм проти істини
                names = sorted(Kk); X = np.stack([Kk[k].ravel() for k in names], 1); co = np.linalg.lstsq(X, k21t.ravel(), rcond=None)[0]
                Kf = sum(c_*Kk[k] for c_, k in zip(co, names))
                lg("      МНК по класах: %s ⟹ R² %.3f поза-топ64 %.3f" % (" ".join("%s=%.2f" % (k, c_) for k, c_ in zip(names, co)), r2(Kf, k21t), r2(P(Kf), P(k21t))))
        if SAVE: out["P3f_%d" % L] = K21
    if SAVE: np.savez("%s/kp119e_%d_%s.npz" % (OUT, i, MODE.replace("+", "_")), **out)
    lg("%s (%d) MODE=%s готово" % (nm, i, MODE))
print("DONE_KP119E")
