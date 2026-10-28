"""16.10.2026 (вечір), крок 2 предиктора κ4: УСІ входи з ланцюга. C, t, Φ, напрямки ліній назад з живих дампів reg-ланцюга
(wblogs/live/d{i}_l{L}.npz), а κ3(p_a, ℓ_j, ℓ_j) = T_l(e_a, ℓ_j, ℓ_j) тепер з ВЛАСНИХ джерел κ3 ланцюга (K3Blocks у K3_PROBE
другого проходу regen), а не з 2^20 семплів. Формули гаус+член κ3 як у k4pred_chain.py.
Джерело: T = Σ_s Sym Σ_j [X⊗P⊗Y (пари 1..4) + M⊗P⊗P], P=None ⟹ P=I; Sym(u⊗p⊗w)(e_a,v,v) = (u_a p̃ w̃ + p_a ũ w̃ + w_a ũ p̃)/3, ũ = uᵀv.
🔒 пломби: (1) згортка з V=I дає D21ᵀ поза діагоналлю і D3 на діагоналі (dslices того самого K3); (2) mu, C другого проходу
= збережені живі дампи; (3) MSE другого проходу = база reg 2.1994e-8 (8 дампів).
Вихід: /home/trim/wblogs/k4pred_raw_k3chain_d{i}.npz (gs, k3, kt, t, s), далі k4pred_fit.py k3chain_ і lean_oracle k4selp K4PSUF=k3chain_.
Передбачення ДО прогону: corr гаус+κ3 на чутливих 25% падає з 0.85–0.91 (семпли) до ~0.80–0.88; у ланцюгу 1.80–1.85e-8
(×1.19–1.22 проти ×1.247 з семплами); шанс ≥×1.2: 50%."""
import os, sys, numpy as np, torch
from math import sqrt, pi
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dev = lk.DEV; i = int(sys.argv[1]); dt = torch.float32
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
W = torch.tensor(w, dtype=torch.float32, device=dev); NL, n = W.shape[0], W.shape[1]
T = lambda k, l: torch.tensor(kp[k][l].astype(np.float64), device=dev)
lv = [np.load('/home/trim/wblogs/live/d%d_l%02d.npz' % (i, l)) for l in range(NL)]
M = [torch.tensor(z['mu'], dtype=torch.float64, device=dev) for z in lv]; C = [torch.tensor(z['C'], dtype=torch.float64, device=dev) for z in lv]
s = [torch.diagonal(C[l]).sqrt() for l in range(NL)]; t = [M[l]/s[l] for l in range(NL)]
Ph = [0.5*(1+torch.erf(t[l]/sqrt(2))) for l in range(NL)]; ph = [torch.exp(-t[l]**2/2)/sqrt(2*pi) for l in range(NL)]
def k4true(l):
    a = T('pre_mean', l); E2, E3, E4 = T('pre_m2', l), T('pre_m3', l), T('pre_m4', l); v = E2 - a*a
    return E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v
G = {}; VG = {}
for L in range(1, NL):
    g_ = W[L].double()
    for l in range(L-1, -1, -1):
        G[(L, l)] = g_.float(); VG[(L, l)] = (Ph[l][:, None]*g_).float()
        if l > 0: g_ = W[l].double() @ (Ph[l][:, None]*g_)
K3A = {k: torch.zeros(n, n, dtype=torch.float32, device=dev) for k in G}
SEAL = {}
def contract(K3, V):
    out = torch.zeros(n, V.shape[1], dtype=torch.float32, device=dev)
    for s_ in K3.sources:
        assert s_.get("Q") is None, "конфайнмент не очікувався"
        P = s_["P"]; Pt = V if P is None else P.T @ V
        Pm = (lambda A: A) if P is None else (lambda A, P=P: P @ A)
        for xk, yk in K3.PAIRS:
            X, Y = s_.get(xk), s_.get(yk)
            if X is None: continue
            Xt, Yt = X.T @ V, Y.T @ V
            out += (X @ (Pt*Yt) + Pm(Xt*Yt) + Y @ (Xt*Pt)) / 3.0
        if s_.get("M") is not None:
            Mx = s_["M"]; Mt = Mx.T @ V
            out += (Mx @ (Pt*Pt) + 2.0*Pm(Mt*Pt)) / 3.0
    return out
def probe(l, K3, D3, D21):
    if l == 3:   # 🔒 пломба 1
        Iv = contract(K3, torch.eye(n, dtype=torch.float32, device=dev)); off = ~torch.eye(n, dtype=torch.bool, device=dev)
        SEAL["p1"] = (float((Iv - D21.T)[off].abs().max()/D21.abs().max()), float((torch.diagonal(Iv) - D3).abs().max()/D3.abs().max()))
    for L in range(l+1, NL): K3A[(L, l)] = contract(K3, VG[(L, l)])
def wkcap(l, wr, wm, w431, live):   # 🔒 пломба 2: стан = живі дампи
    e_mu = float((live["mu"].double() - M[l]).abs().max()/M[l].abs().max()); e_C = float((live["C"].double() - C[l]).abs().max()/C[l].abs().max())
    SEAL.setdefault("p2", []).append((l, e_mu, e_C))
    return wr, wm, w431
st = lk.build_static_coefs(n); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None; lk.C_HOOK = None
fit = lo.FitC(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
lk.K3_PROBE = probe; lk.WK_HOOK = wkcap
p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre,
                       g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
lk.K3_PROBE = None; lk.WK_HOOK = None
mse = float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2))
print("d%d 🔒 MSE reg %.4e · пломба1 (V=I проти dslices) D21 %.1e D3 %.1e · пломба2 (стан = живі дампи) max %.1e" %
      (i, mse, SEAL["p1"][0], SEAL["p1"][1], max([max(e[1], e[2]) for e in SEAL["p2"] if e[1] == e[1] and e[2] == e[2]] or [float("nan")])), flush=True)
print("   пломба2 по шарах (l, mu, C):", " ".join("%d:%.0e/%.0e" % e for e in SEAL["p2"]), flush=True)
x = torch.linspace(-14, 14, 4001, dtype=torch.float64, device=dev); wx = torch.exp(-x*x/2); wx /= wx.sum()
gs = torch.zeros(NL, n, dtype=torch.float64, device=dev); k3 = torch.zeros_like(gs)
for l in range(NL-1):
    m_, s_, t_ = M[l], s[l], t[l]; f2 = (ph[l]/s_).float(); f3 = (-t_*ph[l]/s_**2).float()
    Cf = C[l].float(); Co = Cf.clone(); Co.fill_diagonal_(0); Co2 = Co*Co
    r = torch.relu(m_[:, None]+s_[:, None]*x[None, :]); e1 = (r*wx).sum(1); rc = r-e1[:, None]; k4own = ((rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2).float()
    for L in range(l+1, NL):
        Gl = G[(L, l)]; V = Co @ VG[(L, l)]; Vf = Cf @ VG[(L, l)]
        own = ((Gl**4)*k4own[:, None]).sum(0); star = 4*(Gl*f3[:, None]*V**3).sum(0)
        U = Gl*f2[:, None]*V; path = 12*(U*(Co@U)).sum(0); G2 = Gl*Gl*f2[:, None]; dbl = 3*(G2*(Co2@G2)).sum(0)
        gs[L] += (own+star+path+dbl).double(); k3[L] += (12*(Gl*f2[:, None]*K3A[(L, l)]*Vf).sum(0)).double()
kt = torch.stack([k4true(l) for l in range(NL)])
cc = lambda a, b: float(torch.corrcoef(torch.stack([a, b]))[0, 1])
old = '/home/trim/wblogs/k4pred_raw_chain_d%d.npz' % i
k3s = torch.tensor(np.load(old)['k3'], device=dev) if os.path.exists(old) else None
for L in (4, 8, 12, 15):
    v2 = torch.diagonal(C[L])**2; b = (t[L]**2-1).abs()*torch.exp(-t[L]**2/2)/(24*s[L]**3); sel = torch.topk(b*v2, n//4).indices
    print('d%d L=%2d чутл.25%%: corr гаус %.3f · гаус+κ3(ланцюг) %.3f%s' % (i, L, cc(gs[L][sel], kt[L][sel]), cc((gs[L]+k3[L])[sel], kt[L][sel]),
          (' · гаус+κ3(семпли) %.3f · corr членів κ3 ланцюг/семпли %.3f' % (cc((gs[L]+k3s[L])[sel], kt[L][sel]), cc(k3[L][sel], k3s[L][sel]))) if k3s is not None else ''), flush=True)
np.savez('/home/trim/wblogs/k4pred_raw_k3chain_d%d.npz' % i, gs=gs.cpu().numpy(), k3=k3.cpu().numpy(), kt=kt.cpu().numpy(),
         t=torch.stack(t).cpu().numpy(), s=torch.stack(s).cpu().numpy())
print("DONE_K3CHAIN", i, flush=True)
