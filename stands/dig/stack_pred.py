"""16.10.2026 вечір: чи складається предиктор κ4 (гаус + κ3 з джерел ланцюга, чутливі 25%) з іншими ланцюгами.
CHAIN: reg | dense | g96 (dense + wk22 + G = λC + ранг 96 залишку). Режими:
  raw    — прохід A (зняти mu, C ланцюга), напрямки ліній назад, прохід B (K3_PROBE: T_l(e_a, ℓ, ℓ) з джерел), гаус+κ3, kt банку
           → /home/trim/wblogs/k4pred_raw_{CHAIN}_d{i}.npz (потім k4pred_fit.py {CHAIN}_);
  inject — прохід C: wk4_row на 25% чутливих (b·v² з власних mu, var) := предиктор /home/trim/wblogs/k4pred_{CHAIN}_d{i}.npy, решта своя.
Пломби: MSE проходу A = MSE проходу B (той самий ланцюг), згортка V=I = dslices.
Передбачення ДО прогону: на g96 предиктор дає ×1.12–1.20 (менше, ніж на reg ×1.25, бо несене G уже краще тримає κ4d);
разом g96+предиктор ≈1.50–1.62e-8 (×1.36–1.47 від reg); шанс ≥×1.4: 35%."""
import os, sys, numpy as np, torch
from math import sqrt, pi
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dev = lk.DEV; mode = sys.argv[1]; i = int(sys.argv[2]); CH = os.environ.get("CHAIN", "g96"); dt = torch.float32; A4 = 1.0
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
W = torch.tensor(w, dtype=torch.float32, device=dev); NL, n = W.shape[0], W.shape[1]
st = lk.build_static_coefs(n)
class LowG:
    def __init__(self, r): self.r = r
    def __call__(self, l, G, live):
        C = lo.zd(live["C"]); Go = lo.zd(G); lam = float((C*Go).sum()/(C*C).sum()); Rm = Go - lam*C
        ev, U = torch.linalg.eigh(0.5*(Rm+Rm.T)); idx = torch.argsort(ev.abs(), descending=True)[:self.r]
        return torch.diag(torch.diagonal(G)) + lam*C + lo.zd((U[:, idx]*ev[idx]) @ U[:, idx].T)
def chain():
    lk.REGEN_STATE.clear(); lk.C_HOOK = None; lk.D21_HOOK = None
    if CH == "reg":
        wk, pr = lk.WK_HOOK, lk.K3_PROBE; lk.WK_HOOK = None; lk.K3_PROBE = None
        fit = lo.FitC(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit); lk.WK_HOOK, lk.K3_PROBE = wk, pr
        kw = dict(k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre,
                  shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    elif CH == "dense": kw = dict(k4_aug=True)
    elif CH == "g96": kw = dict(k4_aug=True, no_wk22_fac=True, g_hook=LowG(96))
    else: raise SystemExit("CHAIN?")
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **kw)[-1].detach().cpu().double().numpy()
    return float(np.mean((p - gt)**2))
LIVE = {}
def cap(l, wr, wm, w431, live):
    LIVE[l] = (live["mu"].detach().double().clone(), live["C"].detach().double().clone()); return wr, wm, w431
if mode == "raw":
    lk.K3_PROBE = None; lk.WK_HOOK = cap; mseA = chain()
    M = [LIVE[l][0] for l in range(NL)]; C = [LIVE[l][1] for l in range(NL)]
    s = [torch.diagonal(C[l]).clamp_min(1e-30).sqrt() for l in range(NL)]; t = [M[l]/s[l] for l in range(NL)]
    Ph = [0.5*(1+torch.erf(t[l]/sqrt(2))) for l in range(NL)]; ph = [torch.exp(-t[l]**2/2)/sqrt(2*pi) for l in range(NL)]
    G = {}; VG = {}
    for L in range(1, NL):
        g_ = W[L].double()
        for l in range(L-1, -1, -1):
            G[(L, l)] = g_.float(); VG[(L, l)] = (Ph[l][:, None]*g_).float()
            if l > 0: g_ = W[l].double() @ (Ph[l][:, None]*g_)
    K3A = {k: torch.zeros(n, n, dtype=torch.float32, device=dev) for k in G}; SEAL = {}
    def contract(K3, V):
        out = torch.zeros(n, V.shape[1], dtype=torch.float32, device=dev)
        for s_ in K3.sources:
            assert s_.get("Q") is None
            P = s_["P"]; Pt = V if P is None else P.T @ V
            Pm = (lambda A: A) if P is None else (lambda A, P=P: P @ A)
            for xk, yk in K3.PAIRS:
                X, Y = s_.get(xk), s_.get(yk)
                if X is None: continue
                Xt, Yt = X.T @ V, Y.T @ V
                out += (X @ (Pt*Yt) + Pm(Xt*Yt) + Y @ (Xt*Pt)) / 3.0
            if s_.get("M") is not None:
                Mx = s_["M"]; Mt = Mx.T @ V; out += (Mx @ (Pt*Pt) + 2.0*Pm(Mt*Pt)) / 3.0
        return out
    def probe(l, K3, D3, D21):
        if l == 3:
            Iv = contract(K3, torch.eye(n, dtype=torch.float32, device=dev)); off = ~torch.eye(n, dtype=torch.bool, device=dev)
            SEAL["p1"] = float((Iv - D21.T)[off].abs().max()/D21.abs().max())
        for L in range(l+1, NL): K3A[(L, l)] = contract(K3, VG[(L, l)])
    lk.K3_PROBE = probe; lk.WK_HOOK = None; mseB = chain(); lk.K3_PROBE = None
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
    def k4true(l):
        T = lambda k: torch.tensor(kp[k][l].astype(np.float64), device=dev)
        a = T('pre_mean'); E2, E3, E4 = T('pre_m2'), T('pre_m3'), T('pre_m4'); v = E2 - a*a
        return E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v
    kt = torch.stack([k4true(l) for l in range(NL)])
    print("%s d%d raw: 🔒 MSE A %.4e = B %.4e · V=I %.1e" % (CH, i, mseA, mseB, SEAL.get("p1", float("nan"))), flush=True)
    np.savez('/home/trim/wblogs/k4pred_raw_%s_d%d.npz' % (CH, i), gs=gs.cpu().numpy(), k3=k3.cpu().numpy(), kt=kt.cpu().numpy(),
             t=torch.stack(t).cpu().numpy(), s=torch.stack(s).cpu().numpy())
elif mode == "inject":
    pred = np.load('/home/trim/wblogs/k4pred_%s_d%d.npy' % (CH, i)); FR = float(os.environ.get("FRAC", "0.25"))
    def inj(l, wr, wm, w431, live):
        if l < 1 or wr is None: return wr, wm, w431
        mu = live["mu"].detach().double(); v = live["var"].detach().double(); s_ = v.sqrt(); t_ = mu/s_
        b = ((t_*t_-1)*torch.exp(-t_*t_/2)/(24*s_**3)).abs(); ix = torch.topk(b*v*v, int(round(FR*n))).indices
        out = wr.detach().double().clone(); out[ix] = torch.as_tensor(pred[l], dtype=torch.float64, device=dev)[ix]/A4
        return out.to(wr.dtype), wm, w431
    lk.K3_PROBE = None; lk.WK_HOOK = None; base = chain(); lk.WK_HOOK = inj; m = chain(); lk.WK_HOOK = None
    print("%s d%d inject FRAC=%.2f: база %.4e → з предиктором %.4e (×%.3f)" % (CH, i, FR, base, m, base/m), flush=True)
