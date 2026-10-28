"""16.10.2026 вечір: дешеві варіанти предиктора κ4. Як stack_pred.py, але зберігає внесок КОЖНОЇ пари (L, l) окремо:
gsP[L,l,j] (гаус), k3P[b][L,l,j] (член κ3 з джерел віку b: 0 = вік ≤1, 1 = 2..3, 2 = ≥4), k3sP[L,l,j] (член κ3 через зрізи:
T(e_a,ℓ,ℓ) ≈ D3_a ℓ_a² + Σ_b D21[b,a] ℓ_b² + 2 ℓ_a Σ_b D21[a,b] ℓ_b, без трійок різних індексів), а також лічильник
множень n²×(колонки) на член κ3 по віку (для кошторису). Варіанти (шари L ≥ LMIN, останні KB народжень, вид члена κ3,
частка нейронів) збирає офлайн stack_var.py, інжекцію робить режим inject (PSUF, LMIN).
CHAIN: reg | g96. Режими: raw i | inject i.
Передбачення ДО прогону: зрізове наближення бере ≤40% цінності члена κ3 (трійки різних індексів, «море», домінують);
L ≥ 10 з 3 останніми народженнями бере ~60% виграшу повного предиктора."""
import os, sys, numpy as np, torch
from math import sqrt, pi
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dev = lk.DEV; mode = sys.argv[1]; i = int(sys.argv[2]); CH = os.environ.get("CHAIN", "reg"); dt = torch.float32; A4 = 1.0
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
    f2 = [(ph[l]/s[l]).float() for l in range(NL)]
    k3P = np.zeros((3, NL, NL, n), np.float32); k3sP = np.zeros((NL, NL, n), np.float32); NMM = np.zeros((3, NL), np.int64); SEAL = {}
    bucket = lambda a: 0 if a <= 1 else (1 if a <= 3 else 2)
    def contract_b(K3, V):
        outs = [torch.zeros(n, V.shape[1], dtype=torch.float32, device=dev) for _ in range(3)]; cnt = [0, 0, 0]
        for s_ in K3.sources:
            assert s_.get("Q") is None
            b = bucket(s_.get("age", 0)); P = s_["P"]; Pt = V if P is None else P.T @ V; cnt[b] += 0 if P is None else 1
            Pm = (lambda A: A) if P is None else (lambda A, P=P: P @ A)
            for xk, yk in K3.PAIRS:
                X, Y = s_.get(xk), s_.get(yk)
                if X is None: continue
                Xt, Yt = X.T @ V, Y.T @ V
                outs[b] += (X @ (Pt*Yt) + Pm(Xt*Yt) + Y @ (Xt*Pt)) / 3.0; cnt[b] += 4 + (0 if P is None else 1)
            if s_.get("M") is not None:
                Mx = s_["M"]; Mt = Mx.T @ V; outs[b] += (Mx @ (Pt*Pt) + 2.0*Pm(Mt*Pt)) / 3.0; cnt[b] += 2 + (0 if P is None else 1)
        return outs, cnt
    def probe(l, K3, D3, D21):
        if l == 3:
            Iv = sum(contract_b(K3, torch.eye(n, dtype=torch.float32, device=dev))[0]); off = ~torch.eye(n, dtype=torch.bool, device=dev)
            SEAL["p1"] = float((Iv - D21.T)[off].abs().max()/D21.abs().max())
        Cf = C[l].float()
        for L in range(l+1, NL):
            Vl = VG[(L, l)]; outs, cnt = contract_b(K3, Vl); wgt = 12*G[(L, l)]*f2[l][:, None]*(Cf @ Vl)
            for b in range(3): k3P[b, L, l] = (wgt*outs[b]).sum(0).cpu().numpy()
            if L == l+1: NMM[:, l] = cnt
            ksl = D3[:, None]*Vl*Vl + D21.T @ (Vl*Vl) + 2*Vl*(D21 @ Vl)
            k3sP[L, l] = (wgt*ksl).sum(0).cpu().numpy()
    lk.K3_PROBE = probe; lk.WK_HOOK = None; mseB = chain(); lk.K3_PROBE = None
    x = torch.linspace(-14, 14, 4001, dtype=torch.float64, device=dev); wx = torch.exp(-x*x/2); wx /= wx.sum()
    gsP = np.zeros((NL, NL, n), np.float32)
    for l in range(NL-1):
        m_, s_, t_ = M[l], s[l], t[l]; f3 = (-t_*ph[l]/s_**2).float()
        Cf = C[l].float(); Co = Cf.clone(); Co.fill_diagonal_(0); Co2 = Co*Co
        r = torch.relu(m_[:, None]+s_[:, None]*x[None, :]); e1 = (r*wx).sum(1); rc = r-e1[:, None]; k4own = ((rc**4*wx).sum(1)-3*((rc**2*wx).sum(1))**2).float()
        for L in range(l+1, NL):
            Gl = G[(L, l)]; V = Co @ VG[(L, l)]
            own = ((Gl**4)*k4own[:, None]).sum(0); star = 4*(Gl*f3[:, None]*V**3).sum(0)
            U = Gl*f2[l][:, None]*V; path = 12*(U*(Co@U)).sum(0); G2 = Gl*Gl*f2[l][:, None]; dbl = 3*(G2*(Co2@G2)).sum(0)
            gsP[L, l] = (own+star+path+dbl).cpu().numpy()
    def k4true(l):
        T = lambda k: torch.tensor(kp[k][l].astype(np.float64), device=dev)
        a = T('pre_mean'); E2, E3, E4 = T('pre_m2'), T('pre_m3'), T('pre_m4'); v = E2 - a*a
        return (E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v).cpu().numpy()
    kt = np.stack([k4true(l) for l in range(NL)])
    print("%s d%d raw2: 🔒 MSE A %.4e = B %.4e · V=I %.1e · множень κ3 на шар (вік ≤1/2–3/≥4) l=14: %s" %
          (CH, i, mseA, mseB, SEAL.get("p1", float("nan")), NMM[:, 14].tolist()), flush=True)
    np.savez('/home/trim/wblogs/k4pp_%s_d%d.npz' % (CH, i), gsP=gsP, k3P=k3P, k3sP=k3sP, NMM=NMM, kt=kt,
             t=torch.stack(t).cpu().numpy(), s=torch.stack(s).cpu().numpy())
elif mode == "inject":
    suf = os.environ["PSUF"]; LMIN = int(os.environ.get("LMIN", "1")); FR = float(os.environ.get("FRAC", "0.25"))
    pred = np.load('/home/trim/wblogs/k4pv_%s_%s_d%d.npy' % (CH, suf, i))
    def inj(l, wr, wm, w431, live):
        if l < LMIN or wr is None: return wr, wm, w431
        mu = live["mu"].detach().double(); v = live["var"].detach().double(); s_ = v.sqrt(); t_ = mu/s_
        b = ((t_*t_-1)*torch.exp(-t_*t_/2)/(24*s_**3)).abs(); ix = torch.topk(b*v*v, int(round(FR*n))).indices
        out = wr.detach().double().clone(); out[ix] = torch.as_tensor(pred[l], dtype=torch.float64, device=dev)[ix]/A4
        return out.to(wr.dtype), wm, w431
    base = float(os.environ["BASE"]) if os.environ.get("BASE") else None
    lk.K3_PROBE = None
    if base is None: lk.WK_HOOK = None; base = chain()
    lk.WK_HOOK = inj; m = chain(); lk.WK_HOOK = None
    print("%s d%d %s LMIN=%d FRAC=%.2f: база %.4e → %.4e (×%.3f)" % (CH, i, suf, LMIN, FR, base, m, base/m), flush=True)
