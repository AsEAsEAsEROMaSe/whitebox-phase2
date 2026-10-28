"""Пункт 2: чи є зріз старого ярусу S_old(l) (джерела віку ≥4) функцією ДЕШЕВИХ ознак (стан + молодий зріз + правило зрізу на S_old(l−1))?
Збір даних на GPU: на кожному шарі l≥4 вибірка 100k позадіагональних пар (i,j) з ~24 ознаками і ціллю S_old_ij; понейронно всі n з ціллю D3_old.
Передбачення ДО прогону: GBM out-of-sample R² 0.3–0.4 без правила, ~0.8 з правилом (правило саме 0.76), ε ≥ 30% ⟹ мертво (треба ε ≤ 3%). Шанс живого ~20%."""
import os, sys, numpy as np, torch, time
sys.argv = [sys.argv[0]] + (sys.argv[1:] if len(sys.argv) > 1 else ["0"])
import lean_oracle as lo, lean_k3_aug_gpu as lk
dumps = [int(x) for x in sys.argv[1].split(",")]; AOLD = 4; NS = 100000; OUT = "/home/trim/wb2data/oldfeat_%d.npz"
REC = []
_lin, _wick = lk.K3Blocks.linear_, lk.K3Blocks.wick_
def lin(self, W): REC.append(("W", W.detach().clone())); _lin(self, W)
def wick(self, w1): REC.append(("w1", w1.detach().clone())); _wick(self, w1)
lk.K3Blocks.linear_, lk.K3Blocks.wick_ = lin, wick
class Probe:
    def __init__(self): self.S = {}
    def __call__(self, l, K3, D3, D21):
        old = [s for s in K3.sources if s.get("age", 0) >= AOLD]; young = [s for s in K3.sources if s.get("age", 0) < AOLD]
        def sl(sub):
            if not sub: return None, None
            a, b = K3.dslices(subset=sub); b = b.detach().double().clone(); b.fill_diagonal_(0.0); return a.detach().double().clone(), b
        self.S[l] = dict(old=sl(old), young=sl(young), full=(D3.detach().double().clone(), D21.detach().double().clone()))
class StateHook:
    def __init__(self): self.st = {}
    def __call__(self, l, D3, D21, live):
        self.st[l] = dict(mu=live["mu"].detach().double().clone(), C=live["C"].detach().double().clone(), var=live["var"].detach().double().clone()); return D3, D21, None
statics = None
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; n = w.shape[1]
    if statics is None: statics = lk.build_static_coefs(n)
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; REC.clear(); t0 = time.perf_counter()
    fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    REC.clear(); pr = Probe(); sh = StateHook(); lk.K3_PROBE = pr; lk.D21_HOOK = sh
    lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef),
                       g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.K3_PROBE = None; lk.D21_HOOK = None
    Ws = [r[1].double() for r in REC if r[0] == "W"]; w1s = [r[1].double() for r in REC if r[0] == "w1"]
    dev = Ws[0].device; g = torch.Generator(device="cpu").manual_seed(1000 + i)
    pairs, feats, targ, nfeat, ntarg = [], [], [], [], []
    for l in range(AOLD + 1, 16):
        D3o, So = pr.S[l]["old"]; D3y, Sy = pr.S[l]["young"]; D3f, Sf = pr.S[l]["full"]
        D3op, Sop = pr.S[l - 1]["old"]
        st = sh.st[l]; C = st["C"]; mu = st["mu"]; var = st["var"]; Coff = C.clone(); Coff.fill_diagonal_(0.0)
        T = Ws[l - 1] * w1s[l - 2][None, :]; TT = T * T   # probe(l-1) -> probe(l)
        L1 = TT @ Sop @ T.T; L2 = 2 * (T * (T @ Sop.T)) @ T.T; L3 = (TT * D3op[None, :]) @ T.T; Lsl = L1 + L2 + L3
        d3rule = 3 * ((TT @ Sop) * T).sum(1) + (TT * T) @ D3op
        ii = torch.randint(0, n, (NS,), generator=g); jj = torch.randint(0, n, (NS,), generator=g); m = ii != jj; ii, jj = ii[m].to(dev), jj[m].to(dev)
        CC = Coff @ Coff; CSy = Coff @ Sy; SyC = Sy @ Coff; CSyT = Coff @ Sy.T
        F = [Coff[ii, jj], mu[ii], mu[jj], var[ii], var[jj], Sy[ii, jj], Sy[jj, ii], D3y[ii], D3y[jj], (Coff * Coff)[ii, jj], CC[ii, jj], CSy[ii, jj], SyC[ii, jj], CSyT[ii, jj],
             (Coff[ii, jj] * D3y[ii] / var[ii]), (Coff[ii, jj] * D3y[jj] / var[jj]), Coff[ii, jj] * mu[ii], Coff[ii, jj] * mu[jj], Sf[ii, jj], Sf[jj, ii],
             L1[ii, jj], L2[ii, jj], L3[ii, jj], Lsl[ii, jj], torch.full_like(mu[ii], float(l))]
        pairs.append(torch.stack([ii, jj], 1).cpu().numpy().astype(np.int16)); feats.append(torch.stack(F, 1).float().cpu().numpy()); targ.append(So[ii, jj].float().cpu().numpy())
        NF = [mu, var, D3y, D3f, torch.diagonal(CC), (Coff * Coff).sum(1), Sy.sum(1), Sy.sum(0), Sy.abs().sum(1), d3rule, D3op, torch.full_like(mu, float(l))]
        nfeat.append(torch.stack(NF, 1).float().cpu().numpy()); ntarg.append(D3o.float().cpu().numpy())
        print("dump%d l=%2d |S_old| %.3e |S_young| %.3e |Lsl−S_old|/|S_old| %.3f rule-D3 rel %.3f" % (i, l, float(So.norm()), float(Sy.norm()), float((Lsl - So).norm() / So.norm()), float((d3rule - D3o).norm() / D3o.norm())), flush=True)
    np.savez(OUT % i, pairs=np.concatenate(pairs), feats=np.concatenate(feats), targ=np.concatenate(targ), nfeat=np.concatenate(nfeat), ntarg=np.concatenate(ntarg))
    print("dump%d saved, t=%.0fs" % (i, time.perf_counter() - t0), flush=True)
