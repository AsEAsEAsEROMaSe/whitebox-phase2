"""(а) масштабний зсув виходу v29-ланцюга: a* = <pred,gt>/<pred,pred>, MSE сира vs після a*; (б) ранг D21 старих джерел (age>=4) по шарах."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_k3_aug_gpu as lk
DEV = lk.DEV; dtype = torch.float32
DUMP = "/home/trim/wb2data/dumps_p2/mlp_{:04d}.npz"
dumps = [int(x) for x in sys.argv[1].split(",")]; RANKS = (16, 32, 64, 128, 256, 384)
def zd(M): M = M.clone(); M.fill_diagonal_(0.0); return M
class FitC:
    def __init__(self): self.coef = []
    def __call__(self, l, G, live):
        n = G.shape[0]; m = ~torch.eye(n, dtype=torch.bool, device=G.device); y = zd(G)[m]; x = zd(live["C"])[m]
        a = float((x*y).sum()/(x*x).sum()); self.coef.append(a); return torch.diag(torch.diagonal(G)) + a*zd(live["C"])
class Replay:
    def __init__(self, coef): self.coef = coef
    def __call__(self, l, G, live):
        lk.REGEN_STATE["lam"] = float(self.coef[l]); lk.REGEN_STATE["g"] = torch.diagonal(G).clone()
        return torch.diag(torch.diagonal(G)) + self.coef[l]*zd(live["C"])
def approx_pre(G, C_pre_off):
    lam = lk.REGEN_STATE.get("lam"); return G if lam is None else torch.diag(torch.diagonal(G)) + lam*zd(C_pre_off)
REC = {}
def probe(l, K3, D3, D21):
    S = K3.sources; k = len(S)
    if k <= 4: return
    old = S[:k-4]; young = S[k-4:]
    Do = K3.dslices(subset=old)[1]; Dy = K3.dslices(subset=young)[1]
    so = torch.linalg.svdvals(Do.double()); st = torch.linalg.svdvals(D21.double()); sy = torch.linalg.svdvals(Dy.double())
    eo = (so**2).cumsum(0)/(so**2).sum(); et = (st**2).cumsum(0)/(st**2).sum(); ey = (sy**2).cumsum(0)/(sy**2).sum()
    REC.setdefault(l, []).append((float(Do.norm()/D21.norm()), [float(eo[r-1]) for r in RANKS], [float(et[r-1]) for r in RANKS], [float(ey[r-1]) for r in RANKS], k))
statics = None; A = []; raw = []; cosm = []
for i in dumps:
    d = np.load(DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); lk.K3_PROBE = None
    fit = FitC(); lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, g_hook=fit)
    lk.K3_PROBE = probe if i in dumps[:3] else None
    pred = lk.lean_k3_predict(w, dtype=dtype, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=Replay(fit.coef),
                              g_feed_hook=approx_pre, g_use_hook=approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True,
                              leg_trunc={"rres": 16, "d21": 16}).detach().cpu().double().numpy()[-1]
    a = float((pred*gt).sum()/(pred*pred).sum()); A.append(a); raw.append(float(np.mean((pred-gt)**2))); cosm.append(float(np.mean((a*pred-gt)**2)))
    print("dump%d a*=%.6f raw %.4e після a* %.4e (×%.3f)" % (i, a, raw[-1], cosm[-1], raw[-1]/cosm[-1]), flush=True)
am = float(np.mean(A)); print("СЕРЕДНЄ a* %.6f sd %.6f | raw %.4e | косинусна %.4e (×%.3f) | стала a=%.6f дала б: %.4e" % (am, float(np.std(A)), np.mean(raw), np.mean(cosm), np.mean(raw)/np.mean(cosm),
      am, np.mean([np.mean((am*np.load(DUMP.format(i))["final_means"].astype(np.float64)*0+0)**2) for i in []] or [0])), flush=True)
print("== ранг D21: шар | ‖old‖/‖all‖ | енергія old у топ-r %s | all | young(4)" % (RANKS,))
for l in sorted(REC):
    for (fr, eo, et, ey, k) in REC[l][:1]:
        print("l=%2d k=%2d old/all %.2f | old: %s | all: %s | young: %s" % (l, k, fr, " ".join("%.3f" % v for v in eo), " ".join("%.3f" % v for v in et), " ".join("%.3f" % v for v in ey)), flush=True)
