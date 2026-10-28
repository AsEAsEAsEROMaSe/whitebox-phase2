"""reg (≈v29, регенерація λC + тінь B2) + top-r залишку справжньої G_off − λC_off (k211 рахується лише для залишку).
Пломба: r=0 ⟹ рівно reg 2.1995e-8. Передбачення ДО прогону: r=64 ≤ 2.05e-8 (×1.07); шанс 30% (у dense r64 давав ×1.13)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; R = {}
RS = [int(x) for x in os.environ.get("RS", "0,32,64,96").split(",")]; SC = float(os.environ.get("SC", "1"))
BASE = dict(k4_aug=True, aug_no_k211=False, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
class RegPlus:
    def __init__(self, coef, r): self.coef = coef; self.r = r
    def __call__(self, l, G, live):
        lam = float(self.coef[l]); C = lo.zd(live["C"]); Rm = lo.zd(G) - lam*C
        Rr = torch.zeros_like(G)
        if self.r > 0:
            ev, U = torch.linalg.eigh(0.5*(Rm+Rm.T)); idx = torch.argsort(ev.abs(), descending=True)[:self.r]
            Rr = SC * lo.zd((U[:, idx]*ev[idx]) @ U[:, idx].T)
        lk.REGEN_STATE["lam"] = lam; lk.REGEN_STATE["g"] = torch.diagonal(G).clone(); lk.REGEN_STATE["Rr"] = Rr
        return torch.diag(torch.diagonal(G)) + lam*C + Rr
def pre_plus(G, C_pre_off):
    lam = lk.REGEN_STATE.get("lam")
    if lam is None: return G
    return torch.diag(torch.diagonal(G)) + lam*lo.zd(C_pre_off) + lk.REGEN_STATE.get("Rr", 0.0)
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for r in RS:
        lk.REGEN_STATE.clear()
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=RegPlus(fit.coef, r), g_feed_hook=pre_plus, g_use_hook=pre_plus, **BASE)
        R.setdefault(r, []).append(float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)))
    print("dump", i, " | ".join("r%d %.4e" % (r, R[r][-1]) for r in RS), flush=True)
for r in RS: print("MEAN r=%d %.4e" % (r, np.mean(R[r])), flush=True)
