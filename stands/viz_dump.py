"""13.10.2026: візуалізація «наш best проти теоретичної межі» на дампі 0 (regen-ланцюг, банк N=1e8, істина виходу N=1e9)."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
I = int(os.environ.get("DUMP", "0")); OUT = f"/home/trim/wb2data/viz_{I}.npz"
d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(I)); n = w.shape[1]
st = lk.build_static_coefs(n); dt = torch.float32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
rec = {}
class DH(lo.D21Hook):
    def __call__(self, l, D3, D21, live):
        if l >= 1: rec[("D21", l)] = D21.detach().double().cpu().numpy(); rec[("D3", l)] = D3.detach().double().cpu().numpy(); rec[("C", l)] = live["C"].detach().double().cpu().numpy(); rec[("mu", l)] = live["mu"].detach().double().cpu().numpy()
        return super().__call__(l, D3, D21, live)
class WH(lo.Hook):
    def __call__(self, l, wk4_row, wk4_mat, wk431, live):
        if wk4_row is not None: rec[("k4", l)] = wk4_row.detach().double().cpu().numpy()
        return super().__call__(l, wk4_row, wk4_mat, wk431, live)
def run(oracle):
    lo.mode = "all" if oracle else "rec"; lo.D21USE = 1 if oracle else 0
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None
    lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    lk.WK_HOOK = WH(kp); lk.D21_HOOK = DH(kp)
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
    return p.detach().double().cpu().numpy()
pb = run(False); base = dict(rec); rec.clear()
po = run(True)
res = dict(gt=gt, pred_base=pb[-1], pred_oracle=po[-1], all_base=pb, all_oracle=po, avg_var=float(d["avg_variance"]), n=n)
for (k, l), v in base.items(): res[f"{k}_{l}"] = v
for l in range(1, 16):
    a = kp["pre_mean"][l].astype(np.float64); E2 = kp["pre_m2"][l]; E3 = kp["pre_m3"][l]; E4 = kp["pre_m4"][l]; v = E2 - a*a
    _, _, _, mu21 = lo.true_pair(kp, l); res[f"tD21_{l}"] = mu21; res[f"tD3_{l}"] = E3 - 3*a*E2 + 2*a**3
    res[f"tk4_{l}"] = E4 - 4*a*E3 + 6*a*a*E2 - 3*a**4 - 3*v*v; res[f"tC_{l}"] = kp["pre_M11"][l].astype(np.float64) - np.outer(a, a)
np.savez(OUT, **res)
print("base", float(((pb[-1]-gt)**2).mean()), "oracle", float(((po[-1]-gt)**2).mean()), "floor", res["avg_var"]/1e9)
