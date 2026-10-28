"""reg (≈v29) мінус ОДНЕ спрощення: яке спрощення коштує v29 найбільше MSE. Передбачення: trunc або no31 дають ×1.05–1.10."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; R = {}
BASE = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
VAR = {"reg": {}, "-B2": dict(shadow_B2=False), "-wk22": dict(no_wk22_fac=False), "-31": dict(aug_no_31=False), "-trunc": dict(leg_trunc=None),
       "-truncR": dict(leg_trunc={"d21": 16}), "-truncD": dict(leg_trunc={"rres": 16})}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); st = lk.build_static_coefs(w.shape[1])
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for tag, v in VAR.items():
        kw = dict(BASE); kw.update(v); lk.REGEN_STATE.clear()
        try:
            p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **kw)[-1].cpu().double().numpy()
            R.setdefault(tag, []).append(np.mean((p-gt)**2))
        except Exception as e: print(i, tag, "ERR", type(e).__name__, flush=True)
    print("dump", i, " ".join("%s %.4e" % (k, v[-1]) for k, v in R.items()), flush=True)
b = np.mean(R["reg"]); print("MEAN " + " | ".join("%s %.4e ×%.3f" % (k, np.mean(v), b/np.mean(v)) for k, v in R.items()))
