"""Ланцюг reg (lean) на довільному дампі (env DUMPF з %d): MSE фінального шару."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
i = int(sys.argv[1]); d = np.load(os.environ["DUMPF"] % i); w = d["weights"]; gt = d["final_means"].astype(np.float64); n = w.shape[1]
st = lk.build_static_coefs(n); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K3_PROBE = None; lk.K3_PROBE_POST = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); t0 = time.time(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
lk.REGEN_STATE.clear(); p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
pr = p[-1].detach().cpu().double().numpy()
print("dump%d ланцюг reg mse_final %.4e (%.0f с) | gauss-МК підлога %.2e" % (i, np.mean((pr - gt)**2), time.time() - t0, float(d["final_vars"].mean() / float(d["n_samples"])) if "final_vars" in d else 0), flush=True)
lk.REGEN_STATE.clear(); p0 = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, k4_aug=True)
print("dump%d ланцюг dense(k4_aug) mse_final %.4e" % (i, np.mean((p0[-1].detach().cpu().double().numpy() - gt)**2)), flush=True)
