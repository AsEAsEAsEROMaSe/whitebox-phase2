import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
d = np.load(lo.DUMP.format(0)); w = d["weights"]; st = lk.build_static_coefs(w.shape[1]); dt = torch.float32
fit = lo.FitC(); lk.REGEN_STATE.clear(); t0=time.perf_counter(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit); print("fit %.1fs"%(time.perf_counter()-t0), flush=True)
def reg():
    t0=time.perf_counter(); lk.REGEN_STATE.clear()
    lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    return time.perf_counter()-t0
lk.TERM_SEEN = set(); print("reg %.1fs" % reg(), flush=True); S = sorted(lk.TERM_SEEN); lk.TERM_SEEN = None
print(len(S), S, flush=True)
for g in S[:3]:
    lk.TERM_SCALE = {g: 1.1}; print("pert", g, "%.1fs" % reg(), flush=True)
