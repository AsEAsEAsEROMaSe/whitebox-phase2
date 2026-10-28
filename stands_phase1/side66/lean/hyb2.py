import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; i = int(sys.argv[1]); K = int(sys.argv[2])
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); st = lk.build_static_coefs(w.shape[1])
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
for S in [None, {"k211"}, {"B2"}, {"wk22"}, {"31"}, {"trunc"}, {"hooks"}]:
    lk.HYB_K = None if S is None else K; lk.HYB_SET = S; lk.REGEN_STATE.clear()
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre,
                           g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})[-1].cpu().double().numpy()
    print(i, K, S, "%.4e" % np.mean((p-gt)**2), flush=True)
