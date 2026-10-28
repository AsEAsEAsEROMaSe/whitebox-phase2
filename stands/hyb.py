"""Гібрид: reg до шару K, dense від K. Чи несе хвостовий dense напрямок суміші (×1.236 при K=0)?
Передбачення ДО: K=12 дає ≥60% виграшу суміші (шанс 40%). Контроль: K=0 мусить відтворити dense 1.9006e-8."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; KS = [0, 4, 8, 11, 13, 14, 15]
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); st = lk.build_static_coefs(w.shape[1])
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.HYB_K = None; lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    out = {"gt": gt}
    for K in [None] + KS:
        lk.HYB_K = K; lk.REGEN_STATE.clear()
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre,
                               g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})[-1].cpu().double().numpy()
        out["reg" if K is None else "h%d" % K] = p
    lk.HYB_K = None
    np.savez("/home/trim/wb2data/hyb_%d.npz" % i, **out)
    print("dump", i, " ".join("%s %.4e" % (k, np.mean((v-gt)**2)) for k, v in out.items() if k != "gt"), flush=True)
