"""Передбачення ланцюга REG (рекордна конфігурація) на дампах 0..7, усі шари, для аналізу структури похибки (27.09).
Питання: (1) стеля помережного скаляра α_i на виході (оракул проти істини); (2) чи похибка e_l = pred−true колінеарна
між шарами (перенос) і з m (масштаб); (3) частка похибки в топ-k напрямках C. Передбачення: оракульний α_i дає ≤×1.05 MSE (шанс 70%)."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["all_layer_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    lk.REGEN_STATE.clear(); t0 = time.time()
    hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG).detach().cpu().double().numpy()
    mse = ((p - gt) ** 2).mean(1)
    np.savez("/home/trim/wb2data/pred_%d.npz" % i, pred=p, gt=gt, name=str(d["mlp_name"]))
    print("dump %d final %.4e  per-layer: %s  (%.0f с)" % (i, mse[-1], " ".join("%.2e" % m for m in mse), time.time() - t0), flush=True)
    torch.cuda.empty_cache()
print("DONE_PRED")
