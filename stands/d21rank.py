"""Тест 28.09: чи потрібна ланцюгу REG ЩІЛЬНА D21 (зріз κ₃ (2,1)), яка коштує 2 щільні згортки на джерело-шар (≈половина
ціни молодих джерел). d21_rank=r обрізає D21 після dslices до рангу r (усі споживачі бачать обрізану). Якщо MSE при r=64
незмінна ⟹ D21 можна рахувати range-finder'ом за n²r ⟹ флопи ×~0.7 ⟹ бал ×1.4. Передбачення: r=256 ±1%, r=64 +5–20% (шанс 60%)."""
import os, sys, time, numpy as np, torch
sys.argv = sys.argv[:1]; sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32
for i in [0, 1, 2]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["all_layer_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.K_POST_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for r in [None, 512, 256, 128, 64, 32]:
        lk.REGEN_STATE.clear(); t0 = time.time()
        hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, d21_rank=r, **hk, **REG).detach().cpu().double().numpy()
        mse = ((p - gt) ** 2).mean(1)
        print("dump %d d21_rank %-4s final %.4e  L12 %.3e L14 %.3e  (%.0f с)" % (i, r, mse[-1], mse[12], mse[14], time.time() - t0), flush=True)
    torch.cuda.empty_cache()
print("DONE_D21RANK")
