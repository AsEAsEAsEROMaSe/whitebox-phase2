"""Фронт вік/ранг конфайнменту старих джерел для dense+wk22 проти reg (lean, 8 дампів, fp32).
Питання: чи dense витримує агресивніше стискання, ніж reg (reg: вік 2–3 ⟹ MSE ×1.4+).
Передбачення ДО прогону: dense+wk22 при (4,384)+(7,224) ≈ 1.90e-8; при (2,384) ≈ 2.2e-8 (деградує як reg); шанс, що dense@(2,x) ≤ reg@(4,384): 25%."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32
BASE = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
GRID = {"none": None, "4:384+7:224": [(4, 384), (7, 224)], "3:384+6:224": [(3, 384), (6, 224)], "2:384+5:224": [(2, 384), (5, 224)],
        "2:512+5:256": [(2, 512), (5, 256)], "3:256+6:128": [(3, 256), (6, 128)], "1:512+4:256": [(1, 512), (4, 256)]}
only = os.environ.get("GRIDS"); GRID = {k: v for k, v in GRID.items() if not only or k in only.split(",")}
CH = os.environ.get("CHAINS", "reg,dw").split(","); R = {}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for ch in CH:
        for g, ar in GRID.items():
            lk.REGEN_STATE.clear()
            try:
                if ch == "reg": p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, age_rank=ar, **BASE)
                else: p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, no_wk22_fac=True, age_rank=ar)
                R.setdefault((ch, g), []).append(float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)))
            except Exception as e: print(i, ch, g, "ERR", type(e).__name__, e, flush=True); R.setdefault((ch, g), []).append(np.nan)
    print("dump", i, " | ".join("%s/%s %.3e" % (c, g, R[(c, g)][-1]) for c in CH for g in GRID), flush=True)
for c in CH:
    for g in GRID: print("MEAN %-4s %-12s %.4e" % (c, g, np.nanmean(R[(c, g)])), flush=True)
