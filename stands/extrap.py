"""Річардсон по ПОРЯДКУ (ідея 23.09): чи похибка ланцюга e = gt − p_reg лежить уздовж різниць d_x = p_reg − p_x,
де p_x — ланцюг з обрізаним κ₄-каналом (no_k4, simple, dense aug, без κ3-feedback). Зберігає передбачення; аналіз extrap_an.py.
Передбачення ДО прогону: LOO-корекція p_reg + β·d дає ≤×1.15 у MSE (шанс ×1.3+ = 15%); 🔒 контроль: d чужої мережі ⟹ ×≈1."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_oracle as lo
lk = lo.lk
i = int(sys.argv[1]); dt = torch.float32
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
st = lk.build_static_coefs(w.shape[1]); out = {"gt": gt}
def run(tag, **kw):
    t0 = time.perf_counter(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **kw)[-1].detach().cpu().double().numpy(); out[tag] = p
    print("dump%d %-8s mse %.4e t %.0fs" % (i, tag, np.mean((p-gt)**2), time.perf_counter()-t0), flush=True)
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
run("reg", k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre,
    shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
run("dense", k4_aug=True)
run("simple")
run("nok4", no_k4=True)
run("nofb", k4_aug=True, no_k3_feedback=True)
np.savez("/home/trim/wb2data/extrap_%d.npz" % i, **out); print("DONE_EXTRAP", i, flush=True)
