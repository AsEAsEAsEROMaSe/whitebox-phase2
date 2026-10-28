"""Розклад різниці reg↔dense: dense + ОДНЕ спрощення v29 за раз. Для кожного: MSE і оптимальна суміш з dense (γ LOO).
Передбачення ДО: ≥60% виграшу суміші несе regen (G_off = λC_off) — шанс 40%."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_oracle as lo
lk = lo.lk
i = int(sys.argv[1]); dt = torch.float32
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
st = lk.build_static_coefs(w.shape[1]); out = {}
def run(tag, **kw):
    t0 = time.perf_counter(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **kw)[-1].detach().cpu().double().numpy(); out[tag] = p
    print("dump%d %-8s mse %.4e t %.0fs" % (i, tag, np.mean((p-gt)**2), time.perf_counter()-t0), flush=True)
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
run("k211", k4_aug=True, aug_no_k211=True)
run("regen", k4_aug=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
run("B2", k4_aug=True, shadow_B2=True)
run("wk22", k4_aug=True, no_wk22_fac=True)
run("no31", k4_aug=True, aug_no_31=True)
run("trunc", k4_aug=True, leg_trunc={"rres": 16, "d21": 16})
np.savez("/home/trim/wb2data/extrap2_%d.npz" % i, **out); print("DONE_EXTRAP2", i, flush=True)
