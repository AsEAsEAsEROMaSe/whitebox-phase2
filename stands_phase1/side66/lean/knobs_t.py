"""Ручки-множники на групи нелінійних членів (int_part, #D3/D21, #wk4) у ланцюгу reg (≈v29). Якобіан скінченними
різницями h=0.1 (+ пломба лінійності −h на кожній групі), аналіз LOO по мережах у knobs_t_an.py.
Передбачення ДО: LOO-оптимум дає ≥×1.15 у MSE з шансом 25%; 🔒 чужий якобіан ⟹ ≤×1."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_oracle as lo
lk = lo.lk
i = int(sys.argv[1]); dt = torch.float32; H = 0.02
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
st = lk.build_static_coefs(w.shape[1])
lk.TERM_SCALE = None; fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
def reg(scale):
    lk.TERM_SCALE = scale; lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre,
                           g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
    lk.TERM_SCALE = None; return p[-1].detach().cpu().double().numpy()
G = sorted(g for g in set(lk.term_group(ip, f) for ip, f, c, wk in lk.TERM_TABLE_AUG) if g[1]+g[2] > 0)
p0 = reg(None); out = {"gt": gt, "p0": p0}; t0 = time.perf_counter(); names = []
for k, g in enumerate(G):
    ok = False
    for h in (H, 0.005):
        try:
            pp = reg({g: 1+h}); pm = reg({g: 1-h}); ok = np.isfinite(pp).all() and np.isfinite(pm).all()
        except Exception as e:
            print("dump%d %s h=%.2f ПАДІННЯ %s" % (i, g, h, type(e).__name__), flush=True); ok = False
        if ok: break
    if not ok: continue
    dp = (pp-p0)/h
    if np.abs(dp).max() < 1e-14: continue
    lin = np.linalg.norm(pp+pm-2*p0)/np.linalg.norm(pp-pm)
    out["J_%d" % len(names)] = dp; out["lin_%d" % len(names)] = lin; names.append(str(g))
    print("dump%d %-16s ‖J‖/‖p‖ %.2e нелін %.3f" % (i, g, np.linalg.norm(dp)/np.linalg.norm(p0), lin), flush=True)
out["names"] = np.array(names); np.savez("/home/trim/wb2data/knt_%d.npz" % i, **out)
print("dump%d mse0 %.4e груп %d t %.0fs DONE_KNT" % (i, np.mean((p0-gt)**2), len(names), time.perf_counter()-t0), flush=True)
