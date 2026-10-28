"""Стеля k_max=4 у зрізі (1,1,1) всередині ланцюга reg (lean, 8 дампів): base reg проти reg + щільна Δ (d4dense).
Передбачення ДО прогону: екстраполяція 24.09 давала ×1.02–1.05 MSE; шанс ≥×1.3: 15%."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk; import d4dense
dt = torch.float32
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
modes = os.environ.get("MODES", "base,d4").split(",")
R = {}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.D4 = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for nm in modes:
        lk.REGEN_STATE.clear(); t0 = time.time(); kp = np.load(lo.BANK.format(i)); lo.D21USE = 0; dh = lo.D21Hook(kp); lk.D21_HOOK = dh
        lk.D4 = d4dense.Dense4(w.shape[1], lk.DEV) if nm.startswith("d4") else None
        d4dense.D4SCALE = float(nm[3:]) if nm.startswith("d4x") else 1.0
        hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
        R.setdefault(nm, []).append(float(np.mean((p[-1].detach().cpu().double().numpy() - gt)**2)))
        extra = "" if lk.D4 is None else " часи dens %.0f birth %.0f trans %.0f с" % (lk.D4.t["dens"], lk.D4.t["birth"], lk.D4.t["trans"])
        eps = [float(ln.split("ε ")[1].split()[0]) for ln in dh.lines if "ε " in ln]
        print("dump %d %-6s %.4e  (%.0f с)%s  ε: %s" % (i, nm, R[nm][-1], time.time() - t0, extra, " ".join("%.3f" % e for e in eps)), flush=True)
        lk.D4 = None; lk.D21_HOOK = None; torch.cuda.empty_cache()
for k in modes: print("MEAN %-6s %.4e" % (k, np.mean(R[k])), flush=True)
