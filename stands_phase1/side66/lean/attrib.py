"""Атрибуція оракула D21 (×1.68 на reg) по наближеннях ланцюга: reg мінус ОДНЕ спрощення, для кожного MSE без і з істинною D21 (use-side).
Передбачення ДО прогону: розрив ×(оракул) на reg 1.68; той варіант, де розрив падає найбільше (до ≤1.3), і несе похибку D21.
Кандидати: -regen (щільна G) та -31 (κ₃₁ у aug). Шанс, що хоч один знімає ≥половину log-розриву: 40%."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
dt = torch.float32; R = {}
BASE = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
VAR = {"reg": {}, "-regen": "NOREGEN", "-B2": dict(shadow_B2=False), "-wk22": dict(no_wk22_fac=False), "-31": dict(aug_no_31=False),
       "-k211": dict(aug_no_k211=False), "-trunc": dict(leg_trunc=None), "dense": "DENSE",
       # 25.09: dense + ОДНЕ спрощення назад (без регенерації, щільна G)
       "d+B2": ("D", dict(shadow_B2=True)), "d+trunc": ("D", dict(leg_trunc={"rres": 16, "d21": 16})), "d+wk22": ("D", dict(no_wk22_fac=True)),
       "d+no31": ("D", dict(aug_no_31=True)), "d+nok211": ("D", dict(aug_no_k211=True)), "d+regen": ("DR", {})}
only = os.environ.get("VARS"); VAR = {k: v for k, v in VAR.items() if not only or k in only.split(",")}
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64); kp = np.load(lo.BANK.format(i))
    st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    for tag, v in VAR.items():
        for orc in (0, 1):
            lo.D21USE = orc; lk.D21_HOOK = lo.D21Hook(kp); lk.REGEN_STATE.clear()
            try:
                if v == "DENSE": p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True)
                elif isinstance(v, tuple) and v[0] == "D": p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, **v[1])
                elif isinstance(v, tuple) and v[0] == "DR": p = lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
                else:
                    kw = dict(BASE)
                    if v == "NOREGEN": hk = {}
                    else: kw.update(v); hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
                    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **kw)
                p = p[-1].detach().cpu().double().numpy(); R.setdefault((tag, orc), []).append(np.mean((p-gt)**2))
            except Exception as e: print(i, tag, orc, "ERR", type(e).__name__, e, flush=True); R.setdefault((tag, orc), []).append(np.nan)
    print("dump", i, " | ".join("%s %.3e/%.3e" % (t, R[(t, 0)][-1], R[(t, 1)][-1]) for t in VAR), flush=True)
b = np.nanmean(R[("reg", 0)]) if ("reg", 0) in R else np.nanmean(R[(next(iter(VAR)), 0)])
for t in VAR:
    m0 = np.nanmean(R[(t, 0)]); m1 = np.nanmean(R[(t, 1)])
    print("MEAN %-7s mse %.4e (×%.3f до reg) | з оракулом %.4e | розрив ×%.3f" % (t, m0, b/m0, m1, m0/m1), flush=True)
