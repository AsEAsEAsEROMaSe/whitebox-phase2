"""LOO за шаром народження джерела і «кожне друге»: цінність джерела проти його ціни (~2.17 u × 4 молодих шари + 1.1 u × старі шари).
Передбачення ДО прогону: жоден одиночний drop не дешевший за свою ціну (кожен ≥ +3% MSE); «кожне друге» ×2–4 MSE ⟹ ½ джерел мертво."""
import os, sys, numpy as np, torch, time
sys.argv = [sys.argv[0]] + (sys.argv[1:] if len(sys.argv) > 1 else ["0"])
import lean_oracle as lo, lean_k3_aug_gpu as lk
dumps = [int(x) for x in sys.argv[1].split(",")]
cfgs = [("base", set())] + [("drop%d" % b, {b}) for b in range(1, 13)] + [("odd", set(range(1, 16, 2))), ("even", set(range(2, 16, 2))), ("drop1-3", {1,2,3}), ("drop1-5", {1,2,3,4,5})]
if os.environ.get("CFGS"): cfgs = [c for c in cfgs if c[0] in os.environ["CFGS"].split(",")]
DROPSET = set(); LAYER = [0]
_lin = lk.K3Blocks.linear_
def lin(self, W):
    LAYER[0] += 1; _lin(self, W)   # linear_ раз на шар l=1..15; народження = l − age; видалення ДО dslices
    if DROPSET: self.sources[:] = [s for s in self.sources if (LAYER[0] - s["age"]) not in DROPSET]
lk.K3Blocks.linear_ = lin
statics = None; res = {}
for i in dumps:
    d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = d["final_means"].astype(np.float64)
    if statics is None: statics = lk.build_static_coefs(w.shape[1])
    lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
    DROPSET.clear(); LAYER[0] = 0; fit = lo.FitC(); lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, g_hook=fit)
    for name, S in cfgs:
        DROPSET.clear(); DROPSET.update(S); LAYER[0] = 0; t0 = time.perf_counter()
        pred = lk.lean_k3_predict(w, dtype=torch.float32, statics=statics, k4_aug=True, aug_no_k211=True, g_hook=lo.Replay(fit.coef),
                       g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
        pred = pred.detach().cpu().double().numpy(); mse = float(np.mean((pred[-1] - gt)**2)); res.setdefault(name, []).append(mse)
        print("dump%d %-8s mse %.4e (%.1fs)" % (i, name, mse, time.perf_counter() - t0), flush=True)
    lk.K3_PROBE = None
print("=== MEAN over", dumps)
base = np.mean(res["base"])
for name, _ in cfgs: print("%-8s %.4e  x%.3f" % (name, np.mean(res[name]), np.mean(res[name]) / base))
print("DONE_BIRTHDROP")
