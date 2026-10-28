"""«Git-лог» ланцюга (питання юзера 14.10): кожне число фінального середнього як гілка, кожне джерело κ₃ (народження на шарі b) як коміт.
Серії на одному дампі (lean REG, fp32, без вікового конфайнменту):
  none          усі джерела видалено ⟹ m_G (база без κ₃)
  cum<=B        лишити джерела з народженням ≤ B (B=0..14; джерело b народжується наприкінці шару b, перенос на шарах b+1..15, вік a = l − b) ⟹ кумулятивний лог; коміт Δ_B = m(cum≤B) − m(cum≤B−1)
  only B<=a     лишити ЛИШЕ джерело b=B і лише до віку a (a=1..15−B) ⟹ профіль доставки: d_a = m(B,≤a) − m(B,≤a−1)
  single B      лише джерело B на весь вік (для адитивності Σ_B single_B проти m(cum≤15) − m_G)
Видалення в linear_ (після інкременту віку; народження = l − age), як у birthdrop.py (пломба: base відтворюється).
Зберігає pred усіх шарів (16,1024) кожної конфігурації і mu/var живого стану бази → wb2data/gitlog_<dump>.npz.
Передбачення ДО прогону: (1) base = 2.19e-8 біт у біт з birthdrop; (2) адитивність ≤1% ‖Σ‖; (3) ‖Δ_B‖ росте з B (молоді коміти
більші), кожен ≳ ‖e‖; (4) доставки одного джерела на різних віках майже ортогональні між собою (cos ≤0.3) і R² «решта ∝ перша» ≤0.2
⟹ перенос скаляром не замінити (шанс важеля ціни 10–15%)."""
import os, sys, time, numpy as np, torch
_argv = sys.argv; sys.argv = sys.argv[:1]
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
sys.argv = _argv
dump = int(sys.argv[1]); BIRTHS = [int(x) for x in os.environ.get("BIRTHS", "0,3,6,9,12").split(",")]
dt = torch.float64 if os.environ.get("DT", "32") == "64" else torch.float32
OUT = "/home/trim/wb2data/gitlog_%d%s.npz" % (dump, "_f64" if dt == torch.float64 else "")
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
RULE = {"keep": None}; LAYER = [0]
_lin = lk.K3Blocks.linear_
def lin(self, W):
    LAYER[0] += 1; _lin(self, W)
    if RULE["keep"] is not None:
        l = LAYER[0]
        if os.environ.get("REMOVE") == "1": self.sources[:] = [s for s in self.sources if RULE["keep"](l - s["age"], s["age"])]
        else:   # занулення ніг (усі члени мультилінійні по ногах одного джерела ⟹ тотожно видаленню; порожній список ламає dslices)
            for s in self.sources:
                if not RULE["keep"](l - s["age"], s["age"]):
                    for key in self.KEYS:
                        if s.get(key) is not None: s[key] = torch.zeros_like(s[key])
lk.K3Blocks.linear_ = lin
class Live:
    def __init__(self): self.mu = {}; self.var = {}
    def __call__(self, l, D3, D21, live):
        self.mu[l] = live["mu"].detach().double().cpu().numpy(); self.var[l] = live["var"].detach().double().cpu().numpy(); return D3, D21, None
d = np.load(lo.DUMP.format(dump)); w = d["weights"]; gt = d["final_means"].astype(np.float64); alm = d["all_layer_means"].astype(np.float64)
st = lk.build_static_coefs(w.shape[1]); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_PROBE = None
RULE["keep"] = None; LAYER[0] = 0; fit = lo.FitC(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
cfgs = [("base", None), ("none", lambda b, a: False)]
cfgs += [("cum<=%d" % B, (lambda B: (lambda b, a: b <= B))(B)) for B in range(0, 15)]
cfgs += [("single%d" % B, (lambda B: (lambda b, a: b == B))(B)) for B in range(0, 15)]
for B in BIRTHS:
    cfgs += [("only%d<=%d" % (B, A), (lambda B, A: (lambda b, a: b == B and a <= A))(B, A)) for A in range(1, 16 - B)]
if os.environ.get("CFGS"): cfgs = [c for c in cfgs if c[0] in os.environ["CFGS"].split(",")]
res = {}; live = Live(); t00 = time.perf_counter()
for name, keep in cfgs:
    RULE["keep"] = keep; LAYER[0] = 0; lk.REGEN_STATE.clear(); lk.D21_HOOK = live if name == "base" else None; t0 = time.perf_counter()
    try:
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG)
        p = p.detach().cpu().double().numpy(); res[name] = p; mse = float(np.mean((p[-1] - gt) ** 2))
        print("dump%d %-12s mse %.4e  ‖m−base‖ %.3e  (%.1fs)" % (dump, name, mse, float(np.linalg.norm(p[-1] - res["base"][-1])) if "base" in res else 0.0, time.perf_counter() - t0), flush=True)
    except Exception as e:
        print("dump%d %-12s ERR %s: %s" % (dump, name, type(e).__name__, str(e)[:200]), flush=True)
    lk.D21_HOOK = None
np.savez(OUT, gt=gt, alm=alm, names=np.array(list(res)), preds=np.stack([res[k] for k in res]),
         mu=np.stack([live.mu[l] for l in sorted(live.mu)]), var=np.stack([live.var[l] for l in sorted(live.var)]), mu_layers=np.array(sorted(live.mu)))
print("DONE_GITLOG dump%d %d cfgs %.0fs -> %s" % (dump, len(res), time.perf_counter() - t00, OUT), flush=True)
