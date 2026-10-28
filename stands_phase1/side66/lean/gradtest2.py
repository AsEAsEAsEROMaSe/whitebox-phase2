"""Пломба диференційовності з графом лише від шару L0 (26.09 ніч): grad вмикається у D21_HOOK на шарі L0.
Міряємо пік пам'яті, час backward, градієнт за BIRTH_SCALE (LMIN=L0) проти скінченної різниці."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; i = 0; L0 = int(os.environ.get("L0", "12"))
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda")
st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
def hook(l, D3, D21, live):
    if l == L0: torch.set_grad_enabled(True)
    return D3, D21, None
def run(scale, grad):
    torch.set_grad_enabled(False); lk.D21_HOOK = hook if grad else None
    lk.BIRTH_SCALE = scale; lk.BIRTH_SCALE_LMIN = L0; lk.REGEN_STATE.clear()
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    loss = ((p[-1].double() - gt) ** 2).mean(); torch.set_grad_enabled(False); return loss
sc = torch.tensor(1.0001, device="cuda", requires_grad=True)
torch.cuda.reset_peak_memory_stats(); t0 = time.time(); loss = run(sc, True); t1 = time.time()
print("L0=%d forward %.1f с loss %.6e grad %s пік %.2f ГБ" % (L0, t1 - t0, float(loss), loss.requires_grad, torch.cuda.max_memory_allocated() / 2**30), flush=True)
with torch.enable_grad(): loss.backward()
t2 = time.time(); print("backward %.1f с  d loss/d scale = %.6e  пік %.2f ГБ" % (t2 - t1, float(sc.grad), torch.cuda.max_memory_allocated() / 2**30), flush=True)
lm = float(run(0.9991, False)); lp = float(run(1.0011, False)); print("FD: (%.6e − %.6e)/0.002 = %.6e" % (lp, lm, (lp - lm) / 0.002), flush=True)
print("DONE_GRAD2")
