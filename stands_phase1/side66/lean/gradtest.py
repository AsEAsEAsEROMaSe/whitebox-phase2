"""Пломба диференційовності ланцюга (26.09 ніч): autograd через lean_k3_predict за BIRTH_SCALE (скаляр-тензор), dump 0.
Міряємо: чи проходить backward, пам'ять GPU, час forward/backward, градієнт за скаляром."""
import sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; i = 0
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda")
st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None
with torch.no_grad():
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
torch.set_grad_enabled(True)
sc = torch.tensor(1.0001, device="cuda", requires_grad=True); lk.BIRTH_SCALE = sc; lk.BIRTH_SCALE_LMIN = 0
torch.cuda.reset_peak_memory_stats(); lk.REGEN_STATE.clear(); t0 = time.time()
p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
loss = ((p[-1].double() - gt) ** 2).mean(); t1 = time.time()
print("forward %.1f с, loss %.4e, requires_grad %s, пік пам'яті %.2f ГБ" % (t1 - t0, float(loss), loss.requires_grad, torch.cuda.max_memory_allocated() / 2**30), flush=True)
loss.backward(); t2 = time.time()
print("backward %.1f с, d loss/d BIRTH_SCALE = %.4e, пік пам'яті %.2f ГБ" % (t2 - t1, float(sc.grad), torch.cuda.max_memory_allocated() / 2**30), flush=True)
# пломба: скінченна різниця
with torch.no_grad():
    for s_ in (0.9991, 1.0011):
        lk.BIRTH_SCALE = float(s_); lk.REGEN_STATE.clear(); p2 = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG); print("  FD s=%.3f loss %.6e" % (s_, float(((p2[-1].double() - gt) ** 2).mean())), flush=True)
print("DONE_GRAD")
