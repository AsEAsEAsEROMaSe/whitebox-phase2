import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; i = 0; L0 = int(os.environ.get("L0", "13"))
d = np.load(lo.DUMP.format(i)); w = d["weights"]; gt = torch.tensor(d["all_layer_means"][-1], dtype=torch.float64, device="cuda")
st = lk.build_static_coefs(w.shape[1]); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None
fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
def hook(l, D3, D21, live):
    if l == L0: torch.set_grad_enabled(True)
    return D3, D21, None
sc = torch.tensor(1.0001, device="cuda", requires_grad=True)
torch.set_grad_enabled(False); lk.D21_HOOK = hook; lk.BIRTH_SCALE = sc; lk.BIRTH_SCALE_LMIN = L0; lk.REGEN_STATE.clear()
with torch.autograd.detect_anomaly():
    p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **REG)
    loss = ((p[-1].double() - gt) ** 2).mean()
    with torch.enable_grad(): loss.backward()
print("grad", float(sc.grad)); print("DONE_GRAD3")
