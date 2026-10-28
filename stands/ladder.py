"""12.10.2026 (ідея юзера «драбина гірших методів»): рунги = дешевші/гірші варіанти ланцюга; чи лежить похибка топу на лінії рунг→топ
(екстраполяція з універсальним коефіцієнтом, LOO по мережах). Зберігає p_top, gt, p_рунг для дампів."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
dt = torch.float32; DUMPS = [int(x) for x in os.environ.get("DUMPS", "0").split(",")]; T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
def rung_mask_half(seed):
    def f(l):
        g = torch.Generator(device="cpu").manual_seed(seed * 1000 + l)
        return (torch.rand(N, generator=g) < 0.5).float().to("cuda")
    return f
for I in DUMPS:
    d = np.load(lo.DUMP.format(I)); w = d["weights"]; gt = d["all_layer_means"][-1].astype(np.float64); N = w.shape[1]
    st = lk.build_static_coefs(N); lk.WK_HOOK = None; lk.D21_HOOK = None; lk.K3_INJ = None; lk.D4 = None; lk.HUB_MASK = None; lk.K3_PROBE_POST = None
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.lean_k3_predict(w, dtype=dt, statics=st, k4_aug=True, g_hook=fit)
    hk = dict(g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre)
    def run(skip=None, arm=None, mask=None, bscale=1.0, reg=None):
        lk.SKIP_BIRTH = set(skip) if skip else set(); lk.ARM_DROP = arm; lk.HUB_MASK = mask; lk.BIRTH_SCALE = bscale; lk.BIRTH_SCALE_LMIN = 0
        lk.REGEN_STATE.clear(); torch.set_grad_enabled(False)
        p = lk.lean_k3_predict(w, dtype=dt, statics=st, **hk, **(reg or REG))
        return p[-1].double().cpu().numpy()
    out = dict(gt=gt)
    out["top"] = run(); base = ((out["top"] - gt) ** 2).mean(); log(f"d{I} top {base:.4e}")
    rungs = {
        "k2_allskip": dict(skip=range(16)),
        "skip_ge8": dict(skip=range(8, 16)),
        "skip_le7": dict(skip=range(0, 8)),
        "skip_odd": dict(skip=range(1, 16, 2)),
        "arm3": dict(arm=(3, ("ALL",))),
        "arm6": dict(arm=(6, ("ALL",))),
        "hubhalf_a": dict(mask=rung_mask_half(1)),
        "hubhalf_b": dict(mask=rung_mask_half(2)),
        "bscale09": dict(bscale=0.9),
        "bscale11": dict(bscale=1.1),
        "trunc4": dict(reg=dict(REG, leg_trunc={"rres": 4, "d21": 4})),
    }
    for nm, kw in rungs.items():
        try:
            p = run(**kw); out[nm] = p; e = p - gt; et = out["top"] - gt
            log(f"d{I} {nm:11s} MSE×{((e**2).mean()/base):7.2f} cos(e_rung,e_top)={np.dot(e,et)/np.linalg.norm(e)/np.linalg.norm(et):+.3f} cos(d,e_top)={np.dot(p-out['top'],et)/np.linalg.norm(p-out['top'])/np.linalg.norm(et):+.3f}")
        except Exception as ex:
            log(f"d{I} {nm} ERR {type(ex).__name__}: {str(ex)[:120]}")
    lk.SKIP_BIRTH = set(); lk.ARM_DROP = None; lk.HUB_MASK = None; lk.BIRTH_SCALE = 1.0
    np.savez(f"/home/trim/wb2data/ladder_{I}.npz", **out); log(f"d{I} saved")
print("DONE_LADDER")
