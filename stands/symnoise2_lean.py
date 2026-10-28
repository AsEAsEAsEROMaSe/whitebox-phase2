import numpy as np, torch, sys
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
d = np.load(lo.DUMP.format(0)); w = d["weights"].copy(); gt = d["final_means"].astype(np.float64); n = w.shape[1]
st = lk.build_static_coefs(n); REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
def run(W, seed=0):
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; torch.manual_seed(seed)
    lk.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True, g_hook=fit); torch.manual_seed(seed)
    return lk.lean_k3_predict(W, dtype=torch.float32, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG).detach().double().cpu().numpy()[-1]
p0 = run(w); e0 = p0 - gt
def rep(tag, p):
    dl = p - p0; print(f"{tag}: ‖δ‖/‖e‖ {np.linalg.norm(dl)/np.linalg.norm(e0):.5f} corr(δ,e) {np.corrcoef(dl,e0)[0,1]:+.3f} ΔMSE {((p-gt)**2).mean()-(e0**2).mean():+.3e}", flush=True)
rep("сід 1", run(w, 1)); rep("сід 2", run(w, 2)); rng = np.random.default_rng(7)
rep("ваги ×(1+1e-7 N)", run(w * (1 + 1e-7 * rng.standard_normal(w.shape)))); rep("ваги ×(1+1e-6 N)", run(w * (1 + 1e-6 * rng.standard_normal(w.shape)))); print("DONE")
