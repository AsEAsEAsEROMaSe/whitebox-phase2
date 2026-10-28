import numpy as np, torch, sys
sys.path.insert(0, "/home/trim/wb2lean"); import lean_oracle as lo; lk = lo.lk
d = np.load(lo.DUMP.format(0)); w = d["weights"].copy(); gt = d["final_means"].astype(np.float64); n = w.shape[1]
st = lk.build_static_coefs(n); REG = dict(k4_aug=True, aug_no_k211=True, shadow_B2=True, no_wk22_fac=True, aug_no_31=True, leg_trunc={"rres": 16, "d21": 16})
def run(W):
    fit = lo.FitC(); lk.REGEN_STATE.clear(); lk.WK_HOOK = None; lk.D21_HOOK = None; torch.manual_seed(0)
    lk.lean_k3_predict(W, dtype=torch.float32, statics=st, k4_aug=True, g_hook=fit); torch.manual_seed(0)
    return lk.lean_k3_predict(W, dtype=torch.float32, statics=st, g_hook=lo.Replay(fit.coef), g_feed_hook=lo.approx_pre, g_use_hook=lo.approx_pre, **REG).detach().double().cpu().numpy()[-1]
p0 = run(w); e0 = p0 - gt; out = {"p0": p0, "gt": gt}
for l, s in [(5, s) for s in range(1, 7)] + [(2, 11), (9, 12), (13, 13)]:
    perm = np.random.default_rng(s).permutation(n); W = w.copy(); W[l] = W[l][:, perm]; W[l+1] = W[l+1][perm, :]
    p = run(W); out[f"p_{l}_{s}"] = p; dl = p - p0
    print(f"ш.{l} сід {s}: ‖δ‖/‖e‖ {np.linalg.norm(dl)/np.linalg.norm(e0):.5f} corr(δ,e) {np.corrcoef(dl, e0)[0,1]:+.3f} ΔMSE {((p-gt)**2).mean()-(e0**2).mean():+.3e} = 2⟨e,δ⟩/n {2*(e0@dl)/n:+.3e} + ‖δ‖²/n {dl@dl/n:.3e}", flush=True)
    np.savez("/home/trim/wb2data/symnoise_lean.npz", **out)
print("DONE")
