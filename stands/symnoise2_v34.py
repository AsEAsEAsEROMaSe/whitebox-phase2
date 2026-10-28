"""13.10.2026: чи шум це шум. (a) та сама мережа, інший сід range finder (seed 1,2): δ_seed; (b) та сама мережа, ваги × (1+1e-7·N(0,1)) (нижче роздільності fp32): δ_eps; порівняти з δ_perm ≈ 0.09‖e‖."""
import sys, numpy as np, importlib.util
sys.path.insert(0, "/home/trim/wb2lean")
spec = importlib.util.spec_from_file_location("est", "/home/trim/wb2lean/estimator_v34_m12.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
from whestbench import SetupContext; from whestbench.domain import MLP
import flopscope.numpy as fnp
d = np.load("/home/trim/wb2data/dumps_p2/mlp_0000.npz"); w = d["weights"].astype(np.float64); gt = d["final_means"].astype(np.float64); n = w.shape[1]; L = w.shape[0]
def run(W, seed=0):
    e = est.Estimator(); e.setup(SetupContext(width=n, depth=L, flop_budget=2**41, api_version="1", seed=seed))
    return np.asarray(e.predict(MLP(width=n, depth=L, weights=[fnp.asarray(x) for x in W], seed=0, name="d0"), 2**41))[-1]
s = np.load("/home/trim/wb2data/symnoise_v34.npz"); p0 = s["p0"]; e0 = p0 - gt
def rep(tag, p):
    dl = p - p0; print(f"{tag}: ‖δ‖/‖e‖ {np.linalg.norm(dl)/np.linalg.norm(e0):.4f} corr(δ,e) {np.corrcoef(dl,e0)[0,1]:+.3f} ΔMSE {((p-gt)**2).mean()-(e0**2).mean():+.3e} ‖δ‖²/n {dl@dl/n:.3e}", flush=True); return p
out = {}
out["seed1"] = rep("сід range finder 1", run(w, 1)); out["seed2"] = rep("сід range finder 2", run(w, 2))
rng = np.random.default_rng(7); out["eps7"] = rep("ваги ×(1+1e-7 N)", run(w * (1 + 1e-7 * rng.standard_normal(w.shape))))
out["eps6"] = rep("ваги ×(1+1e-6 N)", run(w * (1 + 1e-6 * rng.standard_normal(w.shape))))
np.savez("/home/trim/wb2data/symnoise2_v34.npz", **out); print("DONE", flush=True)
