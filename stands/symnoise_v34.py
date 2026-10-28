"""13.10.2026: чи має систему шум округлення v34. Кілька перестановок нейронів (шар 5, сіди 1..6; шари 2, 9, 13 по одній),
δ_k = pred_perm − pred_base. Питання: середнє δ по перестановках проти sd (систематична частина), corr(δ, e), corr між δ_k,
концентрація по нейронах, знак ΔMSE і його розклад 2⟨e,δ⟩+‖δ‖²."""
import sys, time, numpy as np, importlib.util
sys.path.insert(0, "/home/trim/wb2lean")
spec = importlib.util.spec_from_file_location("est", "/home/trim/wb2lean/estimator_v34_m12.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
from whestbench import SetupContext; from whestbench.domain import MLP
import flopscope.numpy as fnp
d = np.load("/home/trim/wb2data/dumps_p2/mlp_0000.npz"); w = d["weights"].astype(np.float64); gt = d["final_means"].astype(np.float64); n = w.shape[1]; L = w.shape[0]
def run(W):
    e = est.Estimator(); e.setup(SetupContext(width=n, depth=L, flop_budget=2**41, api_version="1", seed=0))
    return np.asarray(e.predict(MLP(width=n, depth=L, weights=[fnp.asarray(x) for x in W], seed=0, name="d0"), 2**41))[-1]
p0 = run(w); e0 = p0 - gt; out = {"p0": p0, "gt": gt}
cases = [(5, s) for s in range(1, 7)] + [(2, 11), (9, 12), (13, 13)]
for l, s in cases:
    perm = np.random.default_rng(s).permutation(n); W = w.copy(); W[l] = W[l][:, perm]; W[l+1] = W[l+1][perm, :]
    p = run(W); out[f"p_{l}_{s}"] = p; dl = p - p0
    print(f"ш.{l} сід {s}: ‖δ‖/‖e‖ {np.linalg.norm(dl)/np.linalg.norm(e0):.4f} corr(δ,e) {np.corrcoef(dl, e0)[0,1]:+.3f} ΔMSE {((p-gt)**2).mean()-(e0**2).mean():+.3e} = 2⟨e,δ⟩/n {2*(e0@dl)/n:+.3e} + ‖δ‖²/n {dl@dl/n:.3e}", flush=True)
    np.savez("/home/trim/wb2data/symnoise_v34.npz", **out)
print("DONE", flush=True)
