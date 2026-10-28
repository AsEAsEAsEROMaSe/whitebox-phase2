"""13.10.2026: пломба симетрії W₀→−W₀ на ПРОДАКШН-оцінювачі v34 (flopscope, fp32, Штрассен), дамп 0; контроль перестановка ш.5."""
import sys, time, numpy as np, importlib.util
sys.path.insert(0, "/home/trim/wb2lean")
spec = importlib.util.spec_from_file_location("est", "/home/trim/wb2lean/estimator_v34_m12.py"); est = importlib.util.module_from_spec(spec); spec.loader.exec_module(est)
from whestbench import SetupContext; from whestbench.domain import MLP
import flopscope as flops, flopscope.numpy as fnp
d = np.load("/home/trim/wb2data/dumps_p2/mlp_0000.npz"); w = d["weights"].astype(np.float64); gt = d["final_means"].astype(np.float64); n = w.shape[1]; L = w.shape[0]
def run(W):
    e = est.Estimator(); e.setup(SetupContext(width=n, depth=L, flop_budget=2**41, api_version="1", seed=0))
    mlp = MLP(width=n, depth=L, weights=[fnp.asarray(x) for x in W], seed=0, name="d0")
    t0 = time.time(); p = np.asarray(e.predict(mlp, 2**41)); return p, time.time()-t0
p0, t = run(w); mse = lambda p: float(((p[-1]-gt)**2).mean()); print(f"база MSE {mse(p0):.6e} ({t:.0f} с)", flush=True)
w1 = w.copy(); w1[0] = -w1[0]; p1, _ = run(w1); print(f"W₀→−W₀: MSE {mse(p1):.6e}  max|Δ|/max {np.abs(p1[-1]-p0[-1]).max()/np.abs(p0[-1]).max():.2e}", flush=True)
perm = np.random.default_rng(1).permutation(n); w2 = w.copy(); w2[5] = w2[5][:, perm]; w2[6] = w2[6][perm, :]; p2, _ = run(w2)
print(f"перестановка ш.5: MSE {mse(p2):.6e}  max|Δ|/max {np.abs(p2[-1]-p0[-1]).max()/np.abs(p0[-1]).max():.2e}", flush=True)
