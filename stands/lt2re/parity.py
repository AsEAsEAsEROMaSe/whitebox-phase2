"""19.10.2026 white-box-16: паритет LT2 у torch на дампах mini: raw MSE (pred+u15) / raw MSE (pred) має бути ≈0.87."""
import glob, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
from lt2torch import LT2
torch.set_num_threads(1)
dev = "cuda"
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
m = LT2("/home/trim/wb2lean/lt2re/lt2_shipped.npz", dev)


def wseed(s):
    wss, _, _ = np.random.SeedSequence(int(s)).spawn(3)
    rng = np.random.default_rng(wss)
    return [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]


a0 = a1 = 0.0
for f in sorted(glob.glob("/home/trim/wb2data/lt2dump/mini/mini_*.npz")):
    z = np.load(f); k = int(z["idx"])
    W = [torch.from_numpy(w).to(dev)[None] for w in wseed(z["seed"])]
    X = torch.from_numpy(z["X"]).to(dev)[None]; Phi = torch.from_numpy(z["Phi"]).to(dev)[None]
    with torch.no_grad():
        gs, us = m.run(X, Phi, W)
    tr = t["all_layer_means"][k][-1].astype(np.float64); p = z["pred"][-1].astype(np.float64)
    u = us[-1][0].double().cpu().numpy()
    r0 = np.mean((p - tr) ** 2); r1 = np.mean((p + u - tr) ** 2); a0 += r0; a1 += r1
    print("mini %d off %.4e on %.4e ratio %.4f" % (k, r0, r1, r1 / r0), flush=True)
print("aggregate ratio %.4f" % (a1 / a0))
