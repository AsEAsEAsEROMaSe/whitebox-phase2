"""Референсна kprop k_max=4 (factor=True) на справжній мережі: MSE фінального шару проти істини 1e9. Порівняння: k_max=3 SIMPLE ≈ lean dense."""
import sys, os, time, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference"]
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
i = int(sys.argv[1]); K = int(sys.argv[2]); kind = sys.argv[3]; dt = torch.float64 if os.environ.get("F64", "1") == "1" else torch.float32
DEVN = os.environ.get("DEV", "cuda"); torch.set_default_device(DEVN); torch.set_default_dtype(dt)
d = np.load(os.environ.get("DUMPF", "/home/trim/wb2data/dumps_p2/mlp_%04d.npz") % i); w = d["weights"]; gt = d["final_means"].astype(np.float64); L, n = w.shape[0], w.shape[1]
mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(dtype=dt, device=DEVN)
for l in range(L): mlp.Ws[l].weight.data = torch.from_numpy(w[l].T.copy()).to(dtype=dt, device=DEVN)
K_in = {1: torch.zeros(n, dtype=dt), 2: torch.eye(n, dtype=dt)}
t0 = time.time()
torch.set_grad_enabled(False)
for l in range(L): mlp.Ws[l].weight.requires_grad_(False)
Kb = mlp_kprop(mlp, K_in, k_max=K, kind=Kind[kind], factor=True, output_all=True, up_to_layer=f"act{L - 1}")
mu = Kb[f"act{L-1}"][1].to_tensor().double().cpu().numpy()
print("dump%d k_max=%d %s mse_final %.4e t %.0fs peak %.1f GB" % (i, K, kind, np.mean((mu-gt)**2), time.time()-t0, (torch.cuda.max_memory_allocated()/1e9 if DEVN=="cuda" else 0.0)), flush=True)
