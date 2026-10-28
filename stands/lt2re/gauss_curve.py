"""gauss_curve (20.10.2026, white-box-89, ідея юзера «спершу гауси, далі негаусове»): крива впливу негаусовості.
На ПЕРЕДактивації шару l справжній розподіл p_l замінюється гаусом N(m_l, C_l) з ТИМИ САМИМИ першими двома моментами
(МК-оцінки), далі МК до виходу. Delta_l = E_true[out] - E_gauss@l[out] = вплив усієї негаусовості (κ3, κ4, ...),
накопиченої до шару l, на середнє виходу. Порівняння з похибкою K=3 aug (truth - k3a): яку частку ловить кожен шар.
usage: gauss_curve.py <n> <L> <seed> <N_truth> <N_mc>     (n=256: істина з kscan; k3a рахується тут же)
Пломби: l = 0: p_0 = z W_0 гаусів точно ⟹ Delta = шум (перевіряє код); l = L-1: Delta = похибка гаусового кроку
останнього шару при точних m, C. Шум різниці: sqrt(2 var / N_mc)."""
import sys, os, math, time
import numpy as np, torch
torch.set_grad_enabled(False)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind

n, L, seed, N, M = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4])), int(float(sys.argv[5]))
dev = "cuda"
rng = np.random.default_rng(seed)
W = [(rng.standard_normal((n, n)) * math.sqrt(2.0 / n)).astype(np.float32) for _ in range(L)]
z = np.load(os.environ.get("AW_DATA", "data") + f"/kscan_{n}_{L}_{seed}_{N}.npz")
gt = z["gt"]
Wt = [torch.from_numpy(w).to(dev) for w in W]
B = 65536

# 1) moments m_l, C_l of post-act h_l by MC (float64 accumulation)
g = torch.Generator(device=dev).manual_seed(seed + 4242)
S1 = [torch.zeros(n, dtype=torch.float64, device=dev) for _ in range(L)]
S2 = [torch.zeros(n, n, dtype=torch.float64, device=dev) for _ in range(L)]
done = 0
while done < M:
    h = torch.randn(B, n, generator=g, device=dev)
    for l in range(L):
        p = h @ Wt[l]                 # pre-activation of layer l: the object a Gaussian closure replaces
        pd = p.double()
        S1[l] += pd.sum(0); S2[l] += pd.T @ pd
        h = torch.relu(p)
    done += B
m = [s / done for s in S1]
C = [S2[l] / done - torch.outer(m[l], m[l]) for l in range(L)]

# 2) Gaussianize at layer l, propagate by MC
def gauss_from(l):
    ev, V = torch.linalg.eigh(C[l])
    R = (V * torch.sqrt(torch.clamp(ev, min=0.0))).float()
    mu = m[l].float()
    acc = torch.zeros(n, dtype=torch.float64, device=dev)
    gg = torch.Generator(device=dev).manual_seed(seed + 9000 + l)
    d = 0
    while d < M:
        x = torch.randn(B, n, generator=gg, device=dev)
        for sgn in (1.0, -1.0):          # antithetic in the Gaussian (mean exact for the linear part)
            h = torch.relu(mu + sgn * (x @ R.T))
            for k in range(l + 1, L):
                h = torch.relu(h @ Wt[k])
            acc += h.double().sum(0)
        d += 2 * B
    return (acc / d).cpu().numpy()

# 3) k3a prediction on the same net
mlp = MLP(hidden_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0).to(device=dev, dtype=torch.float64)
for i in range(L):
    mlp.Ws[i].weight.data = torch.from_numpy(W[i].T.copy()).to(device=dev, dtype=torch.float64).contiguous()
out = mlp_kprop(mlp, {1: torch.zeros(n, device=dev, dtype=torch.float64), 2: torch.eye(n, device=dev, dtype=torch.float64)},
                k_max=3, kind=Kind.AUGMENT, factor=True, output_all=True, up_to_layer=f"act{L-1}", output_d_max=1)
k3 = out[f"act{L-1}"][1].to_tensor().double().cpu().numpy()
e3 = gt[-1] - k3
print(f"seed {seed} n={n}: MSE k3a {np.mean(e3 ** 2):.3e}; MC M={M:.1e} noise MSE ~{2 * z['var_final'].mean() / M:.1e}", flush=True)
print(" l   |Delta|^2/n   Delta.e3/|e3|^2   MSE(e3 - Delta)", flush=True)
for l in range(L):
    t0 = time.time()
    dl = gt[-1] - gauss_from(l)
    print(f"{l:2d}   {np.mean(dl ** 2):.3e}     {np.dot(dl, e3) / np.dot(e3, e3):+.3f}          {np.mean((e3 - dl) ** 2):.3e}   ({time.time() - t0:.0f}s)", flush=True)
