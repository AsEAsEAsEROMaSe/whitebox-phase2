"""Синтетична мережа n×L He-init (як фаза 2, без біасів) у форматі dumps_p2 + істина МК (E[relu(p_L)]) на GPU.
Використання: python3 synth256.py <n> <L> <seed> <log2 N>. Файл /home/trim/wb2data/synth/mlp_<n>_<seed>.npz."""
import sys, numpy as np, torch
n, L, seed, logN = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng(seed); w = (rng.standard_normal((L, n, n)) * np.sqrt(2.0 / n)).astype(np.float32)
dev = "cuda"; W = torch.tensor(w, device=dev); N = 2 ** logN; CH = 2 ** 16
acc = torch.zeros(n, dtype=torch.float64, device=dev); acc2 = torch.zeros(n, dtype=torch.float64, device=dev)
g = torch.Generator(device=dev); g.manual_seed(1000 + seed)
for c in range(N // CH):
    z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(1, L): p = torch.relu(p) @ W[l]
    r = torch.relu(p).double(); acc += r.sum(0); acc2 += (r * r).sum(0)
mean = (acc / N).cpu().numpy(); var = (acc2 / N - (acc / N) ** 2).cpu().numpy()
np.savez("/home/trim/wb2data/synth/mlp_%d_%d.npz" % (n, seed + 100 * (L != 16)), weights=w, final_means=mean, final_vars=var, n_samples=N)
print("n %d L %d seed %d N 2^%d: mean(final²) %.4f avg_var %.4f підлога MSE %.2e" % (n, L, seed, logN, float((mean**2).mean()), float(var.mean()), float(var.mean() / N)))
