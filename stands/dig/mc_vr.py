"""16.10.2026: МК фази 2 при тій самій кількості прямих проходів N: звичайна, антитетична (z, −z), вибілювання (вибіркова Cov z := I
через Холеського), антитетична + вибілювання. R повторів, MSE проти істини банку (gt дампу). Вихід: множник проти звичайної МК.
Передбачення ДО прогону: антитетична ×1.0–1.3 (ReLU-мережа без біасів: f(−z) ≠ f(z), але на глибині t≈3.5 майже лінійна в масштабі),
вибілювання ×1.5–2.5 (фаза 1 ×2.34)."""
import sys, numpy as np, torch
dev = 'cuda'; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096; R = int(sys.argv[3]) if len(sys.argv) > 3 else 64
for i in [int(x) for x in sys.argv[1].split(",")]:
    d = np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz' % i)
    W = torch.tensor(d['weights'], dtype=torch.float32, device=dev); gt = torch.tensor(d['final_means'], dtype=torch.float64, device=dev)
    n = W.shape[1]; g = torch.Generator(device=dev).manual_seed(1000 + i)
    def fwd(z):
        h = z
        for l in range(W.shape[0]): h = torch.relu(h @ W[l])
        return h
    def white(z):
        zc = z - z.mean(0, keepdim=True); S = zc.T @ zc / z.shape[0]
        Lc = torch.linalg.cholesky(S.double()).float(); return torch.linalg.solve_triangular(Lc, zc.T, upper=False).T
    acc = {k: [] for k in ("plain", "anti", "white", "anti+white")}
    for r in range(R):
        z = torch.randn(N, n, generator=g, device=dev)
        acc["plain"].append(float(((fwd(z).double().mean(0) - gt)**2).mean()))
        zh = z[:N//2]; acc["anti"].append(float((((fwd(zh).double().mean(0) + fwd(-zh).double().mean(0))/2 - gt)**2).mean()))
        acc["white"].append(float(((fwd(white(z)).double().mean(0) - gt)**2).mean()))
        zw = white(zh); acc["anti+white"].append(float((((fwd(zw).double().mean(0) + fwd(-zw).double().mean(0))/2 - gt)**2).mean()))
    m = {k: np.mean(v) for k, v in acc.items()}
    print("dump%d N=%d R=%d | " % (i, N, R) + " · ".join("%s %.3e (×%.2f)" % (k, v, m["plain"]/v) for k, v in m.items()), flush=True)
