"""Змішані зрізи κ₅ (для народження D21): κ41[i,j]=κ(i,i,i,i,j), κ32[i,j]=κ(i,i,i,j,j) проти деревних ознак з півотом
x∈{i,j} (добутки зрізів κ₃ і κ₄, які ланцюг має). Центровані формули:
 κ21=E[X²Y] · κ31=E[X³Y]−3E[X²]E[XY] · κ22=E[X²Y²]−E[X²]E[Y²]−2E[XY]²
 κ41=E[X⁴Y]−6E[X²]E[X²Y]−4E[XY]E[X³] · κ32=E[X³Y²]−3E[X²]E[XY²]−6E[XY]E[X²Y]−E[Y²]E[X³]
Зберігає матриці для шарів LAYS, окремо половини A/B для стелі шуму. Аналіз: kmix_an.py."""
import sys, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]); N = 2**23; CH = 2**14; LAYS = [6, 9, 12, 15]
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev); L, n, _ = W.shape
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); a = torch.tensor(kp["pre_mean"], device=dev, dtype=torch.float32)
KEYS = ["11", "21", "12", "31", "22", "41", "32"]
M = {(h, l, k): torch.zeros(n, n, dtype=torch.float64, device=dev) for h in (0, 1) for l in LAYS for k in KEYS}
V = {(h, l): torch.zeros(6, n, dtype=torch.float64, device=dev) for h in (0, 1) for l in LAYS}
g = torch.Generator(device=dev); g.manual_seed(313 + i)
for c in range(N // CH):
    h = c % 2; z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(L):
        if l in LAYS:
            x1 = p - a[l]; x2 = x1*x1; x3 = x2*x1; x4 = x2*x2
            for k, (A, B) in {"11": (x1, x1), "21": (x2, x1), "12": (x1, x2), "31": (x3, x1), "22": (x2, x2), "41": (x4, x1), "32": (x3, x2)}.items():
                M[(h, l, k)] += (A.T @ B).double()
            q = torch.ones_like(x1, dtype=torch.float64); x1d = x1.double()
            for k in range(6): V[(h, l)][k] += q.sum(0); q = q*x1d
        if l == max(LAYS): break
        p = torch.relu(p) @ W[l+1]
out = {}
for h in (0, 1):
    Nh = float(V[(h, LAYS[0])][0][0])
    for l in LAYS:
        for k in KEYS: out["h%d_l%d_%s" % (h, l, k)] = (M[(h, l, k)]/Nh).float().cpu().numpy()
        out["h%d_l%d_v" % (h, l)] = (V[(h, l)]/Nh).cpu().numpy()
np.savez("/home/trim/wb2data/kmix_%d.npz" % i, **out); print("DONE_KMIX", flush=True)
