"""21.10.2026 white-box-f9: накопичене море. Дерева зв'язного κ4 після ReLU (зірка+ланцюжок) народжуються на кожному шарі s зі стану
передактивацій (μ_s, C_s) і переносяться лінійно до шару L: M_{s→L} = W_{s+1} diag(Φ_{s+1}) … W_L (перша гармоніка ReLU).
c4_sea_j(L) = Σ_s tree_s(M_{s→L}[:, j]). Стан — справжній з МК (оракул формули). Звіт: corr(Σ_{s≥L−m} …/σ⁴, γ4) для m=1..8.
ПЕРЕДБАЧЕННЯ: corr росте з m і насичується на 0.5…0.8. Шанс > 0.5: 35%.
usage: seaacc.py <k> <L>"""
import sys, math, numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
from math import erf
torch.set_grad_enabled(False)
k, L = int(sys.argv[1]), int(sys.argv[2]); N = 2 ** 22; B = 8192; dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(45000 + k)
S1 = torch.zeros(L + 1, 1024, dtype=torch.float64, device=dev); Q = torch.zeros(L, 1024, 1024, dtype=torch.float32, device=dev)
P = torch.zeros(4, 1024, dtype=torch.float64, device=dev); n = 0; sh = {}
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(L + 1):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        z = p - sh[l]
        if l < L:
            S1[l] += z.double().sum(0); Q[l] += z.T @ z
        else:
            q = z.double()
            for i in range(4): P[i] += (q ** (i + 1)).sum(0)
            break
        h = torch.relu(p)
    n += B
q1, q2, q3, q4 = [(P[i] / n).cpu().numpy() for i in range(4)]
var = q2 - q1 ** 2; c4 = q4 - 4 * q1 * q3 + 6 * q1 ** 2 * q2 - 3 * q1 ** 4 - 3 * var ** 2; g4t = c4 / var ** 2
Wd = W.double(); M = Wd[L].clone(); acc = torch.zeros(1024, dtype=torch.float64, device=dev); out = []
for s in range(L - 1, -1, -1):
    m1 = S1[s] / n; C = Q[s].double() / n - torch.outer(m1, m1)
    mu = m1 + sh[s].double(); sd = torch.sqrt(torch.diag(C)); t = mu / sd
    phi = torch.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + torch.erf(t / math.sqrt(2)))
    a1, a2, a3 = sd * Phi, sd * phi, -sd * t * phi
    rho = C / torch.outer(sd, sd); rho.fill_diagonal_(0.0)
    V = rho @ (M * a1[:, None]); U = M * a2[:, None] * V
    tree = 4.0 * torch.sum(M * a3[:, None] * V ** 3, 0) + 12.0 * torch.sum(U * (rho @ U), 0)
    acc += tree
    out.append((s, float(np.corrcoef(acc.cpu().numpy() / var ** 2, g4t)[0, 1]), float(np.corrcoef(tree.cpu().numpy() / var ** 2, g4t)[0, 1]),
                float(np.std(acc.cpu().numpy() / var ** 2) / g4t.std())))
    # transport one layer back: M_{s-1→L} = W_s diag(Φ_s) M_{s→L}   (Φ of pre-activations at layer s)
    M = Wd[s] @ (Phi[:, None] * M)
for s, ca, ct, r in out:
    print("net %d L%d born s=%2d: corr(Σ, γ4) %.4f  corr(this, γ4) %+.4f  rms Σ/γ4 %.3f" % (k, L, s, ca, ct, r), flush=True)
print("γ4 mean %.4f sd %.4f" % (g4t.mean(), g4t.std()))
