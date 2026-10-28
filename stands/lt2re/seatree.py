"""21.10.2026 white-box-f9 (ідея: море обчислюване для конкретної W): деревні діаграми зв'язного κ4 після ReLU для 4 різних
нейронів (зірка a3·a1³·ρρρ, ланцюжок a1·a2·a2·a1·ρρρ), згорнуті з W⁴ матмулами: star = 4 Σ_a W a3 v³, path = 12 uᵀρ̃u,
v = ρ̃ (W∘a1), u = W∘a2∘v. Коефіцієнти Ерміта ReLU: a1 = σΦ(t), a2 = σφ(t), a3 = −σ t φ(t). Стан передактивацій шару l−1
(μ, C) — тут СПРАВЖНІЙ з МК (оракул формули), пізніше ланцюга. Ціль: чи пояснює це понейронну варіацію γ4 шару l.
ПЕРЕДБАЧЕННЯ: corr(sea/σ⁴, γ4_істина) 0.4…0.7; разом з aabb+diag R² 0.5…0.8. Шанс corr > 0.4: 40%.
usage: seatree.py <k> <l>"""
import sys, math, numpy as np
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
from math import erf
torch.set_grad_enabled(False)
k, L = int(sys.argv[1]), int(sys.argv[2]); N = 2 ** 23; B = 8192; dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(44000 + k)
S1 = torch.zeros(1024, dtype=torch.float64, device=dev); Q = torch.zeros(1024, 1024, dtype=torch.float64, device=dev)
P1 = torch.zeros(1024, dtype=torch.float64, device=dev); P2 = torch.zeros(1024, dtype=torch.float64, device=dev)
P3 = torch.zeros(1024, dtype=torch.float64, device=dev); P4 = torch.zeros(1024, dtype=torch.float64, device=dev)
X2 = torch.zeros(1024, 1024, dtype=torch.float64, device=dev)   # E[x_a^2 x_b^2] post l-1 for aabb
H1 = torch.zeros(1024, dtype=torch.float64, device=dev); H2 = torch.zeros(1024, dtype=torch.float64, device=dev)
H3 = torch.zeros(1024, dtype=torch.float64, device=dev); H4 = torch.zeros(1024, dtype=torch.float64, device=dev)
n = 0; sh = None; shq = None
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(L + 1):
        p = h @ W[l]
        if l == L - 1:
            if sh is None: sh = p.mean(0)
            z = (p - sh).double(); S1 += z.sum(0); Q += z.T @ z
        if l == L:
            if shq is None: shq = p.mean(0)
            q = (p - shq).double(); P1 += q.sum(0); P2 += (q ** 2).sum(0); P3 += (q ** 3).sum(0); P4 += (q ** 4).sum(0)
            break
        h = torch.relu(p)
    n += B
m1 = (S1 / n).cpu().numpy(); C = (Q / n).cpu().numpy() - np.outer(m1, m1)
mu = m1 + sh.double().cpu().numpy(); sd = np.sqrt(np.diag(C)); t = mu / sd
q1, q2, q3, q4 = [(x / n).cpu().numpy() for x in (P1, P2, P3, P4)]
var = q2 - q1 ** 2; c4 = q4 - 4 * q1 * q3 + 6 * q1 ** 2 * q2 - 3 * q1 ** 4 - 3 * var ** 2; g4t = c4 / var ** 2
phi = np.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + np.vectorize(erf)(t / math.sqrt(2)))
a1, a2, a3 = sd * Phi, sd * phi, -sd * t * phi
rho = C / np.outer(sd, sd); np.fill_diagonal(rho, 0.0)
Wl = w[L].numpy().astype(np.float64)
V = rho @ (Wl * a1[:, None])                       # v_a for every column j
star = 4.0 * np.sum(Wl * a3[:, None] * V ** 3, axis=0)
U = Wl * a2[:, None] * V
path = 12.0 * np.sum(U * (rho @ U), axis=0)
sea = star + path
# Gaussian (pair-level) post kappa_{aabb} diag-free part from the same state: leading tree a-b double edge: (a2_a a2_b) ρ² ·2/2! ... use exact pair
# formula is not needed for the test: compare sea against the residual after the best fit on the variation
for nm, v in (("star", star), ("path", path), ("sea", sea)):
    print("net %d L%d corr(%s/σ⁴, γ4) %.4f  rms ratio %.3f" % (k, L, nm, np.corrcoef(v / var ** 2, g4t)[0, 1], np.std(v / var ** 2) / np.std(g4t)), flush=True)
print("γ4: mean %.4f sd %.4f" % (g4t.mean(), g4t.std()))
