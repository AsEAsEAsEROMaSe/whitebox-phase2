"""orac22 MC (22.10.2026, white-box-b9): справжні зрізи κ4 передактивацій на кожному з 16 шарів мережі mini k:
(2,2) κ22_ij = E[x̃_i² x̃_j²] − v_i v_j − 2 C_ij²,  (3,1) κ31_ij = E[x̃_i³ x̃_j] − 3 v_i C_ij  (x̃ = p − μ, i ≠ j).
Два проходи з тими самими семплами (μ, потім центровані акумулятори), дві половини по N/2 для перевірки шуму.
Ваги як у run1/nnoff (SeedSequence(seed).spawn(3)[0]), p = h @ W (звірено з істиною). usage: mc22.py <k> <out.npz> [log2N]"""
import sys, os, time
import numpy as np, torch
k, out = int(sys.argv[1]), sys.argv[2]
N = 2 ** (int(sys.argv[3]) if len(sys.argv) > 3 else 22); B = 8192; NB = N // B; n, L = 1024, 16
dev = torch.device("cuda")
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
wss, _, _ = np.random.SeedSequence(int(t["mlp_seed"][k])).spawn(3)
rng = np.random.default_rng(wss); SCALE = (2.0 / n) ** 0.5
W = [torch.tensor((rng.standard_normal((n, n)) * SCALE).astype(np.float32), device=dev) for _ in range(L)]
t0 = time.time()

def batches():
    g = torch.Generator(device=dev).manual_seed(4242 + k)
    for b in range(NB):
        yield b, torch.randn(B, n, generator=g, device=dev)

mu = torch.zeros(L, n, dtype=torch.float64, device=dev)
with torch.no_grad():
    for b, z in batches():
        h = z
        for l in range(L):
            p = h @ W[l]; mu[l] += p.sum(0, dtype=torch.float64); h = torch.relu(p)
    mu /= N
    mu32 = mu.float()
    print("pass1 %.0fs" % (time.time() - t0), flush=True)
    S2 = torch.zeros(2, L, n, n, dtype=torch.float64, device=dev); S22 = torch.zeros_like(S2); S31 = torch.zeros_like(S2)
    for b, z in batches():
        hf = 0 if b < NB // 2 else 1
        h = z
        for l in range(L):
            p = h @ W[l]; x = p - mu32[l]; x2 = x * x
            S2[hf, l] += (x.T @ x).double(); S22[hf, l] += (x2.T @ x2).double(); S31[hf, l] += ((x2 * x).T @ x).double()
            h = torch.relu(p)
        if b % 128 == 0:
            print("pass2 %d/%d %.0fs" % (b, NB, time.time() - t0), flush=True)
Nh = N / 2
eye = torch.eye(n, dtype=torch.bool, device=dev)
res = {"mu": mu.cpu().numpy().astype(np.float32)}
for hf, tag in ((0, "A"), (1, "B")):
    C = S2[hf] / Nh; v = torch.diagonal(C, dim1=1, dim2=2)
    k22 = S22[hf] / Nh - v[:, :, None] * v[:, None, :] - 2 * C ** 2
    k31 = S31[hf] / Nh - 3 * v[:, :, None] * C
    k22[:, eye] = 0; k31[:, eye] = 0
    res["k22" + tag] = k22.float().cpu().numpy(); res["k31" + tag] = k31.float().cpu().numpy(); res["v" + tag] = v.float().cpu().numpy()
    if hf == 0:
        res["C"] = C.float().cpu().numpy()
np.savez(out + ".tmp.npz", **res); os.replace(out + ".tmp.npz", out)
for l in (4, 8, 12, 14):
    a, b2 = res["k22A"][l].ravel(), res["k22B"][l].ravel(); c, d = res["k31A"][l].ravel(), res["k31B"][l].ravel()
    print("ш%2d надійність половин: corr k22 %.3f  k31 %.3f | rms k22 %.2e k31 %.2e" % (
        l, np.corrcoef(a, b2)[0, 1], np.corrcoef(c, d)[0, 1], np.sqrt(np.mean(a * a)), np.sqrt(np.mean(c * c))), flush=True)
print("MCDONE %s %.0fs" % (out, time.time() - t0), flush=True)
