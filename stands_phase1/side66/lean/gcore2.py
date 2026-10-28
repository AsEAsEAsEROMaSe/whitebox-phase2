"""Пломба шуму для gcore (27.09): чи вирок «ядро k=64 знімає ≤45%» обмежений шумом семплів ядра і банку.
Один прохід: для всіх k одночасно, дві половини семплів (парні/непарні чанки) ⟹ шум = ‖h1−h2‖/(2‖m‖);
плюс пряма МК тих самих семплів (k=n) як підлога. LOGN задає N. Передбачення: при N×4 залишок k=64 падає
(шанс 60%), тобто вчорашній вирок ⛔ був шумовим."""
import os, sys, time, numpy as np, torch
i = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(",")]; LOGN = int(os.environ.get("LOGN", "22"))
KS = [0, 4, 16, 64, 128, 256]
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); w = d["weights"]; L, n, _ = w.shape; W = torch.tensor(w, device=dev)
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
SQ2PI = float(np.sqrt(2 * np.pi))
def phi(x): return torch.exp(-0.5 * x * x) / SQ2PI
def Phi(x): return 0.5 * torch.erfc(-x / 2 ** 0.5)
def relu_moments(mu, sig):
    t = mu / sig; p = phi(t); P = Phi(t)
    return sig * p + mu * P, (mu * mu + sig * sig) * P + mu * sig * p
rel = lambda x, y: float((x - y).norm() / y.norm())
for l in layers:
    a = torch.tensor(kp["pre_mean"][l], device=dev, dtype=torch.float64); M11 = torch.tensor(kp["pre_M11"][l], device=dev, dtype=torch.float64)
    C = M11 - torch.outer(a, a); C = 0.5 * (C + C.T)
    m_h = torch.tensor(kp["mean"][l], device=dev, dtype=torch.float64); m2_h = torch.tensor(kp["post_m2"][l], device=dev, dtype=torch.float64)
    lam, U = torch.linalg.eigh(C); lam = lam.flip(0); U = U.flip(1)
    N = 2 ** LOGN; CH = 2 ** 14; af = a.float(); gen = torch.Generator(device=dev); gen.manual_seed(100 + i); t0 = time.time()
    Kmax = max(KS); Uk = U[:, :Kmax].float()
    sigs = {}
    for k in KS:
        Cr = (C.diagonal() - (U[:, :k] ** 2 * lam[:k]).sum(1)).clamp(min=1e-12); sigs[k] = Cr.sqrt().float()
    acc = {k: [torch.zeros(n, device=dev, dtype=torch.float64) for _ in range(4)] for k in KS}   # s1_h0, s2_h0, s1_h1, s2_h1
    mc = [torch.zeros(n, device=dev, dtype=torch.float64) for _ in range(4)]
    for c in range(N // CH):
        h = c % 2
        z = torch.randn(CH, n, device=dev, generator=gen); p = z @ W[0]
        for ll in range(l): p = torch.relu(p) @ W[ll + 1]
        r = torch.relu(p); mc[2 * h] += r.double().sum(0); mc[2 * h + 1] += (r * r).double().sum(0)
        S = (p - af) @ Uk
        for k in KS:
            mu_q = af[None, :] + (S[:, :k] @ Uk[:, :k].T if k else 0.0)
            e1, e2 = relu_moments(mu_q, sigs[k][None, :]); acc[k][2 * h] += e1.double().sum(0); acc[k][2 * h + 1] += e2.double().sum(0)
    Nh = N // 2
    print("dump %d шар %d: N=%d за %.0f с; банк n_samples=%d" % (i, l, N, time.time() - t0, int(kp["n_samples"])), flush=True)
    mc1 = (mc[0] + mc[2]) / N; mc2 = (mc[1] + mc[3]) / N
    print("   пряма МК (k=n): E[ReLU] %.2e (шум половин %.2e) · E[ReLU²] %.2e (шум %.2e)" % (
        rel(mc1, m_h), float((mc[0] - mc[2]).norm() / Nh / 2 / m_h.norm()), rel(mc2, m2_h), float((mc[1] - mc[3]).norm() / Nh / 2 / m2_h.norm())), flush=True)
    for k in KS:
        s1 = (acc[k][0] + acc[k][2]) / N; s2 = (acc[k][1] + acc[k][3]) / N
        n1 = float((acc[k][0] - acc[k][2]).norm() / Nh / 2 / m_h.norm()); n2 = float((acc[k][1] - acc[k][3]).norm() / Nh / 2 / m2_h.norm())
        print("   ядро k=%3d: E[ReLU] %.2e (шум %.2e, чисте %.2e) · E[ReLU²] %.2e (шум %.2e, чисте %.2e) · енергія %.3f" % (
            k, rel(s1, m_h), n1, max(rel(s1, m_h) ** 2 - n1 ** 2, 0) ** 0.5, rel(s2, m2_h), n2, max(rel(s2, m2_h) ** 2 - n2 ** 2, 0) ** 0.5, float(lam[:k].sum() / lam.sum())), flush=True)
print("DONE_GCORE2")
