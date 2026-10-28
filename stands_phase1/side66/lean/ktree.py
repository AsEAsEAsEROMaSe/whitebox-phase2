"""Ідея юзера: міжпорядкові зв'язки, що залежать лише від архітектури (деревні 1/n-зв'язки). Понейронно, передактивації
кожного шару: κ₂..κ₅ з N семплів. Моделі: κ₄ ≈ a·κ₃²/κ₂ (+b·κ₂²·const?) і κ₅ ≈ c·κ₃κ₄/κ₂ (+d·κ₃κ₂). МНК по нейронах шару:
R² і коефіцієнт; універсальність = розкид коефіцієнта між шарами/мережами. 🔒 пломба шуму: половини вибірки (A/B) —
R² між κ₅(A) і κ₅(B) = стеля. Критерій ДО прогону: R²(κ₅ | κ₃κ₄/κ₂) ≥ 0.7 від стелі і розкид c < 10%; шанс 25%."""
import sys, numpy as np, torch
dev = "cuda"; i = int(sys.argv[1]); N = 2**23; CH = 2**15
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); W = torch.tensor(d["weights"], device=dev); L, n, _ = W.shape
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i); a = torch.tensor(kp["pre_mean"], device=dev, dtype=torch.float32)
S = torch.zeros(2, L, 6, n, dtype=torch.float64, device=dev)   # половина × шар × степінь 0..5 центрованих
g = torch.Generator(device=dev); g.manual_seed(91 + i)
for c in range(N // CH):
    h = c % 2; z = torch.randn(CH, n, device=dev, generator=g); p = z @ W[0]
    for l in range(L):
        pc = (p - a[l]).double(); q = torch.ones_like(pc)
        for k in range(6): S[h, l, k] += q.sum(0); q = q*pc
        if l < L-1: p = torch.relu(p) @ W[l+1]
def cum(s):
    m = [s[k]/s[0] for k in range(6)]; m1 = m[1]
    # центрування залишку (a з банку ≈ середнє): повні кумулянти з сирих центрованих моментів
    mu2 = m[2]-m1**2; mu3 = m[3]-3*m1*m[2]+2*m1**3; mu4 = m[4]-4*m1*m[3]+6*m1**2*m[2]-3*m1**4
    mu5 = m[5]-5*m1*m[4]+10*m1**2*m[3]-10*m1**3*m[2]+4*m1**5
    return mu2, mu3, mu4 - 3*mu2**2, mu5 - 10*mu3*mu2
def fit(y, X):
    sol = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]; return 1 - float(((y - X@sol)**2).sum()/((y-y.mean())**2).sum()), [float(v) for v in sol]
print("dump%d  шар | стеля R²(κ₅ A↔B) | κ₄~κ₃²/κ₂: R² a | κ₅~κ₃κ₄/κ₂: R² c | κ₅~[κ₃κ₄/κ₂, κ₃κ₂]: R² c d | κ₅~κ₃³/κ₂²: R² | мед γ₃ γ₄" % i)
for l in range(1, L):
    k2, k3, k4, k5 = cum(S[0, l] + S[1, l]); _, _, _, k5a = cum(S[0, l]); _, _, _, k5b = cum(S[1, l])
    ceil = float(torch.corrcoef(torch.stack([k5a, k5b]))[0, 1])
    r4, a4 = fit(k4, (k3*k3/k2)[:, None]); r5, c5 = fit(k5, (k3*k4/k2)[:, None]); r52, cd = fit(k5, torch.stack([k3*k4/k2, k3*k2], 1)); r53, _ = fit(k5, (k3**3/k2**2)[:, None])
    print("   l=%2d | %.3f | %.3f %.3f | %.3f %.3f | %.3f %.3f %.3f | %.3f | %.3f %.3f" % (l, ceil, r4, a4[0], r5, c5[0], r52, cd[0], cd[1], r53,
          float((k3/k2**1.5).median()), float((k4/k2**2).median())), flush=True)
print("DONE_KTREE")
