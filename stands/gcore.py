"""«Повна інформація без горла» (стенд 26.09): чи негаусовість передактивацій p_l фази 2 стискається у k домінантних
напрямків. (а) Спектр негаусовості: γ₃, γ₄ проєкцій на власні напрямки C (топ, середина, хвіст), випадкові, координатні.
(б) Замикання «ядро k напрямків точно (семпли s = U_kᵀ(p−a)), решта гаусова умовно з C_r = C − U_k Λ_k U_kᵀ»:
E[ReLU(p_a)] і E[ReLU(p_a)²] проти істини банку (N=1e8) як функція k; k=0 = чисте гаусове замикання.
Передбачення ДО прогону: негаусовість (γ₄) найбільша у топ-напрямках (не в хвості, як у фазі 1 при n=256), і k=16–64
знімає ≥70% похибки гаусового замикання середнього (шанс 40%)."""
import os, sys, time, numpy as np, torch
i = int(sys.argv[1]); layers = [int(x) for x in sys.argv[2].split(",")]; LOGN = int(os.environ.get("LOGN", "22")); K = 64
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
d = np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i); w = d["weights"]; L, n, _ = w.shape; W = torch.tensor(w, device=dev)
kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
SQ2PI = float(np.sqrt(2 * np.pi))
def phi(x): return torch.exp(-0.5 * x * x) / SQ2PI
def Phi(x): return 0.5 * torch.erfc(-x / 2 ** 0.5)
def relu_moments(mu, sig):
    t = mu / sig; p = phi(t); P = Phi(t)
    return sig * p + mu * P, (mu * mu + sig * sig) * P + mu * sig * p
for l in layers:
    a = torch.tensor(kp["pre_mean"][l], device=dev, dtype=torch.float64); M11 = torch.tensor(kp["pre_M11"][l], device=dev, dtype=torch.float64)
    C = M11 - torch.outer(a, a); C = 0.5 * (C + C.T)
    m_h = torch.tensor(kp["mean"][l], device=dev, dtype=torch.float64); m2_h = torch.tensor(kp["post_m2"][l], device=dev, dtype=torch.float64)
    lam, U = torch.linalg.eigh(C); lam = lam.flip(0); U = U.flip(1)
    pr = float(lam.sum() ** 2 / (lam * lam).sum()); top = lam[:K].sum() / lam.sum()
    print("dump %d шар %d: PR-ранг C %.1f, енергія топ-8/64/256: %.3f/%.3f/%.3f, медіана |t| %.2f" % (
        i, l, pr, float(lam[:8].sum() / lam.sum()), float(top), float(lam[:256].sum() / lam.sum()), float((a / C.diagonal().sqrt()).abs().median())), flush=True)
    # напрямки для спектра негаусовості
    g = torch.Generator(device=dev); g.manual_seed(7)
    R = torch.randn(n, 64, device=dev, dtype=torch.float64); R = R / R.norm(dim=0, keepdim=True)
    dirs = {"топ-64": U[:, :64], "середина 448..512": U[:, 448:512], "хвіст 960..1024": U[:, 960:1024], "випадкові 64": R, "нейрони 0..64": torch.eye(n, device=dev, dtype=torch.float64)[:, :64]}
    Dm = torch.cat(list(dirs.values()), 1).float()                      # (n, 320)
    dvar = torch.einsum("nd,nm,md->d", Dm.double(), C, Dm.double())        # дисперсії проєкцій
    acc3 = torch.zeros(Dm.shape[1], device=dev, dtype=torch.float64); acc4 = torch.zeros_like(acc3); acc2 = torch.zeros_like(acc3)
    N = 2 ** LOGN; CH = 2 ** 15; S = torch.empty((N, K), device=dev, dtype=torch.float32)
    Uk = U[:, :K].float(); af = a.float(); gen = torch.Generator(device=dev); gen.manual_seed(100 + i); t0 = time.time()
    for c in range(N // CH):
        z = torch.randn(CH, n, device=dev, generator=gen); p = z @ W[0]
        for ll in range(l): p = torch.relu(p) @ W[ll + 1]
        pc = p - af; S[c * CH:(c + 1) * CH] = pc @ Uk
        x = (pc @ Dm).double(); acc2 += (x * x).sum(0); acc3 += (x ** 3).sum(0); acc4 += (x ** 4).sum(0)
    v = acc2 / N; g3 = (acc3 / N) / v ** 1.5; g4 = (acc4 / N) / v ** 2 - 3
    off = 0
    for nm, Dd in dirs.items():
        k_ = Dd.shape[1]; print("   %-20s |γ₃| сер %.3f макс %.3f | γ₄ сер %+.3f макс %+.3f | дисперсія проєкції сер %.2e" % (
            nm, float(g3[off:off + k_].abs().mean()), float(g3[off:off + k_].abs().max()), float(g4[off:off + k_].mean()), float(g4[off:off + k_].max()), float(v[off:off + k_].mean())), flush=True); off += k_
    print("   семпли %d за %.0f с" % (N, time.time() - t0), flush=True)
    # (б) замикання з ядром k
    sig_full = C.diagonal().sqrt()
    e1_g, e2_g = relu_moments(a, sig_full)
    base1 = float((e1_g - m_h).norm() / m_h.norm()); base2 = float((e2_g - m2_h).norm() / m2_h.norm())
    print("   гаусове замикання (k=0): E[ReLU] віднос. похибка %.5f · E[ReLU²] %.5f" % (base1, base2), flush=True)
    for k in [1, 2, 4, 8, 16, 32, 64]:
        Cr_diag = (C.diagonal() - (U[:, :k] ** 2 * lam[:k]).sum(1)).clamp(min=1e-12); sig = Cr_diag.sqrt().float()
        Ukk = U[:, :k].float(); s1 = torch.zeros(n, device=dev, dtype=torch.float64); s2 = torch.zeros_like(s1)
        for c in range(0, N, CH):
            mu_q = af[None, :] + S[c:c + CH, :k] @ Ukk.T
            e1, e2 = relu_moments(mu_q, sig[None, :]); s1 += e1.double().sum(0); s2 += e2.double().sum(0)
        s1 /= N; s2 /= N
        print("   ядро k=%2d: E[ReLU] %.5f (×%.2f) · E[ReLU²] %.5f (×%.2f) · енергія ядра %.3f" % (
            k, float((s1 - m_h).norm() / m_h.norm()), base1 / max(float((s1 - m_h).norm() / m_h.norm()), 1e-12),
            float((s2 - m2_h).norm() / m2_h.norm()), base2 / max(float((s2 - m2_h).norm() / m2_h.norm()), 1e-12), float(lam[:k].sum() / lam.sum())), flush=True)
    # шум оцінки ядра: половини семплів при k=64
    del S; torch.cuda.empty_cache()
