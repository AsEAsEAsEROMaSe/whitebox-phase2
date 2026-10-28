"""Генерація «істини» МК для власних мереж фази 2 (27.09), стенд масштабу даних.
Мережа з сіду (gen_common.make_w), МК N семплів батчами B з центруванням і вибілюванням кожного батчу (точна одинична
коваріація входу; у фазі 1 вибілювання давало ×2.34 дисперсії). Зберігаємо all_layer_means (16,1024) f64 і Var виходу.
Шум істини ≈ avg_var/N_eff: при N=4e6 без вибілювання ~2e-8 (= MSE ланцюга); з вибілюванням менше. Для навчання
шумна, але незміщена ціль придатна; вирок лише на офіційних 100 з істиною 1e9.
Виклик: START END N. Чекпоінт = файл на мережу (atomic), пропуск наявних, 3 збої поспіль ⟹ вихід."""
import os, sys, time, math, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean"); from gen_common import make_w, L, N as n
OUT = "/home/trim/wb2data/gen/truth_%05d.npz"
start, end, NS = int(sys.argv[1]), int(sys.argv[2]), int(float(sys.argv[3])); B = 65536
dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = False
fails = 0
for i in range(start, end):
    if os.path.exists(OUT % i): continue
    try:
        t0 = time.time(); W = make_w(i, dev)
        acc = torch.zeros(L, n, dtype=torch.float64, device=dev); acc2 = torch.zeros(n, dtype=torch.float64, device=dev)
        g = torch.Generator(device=dev); g.manual_seed(7_000_000 + i); nb = NS // B
        for b in range(nb):
            z = torch.randn(B, n, device=dev, generator=g); z -= z.mean(0)
            cov = (z.T @ z) / B; Lc = torch.linalg.cholesky(cov)
            z = torch.linalg.solve_triangular(Lc, z.T, upper=False).T   # cov(z) = I точно
            h = z
            for l in range(L):
                h = torch.relu(h @ W[l]); acc[l] += h.sum(0, dtype=torch.float64)
            acc2 += (h.double() ** 2).sum(0)
        Ntot = nb * B; alm = (acc / Ntot).cpu().numpy(); m2 = (acc2 / Ntot).cpu().numpy(); var = m2 - alm[-1] ** 2
        tmp = OUT % i + ".tmp.npz"; np.savez(tmp, alm=alm, var_final=var, N=Ntot, seed=i); os.replace(tmp, OUT % i)
        print("gen %05d  N %d  avg_var %.4f  noise_mse≈%.2e  %.0f с" % (i, Ntot, var.mean(), var.mean() / Ntot, time.time() - t0), flush=True)
        fails = 0
    except Exception as e:
        fails += 1; print("FAIL %05d %r" % (i, e), flush=True); torch.cuda.empty_cache()
        if fails >= 3: print("3 збої поспіль, вихід"); sys.exit(1)
print("DONE_GEN")
