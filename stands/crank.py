"""Гіпотеза «#1 несе стан O(n²r)»: чи C_l = діагональ + низький ранг? Банк kp_p2 (істинні M11), шари 1..15.
Похибка ||C_off − (U_r S_r U_rᵀ)_off||/||C_off|| для власного розкладу C після вирахування діагоналі (ітерація: D ← diag(C − L_r))."""
import sys, numpy as np
for i in [int(x) for x in sys.argv[1].split(",")]:
    kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
    for l in (1, 4, 8, 12, 15):
        a = kp["pre_mean"][l].astype(np.float64); C = kp["pre_M11"][l].astype(np.float64) - np.outer(a, a)
        n = C.shape[0]; off = ~np.eye(n, dtype=bool); nC = np.linalg.norm(C[off])
        rho = C / np.sqrt(np.outer(np.diag(C), np.diag(C))); mr = float(np.median(np.abs(rho[off])))
        out = []
        for r in (8, 16, 32, 64, 128, 256):
            D = np.diag(C).copy() * 0.0
            for _ in range(6):
                w, V = np.linalg.eigh(C - np.diag(D)); idx = np.argsort(-np.abs(w))[:r]
                Lr = (V[:, idx] * w[idx]) @ V[:, idx].T; D = np.diag(C - Lr)
            out.append("r%d %.3f" % (r, np.linalg.norm((C - Lr)[off]) / nC))
        print("dump%d l=%2d мед|ρ| %.3f ‖C_off‖/‖diag‖ %.2f | похибка позадіаг.: %s" % (i, l, mr, nC / np.linalg.norm(np.diag(C)), " ".join(out)), flush=True)
