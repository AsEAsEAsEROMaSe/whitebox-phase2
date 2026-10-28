"""21.10.2026 white-box-f9: МНК по мережах для e2e-поправок (jac.py). Ціль: min Σ_k ‖p0_k − truth_k + J_k θ‖². Крос: K-fold по мережах.
Звіт: агрегатне base/після для own (in-sample стеля), крос, і окремо сімʼї D3 (θ[:9]) / g4 (θ[9:]); ridge на масштабі ознак."""
import glob, sys, numpy as np
fs = sorted(glob.glob("/home/trim/wblogs/jac/jac_n*.npz")); Z = [np.load(f) for f in fs]
R = [z["p0"] - z["truth"] for z in Z]; J = [z["J"] for z in Z]; n = len(Z)
def fit(idx, cols, lam):
    A = sum(J[i][:, cols].T @ J[i][:, cols] for i in idx); b = sum(J[i][:, cols].T @ R[i] for i in idx)
    s = np.sqrt(np.diag(A)) + 1e-30; As = A / s[:, None] / s[None, :]
    th = -np.linalg.solve(As + lam * np.eye(len(cols)), b / s) / s
    return th
def mse(i, cols, th): return float(np.mean((R[i] + J[i][:, cols] @ th) ** 2))
base = np.array([np.mean(r ** 2) for r in R])
for nm, cols in (("D3", list(range(9))), ("g4", list(range(9, 18))), ("both", list(range(18)))):
    for lam in (1e-4, 1e-2, 1e-1):
        th = fit(range(n), cols, lam); own = np.array([mse(i, cols, th) for i in range(n)])
        K = min(n, 8); cr = np.zeros(n)
        for f in range(K):
            te = [i for i in range(n) if i % K == f]; tr = [i for i in range(n) if i % K != f]
            tf = fit(tr, cols, lam); cr[te] = [mse(i, cols, tf) for i in te]
        print("%-4s lam %.0e n %d own ×%.4f cross ×%.4f (%d/%d better)" % (nm, lam, n, base.sum() / own.sum(), base.sum() / cr.sum(), (cr < base).sum(), n))
