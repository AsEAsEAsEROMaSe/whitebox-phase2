import sys, time
import numpy as np
sys.path.append("/home/trim/gpupkg")
import cupy as cp
rng = np.random.default_rng(0)
A = (rng.standard_normal((1024, 1024)) @ np.diag(np.exp(-np.arange(1024) / 30.0)) @ rng.standard_normal((1024, 1024))).astype(np.float32)
k = 16
def t(f, rep=3):
    f(); cp.cuda.Device().synchronize(); t0 = time.time()
    for _ in range(rep): r = f()
    cp.cuda.Device().synchronize(); return (time.time() - t0) / rep, r
Ag = cp.asarray(A)
ts, (u0, s0, v0) = t(lambda: np.linalg.svd(A, full_matrices=False), 2)
print("numpy svd f32", ts, flush=True)
tg, _ = t(lambda: cp.linalg.svd(Ag, full_matrices=False), 1)
print("cupy gesvd f32", tg, flush=True)
def eig64():
    G = Ag.astype(cp.float64); G = G.T @ G
    w, V = cp.linalg.eigh(G)
    V = V[:, ::-1][:, :k]; s = cp.sqrt(cp.maximum(w[::-1][:k], 0))
    U = (Ag.astype(cp.float64) @ V) / s
    return U, s, V.T
te, (u1, s1, v1) = t(eig64)
print("eigh64", te, "s rel", float(cp.max(cp.abs(s1 - cp.asarray(s0[:k])) / cp.asarray(s0[:k]))), flush=True)
B0 = (u0[:, :k] * s0[:k]) @ v0[:k]; B1 = cp.asnumpy((u1 * s1) @ v1)
print("rank-k recon rel", np.max(np.abs(B0 - B1)) / np.max(np.abs(B0)))
def eig32():
    G = Ag.T @ Ag
    w, V = cp.linalg.eigh(G)
    V = V[:, ::-1][:, :k]; s = cp.sqrt(cp.maximum(w[::-1][:k], 0))
    return (Ag @ V) / s, s, V.T
te2, (u2, s2, v2) = t(eig32)
B2 = cp.asnumpy((u2 * s2) @ v2)
print("eigh32", te2, "recon rel", np.max(np.abs(B0 - B2)) / np.max(np.abs(B0)))
