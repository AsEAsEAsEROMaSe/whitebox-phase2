"""Дані для вкладки «нейрони»: значення × увага × сума топ-k найсхожіших.

Точка = НЕЙРОН. Схожість — за вхідними вагами `W_l[:,j]` («що нейрон слухає»).
  A = softmax_i( cos(w_j, w_i) / τ ),  діагональ виключена
  x_j = h_j                       власне значення
  y_j = Σ_i A[j,i]·h_i            увага по ВСІХ інших
  z_j = Σ_{топ-k A[j,·]} h_i      сума топ-k найсхожіших
τ=0.01 ⟹ ~12 ефективних сусідів (зміряно; τ=0.05 дає 694 = майже рівномірно).
"""
import os, sys, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW = 16, 1024
NSAMP = 64
TAU = 0.01
KS = [4, 16, 64]

f = sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[0]
nm = os.path.basename(f)[3:-4]
W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
z = cp.random.RandomState(seed=2026).standard_normal((6144, NW), dtype=cp.float32)

X = np.zeros((D, NSAMP+1, NW), np.float32)
Y = np.zeros((D, NSAMP+1, NW), np.float32)
Z = np.zeros((D, NSAMP+1, NW, len(KS)), np.float32)
T = np.zeros((D, NW), np.float32)

x = z
for L in range(D):
    p = x @ Wg[L]; x = cp.maximum(p, 0.0)
    T[L] = cp.asnumpy(p.mean(0)/cp.maximum(p.std(0), 1e-20))
    w = Wg[L]; wn = w/cp.linalg.norm(w, axis=0, keepdims=True)
    S = (wn.T @ wn).astype(cp.float32)
    cp.fill_diagonal(S, cp.float32(-1e9))
    A = cp.exp((S - S.max(1, keepdims=True))/TAU); A /= A.sum(1, keepdims=True)
    order = cp.argsort(S, axis=1)                       # зростання
    TOP = {k: order[:, -k:] for k in KS}
    H = cp.concatenate([x.mean(0, keepdims=True), x[:NSAMP]], 0)   # 0 = середнє
    X[L] = cp.asnumpy(H)
    Y[L] = cp.asnumpy(H @ A.T)
    for ki, k in enumerate(KS):
        idx = TOP[k]                                    # (NW, k)
        Z[L, :, :, ki] = cp.asnumpy(H[:, idx].sum(axis=2))
    del S, A, order, TOP
    cp.get_default_memory_pool().free_all_blocks()

np.savez_compressed("/home/trim/wblogs/viz_attn.npz", name=nm, X=X, Y=Y, Z=Z, T=T,
                    KS=np.array(KS), tau=TAU)
print("готово:", nm, "· шарів", D, "· семплів", NSAMP, "+ середнє")
for L in (0, 8, 15):
    print("  ш%-2d  x[σ]=%.3f  y[σ]=%.3f  z16[σ]=%.3f  мертвих (h=0) %.1f%%"
          % (L, X[L,0].std(), Y[L,0].std(), Z[L,0,:,1].std(), 100*(X[L,0] == 0).mean()))
