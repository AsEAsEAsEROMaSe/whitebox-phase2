"""Вкладка 3: точка = СЕМПЛ. значення × увага × сума топ-k найсхожіших семплів.

  схожість s↔r : cos(h_s, h_r) у 1024-вимірному стані
  🚨 сирий cos на глибині зібраний біля 0.92 з sd 0.013 ⟹ фіксована τ вироджується.
     Тому Z-НОРМУЄМО: logit = (cos − сер)/sd / τ'. Тоді шари порівнянні.
     [зміряно] τ'=0.2 → 4..15 ефективних сусідів · τ'=0.8 → 437..864 (майже рівномірно)
  власне значення семпла: proj = ⟨h_s, m̂⟩  (саме воно будує m: mean_s proj = ‖m‖)
                          norm = ‖h_s‖      (альтернатива)
  x = v_s ·  y = Σ_r A[s,r]·v_r ·  z = Σ_{топ-k} v_r
"""
import os, sys, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW, NSMP = 16, 1024, 6144
TAUS = [0.2, 0.4, 0.8]
KS   = [8, 32, 128]
CH   = 1024

f = sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[0]
nm = os.path.basename(f)[3:-4]
W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
x = cp.random.RandomState(seed=2026).standard_normal((NSMP, NW), dtype=cp.float32)

V = np.zeros((D, NSMP, 2), np.float32)
Y = np.zeros((D, NSMP, len(TAUS), 2), np.float32)
Z = np.zeros((D, NSMP, len(KS), 2), np.float32)
ST = np.zeros((D, 2 + len(TAUS)), np.float32)     # сер cos, sd cos, ефект.сусідів на τ

for L in range(D):
    x = cp.maximum(x @ Wg[L], 0.0)
    n = cp.linalg.norm(x, axis=1); hn = x/cp.maximum(n[:, None], 1e-30)
    mu = x.mean(0); mu = mu/cp.linalg.norm(mu)
    vals = cp.stack([x @ mu, n], 1)                       # (S,2)
    S = (hn @ hn.T).astype(cp.float32)
    cp.fill_diagonal(S, cp.float32(0.0))
    tot = float(S.sum()); cnt = NSMP*(NSMP-1)
    m1 = tot/cnt; m2 = (float((S*S).sum()) - 0.0)/cnt
    sd = float(np.sqrt(max(m2 - m1*m1, 1e-12)))
    ST[L, 0], ST[L, 1] = m1, sd
    Sz = (S - m1)/sd
    cp.fill_diagonal(Sz, cp.float32(-1e9))
    for ti, tp in enumerate(TAUS):
        A = cp.exp((Sz - Sz.max(1, keepdims=True))/tp); A /= A.sum(1, keepdims=True)
        ST[L, 2+ti] = float((1.0/(A*A).sum(1)).mean())
        Y[L, :, ti, :] = cp.asnumpy(A @ vals)
        del A
    for ki, k in enumerate(KS):
        idx = cp.argpartition(-Sz, k, axis=1)[:, :k]
        Z[L, :, ki, :] = cp.asnumpy(cp.take(vals, idx, axis=0).sum(1))
        del idx
    V[L] = cp.asnumpy(vals)
    del S, Sz, hn
    cp.get_default_memory_pool().free_all_blocks()
    print("шар %2d  cos %.4f±%.4f  ефект.сусідів %s" %
          (L, m1, sd, " ".join("%.0f" % ST[L, 2+i] for i in range(len(TAUS)))), flush=True)

np.savez_compressed("/home/trim/wblogs/viz_sattn.npz", name=nm, V=V, Y=Y, Z=Z, ST=ST,
                    TAUS=np.array(TAUS), KS=np.array(KS))
print("готово:", nm)
