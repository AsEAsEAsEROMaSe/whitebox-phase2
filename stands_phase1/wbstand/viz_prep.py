"""Дані для 3D-візуалізації: стан кожного семпла на кожному шарі, зведений трійками.

1024 = 341·3 + 1 ⟹ 341 повних трійок (остання координата відкидається).
v(h) = Σ_k (h_{3k}, h_{3k+1}, h_{3k+2})  ∈ R³   — «скласти всі 341 вектори»
"""
import os, sys, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW, NT = 16, 1024, 341
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 6144
WHICH = sys.argv[2] if len(sys.argv) > 2 else None

f = sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[0] if WHICH is None \
    else "/home/trim/wblogs/w2_%s.npy" % WHICH
nm = os.path.basename(f)[3:-4]
W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
z = cp.random.RandomState(seed=2026).standard_normal((NS, NW), dtype=cp.float32)

P = np.zeros((D, NS, 3), dtype=np.float32)      # 3D-проєкція стану
NRM = np.zeros((D, NS), dtype=np.float32)       # ‖h‖ семпла
COS = np.zeros((D, NS), dtype=np.float32)       # cos(h, середнє) — реальний, у 1024D
x = z
for i in range(D):
    x = cp.maximum(x @ Wg[i], 0.0)
    tri = x[:, :NT*3].reshape(NS, NT, 3).sum(axis=1)
    P[i] = cp.asnumpy(tri)
    n = cp.linalg.norm(x, axis=1); NRM[i] = cp.asnumpy(n)
    mu = x.mean(0); mu = mu/cp.linalg.norm(mu)
    COS[i] = cp.asnumpy((x @ mu)/cp.maximum(n, 1e-30))
np.savez_compressed("/home/trim/wblogs/viz3d.npz", name=nm, P=P, NRM=NRM, COS=COS,
                    mean=P.mean(axis=1))
print("готово:", nm, "· семплів", NS, "· шарів", D)
for i in (0, 1, 4, 8, 15):
    c = P[i] - P[i].mean(0)
    print("  ш%-2d  ‖середнє‖ %9.1f   розкид %8.2f   cos(h,μ) %.4f"
          % (i, np.linalg.norm(P[i].mean(0)), np.sqrt((c*c).sum(1).mean()), COS[i].mean()))
