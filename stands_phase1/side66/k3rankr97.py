"""k3rankr97 — залишок ранг-r наближення κ₃(ReLU) по шарах: число для вибору r.
Mode-1 усічений SVD (Еккарт-Янг) — нижня межа для симетричного CP, добрий орієнтир."""
import sys, numpy as np
sys.path.insert(0, ".")
from k3gate97 import K3Acc, W_PATH, NSUB
LAYERS, B, N = [1, 8, 16, 24, 31], 5000, 200000
Wall = np.load(W_PATH, mmap_mode="r")
print("залишок ‖T − T_r‖/‖T‖ для κ₃(ReLU), 64 нейрони, N=%d" % N)
for net in (0, 1):
    W = np.array(Wall[net]).astype(np.float32)
    rs = np.sort(np.random.default_rng(11).permutation(256)[:NSUB])
    acc = {l: K3Acc(NSUB) for l in LAYERS}; accH = {l: K3Acc(NSUB) for l in LAYERS}
    rng = np.random.default_rng(3000 + net)
    for i in range(N // B):
        a = rng.standard_normal((B, 256), dtype=np.float32)
        for l in range(max(LAYERS) + 1):
            h = a @ W[l]
            if l in acc:
                r_ = np.maximum(h[:, rs], 0.0); acc[l].add(r_)
                if i < (N // B) // 2: accH[l].add(r_)
            a = np.maximum(h, 0.0)
    print("\nМЕРЕЖА %d\n  шар |   r=1     r=2     r=4     r=8    r=16  |  шум МК" % net)
    for l in LAYERS:
        _, _, T = acc[l].cum3(); _, _, Th = accH[l].cum3()
        s = np.linalg.svd(T.reshape(NSUB, NSUB * NSUB), compute_uv=False)
        e = s ** 2; tot = e.sum()
        row = [np.sqrt(max(tot - e[:r].sum(), 0) / tot) for r in (1, 2, 4, 8, 16)]
        nz = np.linalg.norm(Th - T) / np.linalg.norm(T)
        print("  %4d | " % l + " ".join("%6.3f " % v for v in row) + " |  %.3f" % nz)
