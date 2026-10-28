"""k3gen97b — загострена пломба на замкнену форму: сурогат на 2e6 семплів.
Шум МК падає в 3.2x; якщо формула вірна, розбіжність ПАДАЄ разом із шумом
(пломба масштабування, протокол п.4). Якщо стоїть — це СИСТЕМАТИКА формули."""
import sys, numpy as np
sys.path.insert(0, ".")
from k3gate97 import K3Acc, Phi, pr_and_top, W_PATH, NSUB
from k3gen97 import k3_generated

LAYER, B = 31, 5000
Wall = np.load(W_PATH, mmap_mode="r")
for net in (0, 1):
    W = np.array(Wall[net]).astype(np.float32)
    rs = np.sort(np.random.default_rng(11).permutation(256)[:NSUB])
    acc = K3Acc(NSUB); rng = np.random.default_rng(3000 + net)
    for _ in range(200000 // B):
        a = rng.standard_normal((B, 256), dtype=np.float32)
        for l in range(LAYER + 1):
            h = a @ W[l]
            if l == LAYER: acc.add(h[:, rs])
            a = np.maximum(h, 0.0)
    muh, Ch, _ = acc.cum3()
    s = np.sqrt(np.diag(Ch)); t = muh / s; rho = Ch / np.outer(s, s)
    ev, U = np.linalg.eigh(Ch); L = U * np.sqrt(np.clip(ev, 0, None))
    F = k3_generated(rho, s, t, K=4)
    print("\nМЕРЕЖА %d  мед|t|=%.3f" % (net, np.median(np.abs(t))))
    print("   N_сурогат |  шум МК  | ‖формула−вимір‖/‖вимір‖ | відношення похибка/шум")
    for NS in (200000, 800000, 3200000):
        accG, accH = K3Acc(NSUB), K3Acc(NSUB)
        rg = np.random.default_rng(555 + net)
        for i in range(NS // B):
            g = muh + rg.standard_normal((B, NSUB)) @ L.T
            r = np.maximum(g, 0.0); accG.add(r)
            if i < (NS // B) // 2: accH.add(r)
        _, _, Tg = accG.cum3(); _, _, Th = accH.cum3()
        nz = np.linalg.norm(Th - Tg) / np.linalg.norm(Tg)
        er = np.linalg.norm(F - Tg) / np.linalg.norm(Tg)
        print("   %9d |  %.4f  |        %.4f           |    %.2f" % (NS, nz, er, er / nz))
