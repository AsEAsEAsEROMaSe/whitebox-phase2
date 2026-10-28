"""22.10.2026 a0: розбір t_k22: частка залишку c4t − g4, яку знімає підстановка справжніх K22-рядк. сум / K4v у народження."""
import sys, numpy as np
d = np.load(sys.argv[1]); W = d["W"].astype(np.float64); n = 1024
cA, cI = 6.0 / (n + 4.0), -3.0 / ((n + 2.0) * (n + 4.0))
def tr(x, h=None):
    return (d["%s_%d_0" % (x, h)] + d["%s_%d_1" % (x, h)]) / 2
print("шар  corr(k22 ланц,іст) масшт  corr(K4v)  | c4t~g4 corr slope | частка залишку: K22  K4v  обидва | шум c4t між половинами / залишок")
for L in range(1, 15):
    if "k22_%d_a" % L not in d.files or "g4_%d_a" % (L + 1) not in d.files: continue
    k22c, k4c = d["k22_%d_a" % L].astype(np.float64), d["k22_%d_b" % L].astype(np.float64)
    k22t, k4t = tr("k22t", L), tr("k4t", L)
    g4 = d["g4_%d_a" % (L + 1)].astype(np.float64); c4t = tr("c4t", L + 1)
    W2 = W[L + 1] ** 2                       # p = h @ W, W[in, out]
    def dg(dl): return 2.0 * (cA * (dl @ W2) + cI * dl.sum() * W2.sum(0))
    e0 = c4t - g4
    sh = []
    for dl in (k22t - k22c, k4t - k4c, (k22t - k22c) + (k4t - k4c)):
        sh.append(1 - np.sum((e0 - dg(dl)) ** 2) / np.sum(e0 ** 2))
    noise = np.sum(((d["c4t_%d_0" % (L + 1)] - d["c4t_%d_1" % (L + 1)]) / 2) ** 2) / np.sum(e0 ** 2)
    c = lambda a, b: np.corrcoef(a, b)[0, 1]
    print("%2d   %.3f  %.3f   %.3f   |  %.3f %.3f  |  %.3f  %.3f  %.3f  | %.3f" % (L, c(k22c, k22t), k22c @ k22t / (k22t @ k22t), c(k4c, k4t),
          c(g4, c4t), g4 @ c4t / (g4 @ g4), sh[0], sh[1], sh[2], noise))
