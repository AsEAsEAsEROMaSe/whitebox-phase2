"""feat104b — ПОВНИЙ ЛАНЦЮГ: чи дає навчена поправка РЕАЛЬНИЙ виграш.

feat104 дав: R²(нейрон) = 0.346 (🔒перемішані 0.003), але після ПЕРЕНОСУ
cos(b̂, b_оракул) = +0.9309. Перенос усереднює ⟹ некорельований шум гасне.
🚨 МІЙ ПОРІГ `R²>0.6` БУВ НА НЕ ТУ ВЕЛИЧИНУ — поставлений із припущення
cos=√R², яке для перенесеної суми хибне.

ЩО ЛИШИЛОСЬ ПЕРЕВІРИТИ — І ЦЕ ВИРІШУЄ ГІЛКУ:
 (1) ознаки в feat104 бралися з ІСТИННИХ (m,C) ⟹ метод не існує.
     Тут — ДВА набори: `or` (істинні, верхня межа) і `ou` (НАШІ, реалізовний).
 (2) реальний виграш у КОСИНУСНІЙ похибці: m_наше − λ·d̂ проти істини,
     λ підбирається на TRAIN-мережах, вирок на TEST-мережах.
 🔒 (3) КОНТРОЛЬ «ЧУЖА»: d̂ з іншої мережі мусить дати λ*→0.
 🔒 (4) стеля відома: оракульне `b` дало ×1.2436 [equil104b] ⟹ більше не буде.
 🔒 (5) λ=0 мусить відтворити базову косинусну похибку 0.4796%.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - на ІСТИННИХ ознаках виграш на TEST > ×1.15:              55%
   - на НАШИХ ознаках виграш на TEST > ×1.10:                 30%
   - 🔒 «чужа» дасть λ*→0:                                     80%
   - гілка виявиться реалізовною (>×1.05 на наших, крос):     35%
"""
import glob
import math
import os
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NPAIRS = 131072
CHUNK = 4096
KSIG, KAP, GAM = 8, 0.9985, 1.048
GX = [0, 1, 4, 8, 16, 24, 31]
G3T = np.interp(np.arange(DEPTH), GX,
                [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398])
G4T = np.interp(np.arange(DEPTH), GX,
                [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220])
NF = 14
LAM = [round(-1.5 + 0.05 * k, 3) for k in range(61)]
OUT = "/home/trim/wblogs/feat104b.npz"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту — пінування зламане")
    sys.exit(2)


def stream(nmax):
    got = 0
    for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= nmax:
            break
        for b in pa.ipc.open_stream(pa.memory_map(fp, "rb")):
            if got >= nmax:
                break
            xw = b.column("weights")
            while not pa.types.is_primitive(xw.type):
                xw = xw.flatten()
            xt = b.column("final_means")
            while not pa.types.is_primitive(xt.type):
                xt = xt.flatten()
            nb = b.num_rows
            wv = np.asarray(xw).reshape(nb, DEPTH, W_, W_)
            tv = np.asarray(xt).reshape(nb, W_)
            for j in range(min(nb, nmax - got)):
                yield np.array(wv[j], dtype=np.float32), \
                    np.array(tv[j], dtype=np.float64)
            got += min(nb, nmax - got)
            del xw, xt, wv, tv


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def feats(m, C, w, i):
    """ознаки + крок. Працює і на істинних (m,C), і на наших — саме в цьому
    вся суть перевірки (1)."""
    cpm = w.T @ C @ w
    if i > KSIG:
        cpm = cpm * KAP
    s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
    s = cp.sqrt(s2)
    mp = m @ w
    u = GAM * (mp / s)
    Ph, ph = ndtr(u), phi(u)
    mn = mp * Ph + s * ph
    Dj = Ph + u * ph * (1.0 - GAM)
    rw = cp.sqrt((cpm * cpm).sum(1))
    tr = float(cp.trace(cpm))
    er = tr * tr / max(float((cpm * cpm).sum()), 1e-300)
    fr = float((m @ m) / max(float((m @ m) + cp.trace(C)), 1e-300))
    f = cp.stack([u, Ph, ph, u * u, s / cp.mean(s),
                  cp.full(W_, i / (DEPTH - 1.0)),
                  cp.full(W_, float(cp.mean(u))),
                  cp.full(W_, float(cp.std(u))), s2 / cp.mean(s2),
                  rw / max(float(cp.linalg.norm(cpm)), 1e-300),
                  (cpm * cpm).sum(1) / cp.maximum(s2 * tr, 1e-300),
                  cp.full(W_, er / 256.0), cp.full(W_, fr),
                  (-(G3T[i] / 6.0) * u * ph
                   + (G4T[i] / 24.0) * (u * u - 1.0) * ph)], axis=1)
    if i == DEPTH - 1:
        return mn, cpm, s, Dj, f.astype(cp.float32)
    Co = (cpm * cp.outer(Ph, Ph)
          + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
    d = (mp * mp + s2) * Ph + (mp * s) * ph - mn * mn
    Co = Co + cp.diag(d - cp.diagonal(Co))
    return mn, Co, s, Dj, f.astype(cp.float32)


def one_net(net, Wn):
    Wg = [cp.asarray(Wn[L]) for L in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    N = 2 * NPAIRS
    rng = cp.random.default_rng(1104 + net)
    S1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    S2 = cp.zeros((DEPTH, W_, W_), dtype=cp.float64)
    O1 = cp.zeros(W_, dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            a64 = a.astype(cp.float64)
            S1[i] += a64.sum(0)
            S2[i] += a64.T @ a64
            p = a @ Wg[i]
            a = cp.maximum(p, 0.0)
            if i == DEPTH - 1:
                O1 += a.astype(cp.float64).sum(0)
        done += nb
    del z, a, a64, p
    cp.get_default_memory_pool().free_all_blocks()
    mt, out_t = S1 / N, O1 / N
    Ct = S2 / N - mt[:, :, None] * mt[:, None, :]
    Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
    del S1, S2
    cp.get_default_memory_pool().free_all_blocks()

    XO = cp.zeros((DEPTH, W_, NF), dtype=cp.float32)   # ознаки з ІСТИННИХ
    XU = cp.zeros((DEPTH, W_, NF), dtype=cp.float32)   # ознаки з НАШИХ
    Y = cp.zeros((DEPTH, W_), dtype=cp.float32)
    SO = cp.zeros((DEPTH, W_), dtype=cp.float64)
    SU = cp.zeros((DEPTH, W_), dtype=cp.float64)
    DU = cp.zeros((DEPTH, W_), dtype=cp.float64)
    for i in range(DEPTH):
        mn, _, s, _, f = feats(mt[i], Ct[i], W64[i], i)
        tgt = out_t if i == DEPTH - 1 else mt[i + 1]
        Y[i] = ((mn - tgt) / s).astype(cp.float32)
        SO[i], XO[i] = s, f
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        m, C, s, Dj, f = feats(m, C, W64[i], i)
        SU[i], DU[i], XU[i] = s, Dj, f
    r = (cp.asnumpy(XO), cp.asnumpy(XU), cp.asnumpy(Y), cp.asnumpy(SO),
         cp.asnumpy(SU), cp.asnumpy(DU), cp.asnumpy(cp.stack(W64)
                                                    ).astype(np.float32),
         cp.asnumpy(m), cp.asnumpy(out_t))
    del Wg, W64, Ct, mt, XO, XU, Y, SO, SU, DU
    cp.get_default_memory_pool().free_all_blocks()
    return r


if not os.path.exists(OUT):
    print(f"feat104b · збір {NNET} мереж × {2*NPAIRS} семплів", flush=True)
    t0 = time.time()
    A = [[] for _ in range(10)]
    for k, (wn, tn) in enumerate(stream(NNET)):
        for q, v in enumerate(one_net(k, wn)):
            A[q].append(v)
        A[9].append(tn)
        print(f"  ✅ мережа {k} ({time.time()-t0:.1f} с)", flush=True)
    np.savez_compressed(OUT + ".tmp.npz", XO=np.stack(A[0]),
                        XU=np.stack(A[1]), Y=np.stack(A[2]), SO=np.stack(A[3]),
                        SU=np.stack(A[4]), DU=np.stack(A[5]), W=np.stack(A[6]),
                        MU=np.stack(A[7]), OT=np.stack(A[8]), T=np.stack(A[9]))
    os.replace(OUT + ".tmp.npz", OUT)
    del A

z = np.load(OUT)
XO, XU, Y = z["XO"], z["XU"], z["Y"]
SU, DU, WW, MU, T = z["SU"], z["DU"], z["W"], z["MU"], z["T"]
NN = XO.shape[0]
TR, TE = list(range(NN // 2)), list(range(NN // 2, NN))
print(f"\nмереж {NN} · TRAIN {len(TR)} · TEST {len(TE)} · "
      f"рядків {NN*DEPTH*W_}")


def train_mlp(xtr, ytr, seed=5, ep=600):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
    ym, ys = ytr.mean(), ytr.std() + 1e-12
    g = cp.random.default_rng(seed)
    xg = cp.asarray((xtr - mu) / sd)
    yg = cp.asarray((ytr - ym) / ys)
    P = [g.standard_normal((NF, 32)) * 0.3, cp.zeros(32),
         g.standard_normal((32, 32)) * 0.2, cp.zeros(32),
         g.standard_normal((32, 1)) * 0.2, cp.zeros(1)]
    M = [cp.zeros_like(p) for p in P]
    V = [cp.zeros_like(p) for p in P]
    n = len(yg)
    for e in range(ep):
        h1 = cp.tanh(xg @ P[0] + P[1])
        h2 = cp.tanh(h1 @ P[2] + P[3])
        er = (h2 @ P[4] + P[5])[:, 0] - yg
        g5 = (h2.T @ er[:, None]) / n
        g6 = er.mean(keepdims=True)
        d2 = (er[:, None] @ P[4].T) * (1 - h2 * h2)
        g3, g4 = (h1.T @ d2) / n, d2.mean(0)
        d1 = (d2 @ P[2].T) * (1 - h1 * h1)
        g1, g2 = (xg.T @ d1) / n, d1.mean(0)
        for k, gr in enumerate([g1, g2, g3, g4, g5, g6]):
            M[k] = 0.9 * M[k] + 0.1 * gr
            V[k] = 0.999 * V[k] + 0.001 * gr * gr
            P[k] -= 0.02 * (M[k] / (1 - 0.9 ** (e + 1))) / (
                cp.sqrt(V[k] / (1 - 0.999 ** (e + 1))) + 1e-8)

    def pred(x):
        xx = cp.asarray((x - mu) / sd)
        h1 = cp.tanh(xx @ P[0] + P[1])
        h2 = cp.tanh(h1 @ P[2] + P[3])
        return cp.asnumpy((h2 @ P[4] + P[5])[:, 0]) * ys + ym
    return pred


def transport(bb, n):
    d = np.zeros(W_)
    for i in range(DEPTH):
        if i > 0:
            d = DU[n, i] * (d @ WW[n, i])
        d = d + bb[i]
    return d


def cerr(sub, dd, lam, shift=0):
    tot = tt = 0.0
    for a, n in enumerate(sub):
        d = dd[(a + shift) % len(sub)]
        x = MU[n] - lam * d
        tot += float(T[n] @ T[n]) - float(x @ T[n]) ** 2 / max(float(x @ x),
                                                               1e-300)
        tt += float(T[n] @ T[n])
    return math.sqrt(max(tot, 0.0) / tt)


ytr = Y[TR].reshape(-1).astype(np.float64)
res = {}
for tag, XX, name in (("or", XO, "ознаки з ІСТИННИХ (m,C) — верхня межа"),
                      ("ou", XU, "ознаки з НАШИХ (m,C) — РЕАЛІЗОВНИЙ")):
    p = train_mlp(XX[TR].reshape(-1, NF).astype(np.float64), ytr)
    BH = {}
    for n in range(NN):
        BH[n] = p(XX[n].reshape(-1, NF).astype(np.float64)
                  ).reshape(DEPTH, W_) * SU[n]
    res[tag] = ([transport(BH[n], n) for n in TR],
                [transport(BH[n], n) for n in TE], name)

zer = [np.zeros(W_) for _ in TE]
b0 = cerr(TE, zer, 0.0)
print(f"\n🔒 ПЛОМБА (5): λ=0 на TEST ⟹ {100*b0:.4f}% (база, косинусна)")
print(f"\n{'варіант':<40}{'λ*(TRAIN)':>10}{'TEST':>10}{'×':>9}")
for tag in ("or", "ou"):
    dtr, dte, name = res[tag]
    bt = cerr(TR, [np.zeros(W_) for _ in TR], 0.0)
    lam = min((cerr(TR, dtr, l), l) for l in LAM)[1]
    e = cerr(TE, dte, lam)
    print(f"{name:<40}{lam:>10.2f}{100*e:>9.4f}%{b0/e:>9.4f}")
    sh = min((cerr(TE, dte, l, 1), l) for l in LAM)
    print(f"{'  🔒 контроль «чужа» (λ вільний на TEST)':<40}"
          f"{sh[1]:>10.2f}{100*sh[0]:>9.4f}%{b0/sh[0]:>9.4f}")
print(f"\n🔒 (4) стеля гілки (оракульне b, equil104b): ×1.2436")
