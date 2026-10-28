"""feat104c — АУДИТ ВЛАСНОЇ РОБОТИ: п'ять пропусків feat104b за один прогін.

feat104b дав ×1.0311 (крос, 🔒чужа λ*→0), але я перевірив лише НАЙВУЖЧУ форму
ідеї. Пропуски, знайдені на вимогу ЮЗЕРА «може ти не поширив на інше»:

 1. 🚨 ЩОКРОКОВЕ застосування замість фінального віднімання. Було:
    `m_31 ← m_31 − λ·Σ B_l b̂_l` (лінеаризація + невиправлені ознаки).
    Стало: `m ← крок(·) − c_m·b̂_l` НА КОЖНОМУ КРОЦІ ⟹ (а) лінеаризації немає,
    (б) наступні кроки бачать виправлений стан, (в) ознаки САМОУЗГОДЖЕНІ.
 2. 🚨 КАНАЛ `C`. |ΔC|/C = 43.6% проти |Δm|/m = 0.668% [mix103], а orac101
    дає для нього до ×1.149. Друга модель вчить похибку diag(C) — та сама
    понейронна структура, інша ціль.
 3. 🚨 κ і gam ПЕРЕПІДБИРАЮТЬСЯ РАЗОМ з (c_m, c_s) — закон С100: калібрувальні
    константи не переносяться між конфігураціями. Без цього виграш міг бути
    просто компенсацією неоптимальних констант.
 4. ознаки з ВИПРАВЛЕНОГО стану (наслідок п.1).
 6. `i > KSIG` проти «всі шари»: у сабміті поправка на мілких шарах зсуває
    опори λ-блоку і коштувала б ×0.57 [пастка С100].

🔒 ПЛОМБИ: (1) c_m=c_s=0 мусить відтворити базу при (κ,gam) = (0.9985,1.048)
           (2) 🔒 «ЧУЖА»: модель, застосована з переставленими мережами
           (3) вирок ЛИШЕ на TEST-мережах, (c_m,c_s,κ,gam) з TRAIN
           (4) стеля відома: ×1.2436 [equil104b]

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - щокрокове застосування поб'є фінальне (×1.0311):            60%
   - канал C додасть понад ×1.02 ПОНАД канал m:                  35%
   - спільний перепідбір κ,gam з'їсть половину виграшу m:        40%
   - разом на TEST вийде >×1.10:                                 25%
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
KSIG = 8
GX = [0, 1, 4, 8, 16, 24, 31]
G3T = np.interp(np.arange(DEPTH), GX,
                [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398])
G4T = np.interp(np.arange(DEPTH), GX,
                [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220])
NF = 14
CM = [0.0, 0.25, 0.5, 0.75, 1.0]
CS = [0.0, 0.25, 0.5, 0.75, 1.0]
KG = [(0.9970, 1.040), (0.9985, 1.040), (0.9985, 1.048), (1.0000, 1.048)]
OUT = "/home/trim/wblogs/feat104c.npz"

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


def feat_b(mp, s, s2, u, Ph, ph, cpm, dg, i, P):
    """ознаки для БАТЧУ мереж: (P,256,NF). Усе з локального стану."""
    tr = cpm[:, dg, dg].sum(1)
    er = tr * tr / cp.maximum((cpm * cpm).sum((1, 2)), 1e-30)
    rw = cp.sqrt((cpm * cpm).sum(2))
    one = cp.ones((P, W_), dtype=cp.float64)
    return cp.stack([
        u, Ph, ph, u * u, s / cp.mean(s, 1, keepdims=True),
        one * (i / (DEPTH - 1.0)), one * cp.mean(u, 1, keepdims=True),
        one * cp.std(u, 1, keepdims=True), s2 / cp.mean(s2, 1, keepdims=True),
        rw / cp.maximum(cp.sqrt((cpm * cpm).sum((1, 2)))[:, None], 1e-30),
        (cpm * cpm).sum(2) / cp.maximum(s2 * tr[:, None], 1e-30),
        one * (er / 256.0)[:, None],
        one * 0.0, (-(G3T[i] / 6.0) * u * ph
                    + (G4T[i] / 24.0) * (u * u - 1.0) * ph)], axis=2)


def recur(Wb, kap, gam, mdl, cm, cs, onlydeep=True, perm=None):
    """Батчева рекурсія з ЩОКРОКОВОЮ поправкою. mdl=None ⟹ чиста база."""
    P = Wb.shape[0]
    m = cp.zeros((P, W_), dtype=cp.float64)
    C = cp.empty((P, W_, W_), dtype=cp.float32)
    C[:] = cp.eye(W_, dtype=cp.float32)
    dg = cp.arange(W_)
    for i in range(DEPTH):
        w = Wb[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w).astype(
            cp.float64)
        if i > KSIG:
            cpm = cpm * kap
        s2 = cp.maximum(cpm[:, dg, dg], 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w.astype(cp.float64))
        u = gam * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        on = (i > KSIG) if onlydeep else True
        if mdl is not None and on and (cm != 0.0 or cs != 0.0):
            f = feat_b(mp, s, s2, u, Ph, ph, cpm, dg, i, P)
            if perm is not None:            # 🔒 КОНТРОЛЬ «ЧУЖА»
                f = f[perm]
            bm, bs = mdl(f)
            if cs != 0.0:                   # поправка до σ ⟹ ПЕРЕРАХУНОК кроку
                s = cp.maximum(s * (1.0 - cs * bs), 1e-30)
                s2 = s * s
                u = gam * (mp / s)
                Ph, ph = ndtr(u), phi(u)
            mn = mp * Ph + s * ph - cm * bm * s
        else:
            mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn
        pv = (ph / s).astype(cp.float32)
        P32 = Ph.astype(cp.float32)
        c32 = cpm.astype(cp.float32)
        C2 = c32 * c32
        C2 *= 0.5
        C2 *= pv[:, :, None]
        C2 *= pv[:, None, :]
        c32 *= P32[:, :, None]
        c32 *= P32[:, None, :]
        c32 += C2
        del C2
        c32[:, dg, dg] = ((mp * mp + s2) * Ph + (mp * s) * ph
                          - mn * mn).astype(cp.float32)
        C, m = c32, mn
    return m


def one_net(net, Wn):
    """МК-істина + ознаки та ДВІ цілі (m і diag C) з ІСТИННИХ входів."""
    Wg = [cp.asarray(Wn[L]) for L in range(DEPTH)]
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
    X = cp.zeros((DEPTH, W_, NF), dtype=cp.float32)
    YM = cp.zeros((DEPTH, W_), dtype=cp.float32)
    YS = cp.zeros((DEPTH, W_), dtype=cp.float32)
    dg = cp.arange(W_)
    for i in range(DEPTH):
        w = Wg[i].astype(cp.float64)
        cpm = w.T @ Ct[i] @ w
        if i > KSIG:
            cpm = cpm * 0.9985
        s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
        s = cp.sqrt(s2)
        mp = mt[i] @ w
        u = 1.048 * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        tgt = out_t if i == DEPTH - 1 else mt[i + 1]
        YM[i] = ((mn - tgt) / s).astype(cp.float32)
        if i < DEPTH - 1:                    # ціль каналу C: похибка σ кроку
            st = cp.sqrt(cp.maximum(cp.diagonal(Ct[i + 1]), 1e-30))
            sn = cp.sqrt(cp.maximum((mp * mp + s2) * Ph + (mp * s) * ph
                                    - mn * mn, 1e-30))
            YS[i] = ((sn - st) / st).astype(cp.float32)
        X[i] = feat_b(mp[None], s[None], s2[None], u[None], Ph[None],
                      ph[None], cpm[None], dg, i, 1)[0].astype(cp.float32)
    r = (cp.asnumpy(X), cp.asnumpy(YM), cp.asnumpy(YS), cp.asnumpy(out_t))
    del Wg, Ct, mt, X, YM, YS
    cp.get_default_memory_pool().free_all_blocks()
    return r


if not os.path.exists(OUT):
    print(f"feat104c · збір {NNET} мереж × {2*NPAIRS} семплів", flush=True)
    t0 = time.time()
    A = [[] for _ in range(6)]
    for k, (wn, tn) in enumerate(stream(NNET)):
        x, ym, ys, ot = one_net(k, wn)
        for q, v in enumerate((x, ym, ys, ot, wn, tn)):
            A[q].append(v)
        print(f"  ✅ мережа {k} ({time.time()-t0:.1f} с)", flush=True)
    np.savez(OUT + ".tmp.npz", X=np.stack(A[0]), YM=np.stack(A[1]),
             YS=np.stack(A[2]), OT=np.stack(A[3]), W=np.stack(A[4]),
             T=np.stack(A[5]))
    os.replace(OUT + ".tmp.npz", OUT)
    del A

z = np.load(OUT)
X, YM, YS, WA, T = z["X"], z["YM"], z["YS"], z["W"], z["T"]
NN = X.shape[0]
TR, TE = list(range(NN // 2)), list(range(NN // 2, NN))
print(f"\nмереж {NN} · TRAIN {len(TR)} / TEST {len(TE)} · "
      f"рядків {NN*DEPTH*W_}")

xtr = X[TR].reshape(-1, NF).astype(np.float64)
mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
YT = np.stack([YM[TR].reshape(-1), YS[TR].reshape(-1)], 1).astype(np.float64)
ym, ys_ = YT.mean(0), YT.std(0) + 1e-12
g = cp.random.default_rng(5)
xg = cp.asarray((xtr - mu) / sd)
yg = cp.asarray((YT - ym) / ys_)
Pp = [g.standard_normal((NF, 32)) * 0.3, cp.zeros(32),
      g.standard_normal((32, 32)) * 0.2, cp.zeros(32),
      g.standard_normal((32, 2)) * 0.2, cp.zeros(2)]
Mm = [cp.zeros_like(p) for p in Pp]
Vv = [cp.zeros_like(p) for p in Pp]
n = len(yg)
for e in range(800):
    h1 = cp.tanh(xg @ Pp[0] + Pp[1])
    h2 = cp.tanh(h1 @ Pp[2] + Pp[3])
    er = (h2 @ Pp[4] + Pp[5]) - yg
    gr = [None] * 6
    gr[4] = (h2.T @ er) / n
    gr[5] = er.mean(0)
    d2 = (er @ Pp[4].T) * (1 - h2 * h2)
    gr[2], gr[3] = (h1.T @ d2) / n, d2.mean(0)
    d1 = (d2 @ Pp[2].T) * (1 - h1 * h1)
    gr[0], gr[1] = (xg.T @ d1) / n, d1.mean(0)
    for k in range(6):
        Mm[k] = 0.9 * Mm[k] + 0.1 * gr[k]
        Vv[k] = 0.999 * Vv[k] + 0.001 * gr[k] * gr[k]
        Pp[k] -= 0.02 * (Mm[k] / (1 - 0.9 ** (e + 1))) / (
            cp.sqrt(Vv[k] / (1 - 0.999 ** (e + 1))) + 1e-8)
mug, sdg = cp.asarray(mu), cp.asarray(sd)
ymg, ysg = cp.asarray(ym), cp.asarray(ys_)


def mdl(f):
    x = ((f - mug) / sdg).reshape(-1, NF)
    h1 = cp.tanh(x @ Pp[0] + Pp[1])
    h2 = cp.tanh(h1 @ Pp[2] + Pp[3])
    o = (h2 @ Pp[4] + Pp[5]) * ysg + ymg
    return o[:, 0].reshape(f.shape[0], W_), o[:, 1].reshape(f.shape[0], W_)


pr = cp.asnumpy(cp.stack(mdl(cp.asarray(X[TE].reshape(-1, W_, NF)
                                        .astype(np.float64)))))
for q, nm in ((0, "m"), (1, "σ")):
    yv = (YM if q == 0 else YS)[TE].reshape(-1)
    p = pr[q].reshape(-1)
    print(f"  R²(TEST) канал {nm}: "
          f"{1 - ((yv-p)**2).sum()/((yv-yv.mean())**2).sum():.4f}")


def cerr(idx, mm):
    tot = tt = 0.0
    for a, nn_ in enumerate(idx):
        t = cp.asarray(T[nn_])
        x = mm[a]
        tot += float(t @ t) - float(x @ t) ** 2 / max(float(x @ x), 1e-300)
        tt += float(t @ t)
    return math.sqrt(max(tot, 0.0) / tt)


def run(idx, kap, gam, cm, cs, od=True, perm=None):
    return cerr(idx, recur(cp.asarray(WA[idx]), kap, gam, mdl, cm, cs, od,
                           perm))


print(f"\n🔒 ПЛОМБА (1) база (c=0, κ=0.9985, gam=1.048): TRAIN "
      f"{100*run(TR,0.9985,1.048,0,0):.4f}% · TEST "
      f"{100*run(TE,0.9985,1.048,0,0):.4f}%")
bTE = min(run(TE, k, g_, 0, 0) for k, g_ in KG)
bTR = min(run(TR, k, g_, 0, 0) for k, g_ in KG)
print(f"   найкраща база по (κ,gam): TRAIN {100*bTR:.4f}% · "
      f"TEST {100*bTE:.4f}%")

print(f"\n{'варіант':<34}{'c_m':>5}{'c_s':>5}{'κ':>8}{'gam':>6}"
      f"{'TRAIN':>9}{'TEST':>9}{'×':>8}")
for od, nm in ((True, "лише i>KSIG (безпечно)"),
               (False, "ВСІ шари (ламає λ-блок)")):
    best = None
    for cm in CM:
        for cs in CS:
            for k, g_ in KG:
                e = run(TR, k, g_, cm, cs, od)
                if best is None or e < best[0]:
                    best = (e, cm, cs, k, g_)
    e, cm, cs, k, g_ = best
    te = run(TE, k, g_, cm, cs, od)
    print(f"{nm:<34}{cm:>5.2f}{cs:>5.2f}{k:>8.4f}{g_:>6.3f}"
          f"{100*e:>8.4f}%{100*te:>8.4f}%{bTE/te:>8.4f}")
    if od:
        pm = cp.asarray(np.roll(np.arange(len(TE)), 1))
        ce = run(TE, k, g_, cm, cs, od, pm)
        print(f"{'  🔒 контроль «чужа»':<34}{'':>5}{'':>5}{'':>8}{'':>6}"
              f"{'':>9}{100*ce:>8.4f}%{bTE/ce:>8.4f}")
        for a, b_, t_ in ((cm, 0.0, "лише канал m"), (0.0, cs, "лише канал σ")):
            t2 = run(TE, k, g_, a, b_, od)
            print(f"{'  '+t_:<34}{a:>5.2f}{b_:>5.2f}{'':>8}{'':>6}{'':>9}"
                  f"{100*t2:>8.4f}%{bTE/t2:>8.4f}")
print(f"\n🔒 (4) стеля гілки [equil104b]: ×1.2436 · попереднє (фінальне "
      f"віднімання, feat104b): ×1.0311")
