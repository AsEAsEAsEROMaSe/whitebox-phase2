"""feat104 — ЧИ ПЕРЕДБАЧУВАНА ПОНЕЙРОННА ПОПРАВКА `b_l` З ЛОКАЛЬНИХ ВЕЛИЧИН.

ЗВІДКИ: ідея ЮЗЕРА про навчену модель, яка видає поправки. Форма «модель видає
КОНСТАНТИ з ознак МЕРЕЖІ» вже виміряна тричі і дала нуль (С102: 55 ознак,
1000 мереж, крос ×1.0019, 🔒перемішані ×0.9915; те саме для gam_i і W*_i).
Форма «модель видає поправку на НЕЙРОН» не міряна жодного разу, а арифметика
перефіту там інша: 16×32×256 = 131 072 приклади замість 50…1000.

ЦІЛЬ: `y = b_l[j]/s_l[j]` — безрозмірна похибка кроку на нейроні, де
`b_l = крок(m_l^іст, C_l^іст) − m_{l+1}^іст`. Саме вона дала ×1.2436 з
оракулом і пройшла контроль «чужа» [equil104b].

ОЗНАКИ (усі з локального стану, БЕЗ істини на виході):
   u, Φ(u), φ(u), u², s/⟨s⟩, глибина, ⟨u⟩, sd(u), C_jj/⟨C_jj⟩,
   ‖C_j:‖/‖C‖, participation_j, eff_rank/256, frozen, γ₄-табличний член
🔑 останній — саме той, що провалився в equil104b (cos≈0); лишений НАВМИСНО,
   щоб було видно, чи модель знаходить щось ПОНАД нього.

🔒 ПЛОМБИ І КОНТРОЛІ:
   (1) крос ПО МЕРЕЖАХ (навчання 0..7, вирок 8..15) — ніколи не по рядках
   (2) 🔒 ПЕРЕМІШАНІ ОЗНАКИ: R² мусить впасти до ~0, інакше прилад бреше
   (3) 🚨 R² НЕ Є ВИГРАШЕМ (закон корпусу) ⟹ головна метрика — реальний
       виграш у КОСИНУСНІЙ похибці після переносу b̂ і λ-скану на TRAIN
   (4) стеля відома: оракульне `b` дає ×1.2436 ⟹ модель не може дати більше

ПОРІГ, ОГОЛОШЕНИЙ ДО ПРОГОНУ: потрібен R²(test) > 0.6, бо виграш іде як
   ×1/√(1 − (0.762·√R²)²):   R²=0.3 → ×1.10 · R²=0.6 → ×1.24 · R²=0.9 → ×1.44
   ⟹ нижче 0.6 це чергове ×1.02 у балі.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - R²(test) лінійної моделі > 0.3:                     35%
   - R²(test) MLP > 0.6:                                 20%
   - 🔒 перемішані дадуть |R²| < 0.02:                    85%
   - реальний виграш на TEST > ×1.05 у косинусній:       20%
"""
import glob
import json
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NPAIRS = 131072
CHUNK = 4096
KSIG, KAP, GAM = 8, 0.9985, 1.048
GX = [0, 1, 4, 8, 16, 24, 31]
G3T = np.interp(np.arange(DEPTH), GX,
                [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398])
G4T = np.interp(np.arange(DEPTH), GX,
                [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220])
NF = 14
OUT = "/home/trim/wblogs/feat104.npz"

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

    X = cp.zeros((DEPTH, W_, NF), dtype=cp.float32)
    Y = cp.zeros((DEPTH, W_), dtype=cp.float32)
    S = cp.zeros((DEPTH, W_), dtype=cp.float64)
    D = cp.zeros((DEPTH, W_), dtype=cp.float64)
    for i in range(DEPTH):
        C, w = Ct[i], W64[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAP
        s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
        s = cp.sqrt(s2)
        mp = mt[i] @ w
        u = GAM * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        tgt = out_t if i == DEPTH - 1 else mt[i + 1]
        Y[i] = ((mn - tgt) / s).astype(cp.float32)
        S[i] = s
        D[i] = Ph + u * ph * (1.0 - GAM)
        rw = cp.sqrt((cpm * cpm).sum(1))
        fr = float((mt[i] @ mt[i]) / max(float((mt[i] @ mt[i])
                                               + cp.trace(C)), 1e-300))
        tr = float(cp.trace(cpm))
        er = tr * tr / max(float((cpm * cpm).sum()), 1e-300)
        f = [u, Ph, ph, u * u, s / cp.mean(s),
             cp.full(W_, i / (DEPTH - 1.0)), cp.full(W_, float(cp.mean(u))),
             cp.full(W_, float(cp.std(u))), s2 / cp.mean(s2),
             rw / max(float(cp.linalg.norm(cpm)), 1e-300),
             (cpm * cpm).sum(1) / cp.maximum(s2 * tr, 1e-300),
             cp.full(W_, er / 256.0), cp.full(W_, fr),
             (-(G3T[i] / 6.0) * u * ph + (G4T[i] / 24.0) * (u * u - 1.0) * ph)]
        X[i] = cp.stack(f, axis=1).astype(cp.float32)
    Wst = cp.stack(W64).astype(cp.float32)
    r = (cp.asnumpy(X), cp.asnumpy(Y), cp.asnumpy(S), cp.asnumpy(D),
         cp.asnumpy(Wst), cp.asnumpy(out_t))
    del Wg, W64, Ct, mt, X, Y, S, D, Wst
    cp.get_default_memory_pool().free_all_blocks()
    return r


if not os.path.exists(OUT):
    print(f"feat104 · збір {NNET} мереж × {2*NPAIRS} семплів", flush=True)
    t0 = time.time()
    AX, AY, AS, AD, AW, AT = [], [], [], [], [], []
    for k, (wn, tn) in enumerate(stream(NNET)):
        x, y, s, d, ws, ot = one_net(k, wn)
        AX.append(x)
        AY.append(y)
        AS.append(s)
        AD.append(d)
        AW.append(ws)
        AT.append(np.stack([tn, ot]))
        print(f"  ✅ мережа {k} ({time.time()-t0:.1f} с)", flush=True)
    np.savez_compressed(OUT + ".tmp.npz", X=np.stack(AX), Y=np.stack(AY),
                        S=np.stack(AS), D=np.stack(AD), W=np.stack(AW),
                        T=np.stack(AT))
    os.replace(OUT + ".tmp.npz", OUT)
    del AX, AY, AS, AD, AW, AT

z = np.load(OUT)
X, Y, S, D, WW, TT = z["X"], z["Y"], z["S"], z["D"], z["W"], z["T"]
NN = X.shape[0]
TR, TE = list(range(NN // 2)), list(range(NN // 2, NN))
print(f"\nданих: {NN}×{DEPTH}×{W_} = {NN*DEPTH*W_} рядків · {NF} ознак")


def flat(idx, shuf=False, seed=0):
    x = X[idx].reshape(-1, NF).astype(np.float64)
    y = Y[idx].reshape(-1).astype(np.float64)
    if shuf:                       # 🔒 КОНТРОЛЬ: рядки ознак перемішано
        x = x[np.random.default_rng(seed).permutation(len(x))]
    return x, y


def stdz(a, mu, sd):
    return (a - mu) / sd


xtr, ytr = flat(TR)
mu, sd = xtr.mean(0), xtr.std(0) + 1e-12
ym, ys = ytr.mean(), ytr.std() + 1e-12


def ridge(xt, yt, lam=1e-3):
    A = np.hstack([stdz(xt, mu, sd), np.ones((len(xt), 1))])
    return np.linalg.solve(A.T @ A + lam * len(xt) * np.eye(A.shape[1]),
                           A.T @ ((yt - ym) / ys))


def r2(w, xt, yt, quad=False):
    a = stdz(xt, mu, sd)
    if quad:
        a = np.hstack([a, a * a, a[:, :6] * a[:, 1:7]])
    A = np.hstack([a, np.ones((len(a), 1))])
    p = A @ w * ys + ym
    return 1.0 - ((yt - p) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()


def ridge_q(xt, yt, lam=1e-3):
    a = stdz(xt, mu, sd)
    a = np.hstack([a, a * a, a[:, :6] * a[:, 1:7]])
    A = np.hstack([a, np.ones((len(a), 1))])
    return np.linalg.solve(A.T @ A + lam * len(xt) * np.eye(A.shape[1]),
                           A.T @ ((yt - ym) / ys))


xte, yte = flat(TE)
wl = ridge(xtr, ytr)
wq = ridge_q(xtr, ytr)
xs, yss = flat(TR, True, 7)
wsx = ridge_q(xs, yss)
print(f"\n{'модель':<28}{'R² train':>10}{'R² TEST':>10}")
print(f"{'лінійна (14 ознак)':<28}{r2(wl, xtr, ytr):>10.4f}"
      f"{r2(wl, xte, yte):>10.4f}")
print(f"{'квадратична (+34 члени)':<28}{r2(wq, xtr, ytr, True):>10.4f}"
      f"{r2(wq, xte, yte, True):>10.4f}")
print(f"{'🔒 ПЕРЕМІШАНІ ОЗНАКИ':<28}{r2(wsx, xs, yss, True):>10.4f}"
      f"{r2(wsx, xte, yte, True):>10.4f}")

# ---- MLP 14→32→32→1 на GPU, Adam
g = cp.random.default_rng(5)
xg = cp.asarray(stdz(xtr, mu, sd))
yg = cp.asarray((ytr - ym) / ys)
xv = cp.asarray(stdz(xte, mu, sd))
P = [g.standard_normal((NF, 32)) * 0.3, cp.zeros(32),
     g.standard_normal((32, 32)) * 0.2, cp.zeros(32),
     g.standard_normal((32, 1)) * 0.2, cp.zeros(1)]
M = [cp.zeros_like(p) for p in P]
V = [cp.zeros_like(p) for p in P]


def fwd(x, P):
    h1 = cp.tanh(x @ P[0] + P[1])
    h2 = cp.tanh(h1 @ P[2] + P[3])
    return (h2 @ P[4] + P[5])[:, 0], h1, h2


for ep in range(600):
    p, h1, h2 = fwd(xg, P)
    e = p - yg
    n = len(yg)
    g5 = (h2.T @ e[:, None]) / n
    g6 = e.mean(keepdims=True)
    d2 = (e[:, None] @ P[4].T) * (1 - h2 * h2)
    g3 = (h1.T @ d2) / n
    g4 = d2.mean(0)
    d1 = (d2 @ P[2].T) * (1 - h1 * h1)
    g1 = (xg.T @ d1) / n
    g2 = d1.mean(0)
    for k, gr in enumerate([g1, g2, g3, g4, g5, g6]):
        M[k] = 0.9 * M[k] + 0.1 * gr
        V[k] = 0.999 * V[k] + 0.001 * gr * gr
        P[k] -= 0.02 * (M[k] / (1 - 0.9 ** (ep + 1))) / (
            cp.sqrt(V[k] / (1 - 0.999 ** (ep + 1))) + 1e-8)
ptr = cp.asnumpy(fwd(xg, P)[0]) * ys + ym
pte = cp.asnumpy(fwd(xv, P)[0]) * ys + ym
rt = 1 - ((ytr - ptr) ** 2).sum() / ((ytr - ytr.mean()) ** 2).sum()
rv = 1 - ((yte - pte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
print(f"{'MLP 14→32→32→1 (Adam 600)':<28}{rt:>10.4f}{rv:>10.4f}")

# ---- 🚨 R² НЕ Є ВИГРАШЕМ: женемо b̂ через перенос і міряємо косинусну
BH = pte.reshape(len(TE), DEPTH, W_) * S[TE]
BQ = (np.hstack([stdz(xte, mu, sd), stdz(xte, mu, sd) ** 2,
                 stdz(xte, mu, sd)[:, :6] * stdz(xte, mu, sd)[:, 1:7],
                 np.ones((len(xte), 1))]) @ wq * ys + ym
      ).reshape(len(TE), DEPTH, W_) * S[TE]


def transport(bb, n):
    d = np.zeros(W_)
    for i in range(DEPTH):
        if i > 0:
            d = D[n, i] * (d @ WW[n, i])
        d = d + bb[i]
    return d


def cerr(lam, src):
    tot = tt = 0.0
    for a, n in enumerate(TE):
        m = TT[n, 1] * 0
        mm = TT[n, 1]
        t = TT[n, 0]
        x = mm - lam * src[a]
        tot += (t @ t) - (x @ t) ** 2 / max(x @ x, 1e-300)
        tt += t @ t
        del m
    return math.sqrt(max(tot, 0.0) / tt)


# ⚠️ TT[n,1] — це ІСТИННИЙ вихід МК, а не наше передбачення: тут ми міряємо
#    лише ЧИ ЗБІГАЄТЬСЯ напрямок b̂ з оракульним; повний ланцюг — у equil104b
dm = [transport(BH[a], n) for a, n in enumerate(TE)]
dq = [transport(BQ[a], n) for a, n in enumerate(TE)]
do = [transport(Y[n] * S[n], n) for n in TE]
cm = np.mean([float(dm[a] @ do[a] / (np.linalg.norm(dm[a])
                                     * np.linalg.norm(do[a]) + 1e-300))
              for a in range(len(TE))])
cq = np.mean([float(dq[a] @ do[a] / (np.linalg.norm(dq[a])
                                     * np.linalg.norm(do[a]) + 1e-300))
              for a in range(len(TE))])
print(f"\n🔑 cos(перенесене b̂, перенесене b_оракул) на TEST:")
print(f"   квадратична {cq:+.4f}   ·   MLP {cm:+.4f}")
best = max(abs(cq), abs(cm))
eff = 0.762 * best
print(f"\nефективний cos = 0.762 × {best:.4f} = {eff:.4f}")
print(f"⟹ виграш у точності ×{1/math.sqrt(max(1-eff*eff,1e-9)):.4f}"
      f"  ⟹ у балі ≈×{1+(1/math.sqrt(max(1-eff*eff,1e-9))-1)*0.22:.4f}")
print(f"\nПОРІГ ДО ПРОГОНУ: R²(test) > 0.6 · стеля гілки ×1.2436")
