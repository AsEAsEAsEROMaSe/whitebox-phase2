"""equil104b — ТРИ ДОМІРИ, ЯКІ ВИРІШУЮТЬ ГІЛКУ ПЕРЕНОСУ ОСТАТОЧНО.

equil104 дав: лінійний перенос крокових похибок відтворює накопичену з
cos=0.762 (випадковий дав би 0.06), завищуючи норму на 30%; розклад по шарах
РІВНОМІРНИЙ; потенціал віднімання ×1.543 У СИРІЙ похибці.

ТРИ БЛОКЕРИ, ЯКІ ТРЕБА ЗНЯТИ АБО ВИЗНАТИ:
 (а) 🚨 СИРА ≠ КОСИНУСНА. Закон С101-в: щілина ×1.9, і в ній сидить рівно те,
     що `_ALPHA` бере даром. Міряємо ПРЯМО: min_λ √(1−cos²(m−λd, T)).
 (б) `b_l` рахувався з ІСТИННИХ (m,C) ⟹ метод не існує. Реалізовна версія:
     `b_l` з НАШИХ моментів через Еджворта.
 (в) чи є `b_l` узагалі еджвортівським членом — cos(b_реальне, b_Еджворт).

ЕДЖВОРТ БЕЗ ІСТИНИ: 1-D розклад `E[ReLU]/σ = ψ(t) − (γ₃/6)tφ + (γ₄/24)(t²−1)φ`
з УНІВЕРСАЛЬНИМИ табличними γ по глибині (записані в корпусі, підтверджені
двічі — orac101e і neff101, збіг 1–7%):
   γ₃  шари 0,1,4,8,16,24,31: 0.003 0.211 0.302 0.331 0.411 0.412 0.398
   γ₄                         0.005 0.049 0.133 0.125 0.231 0.239 0.220
⟹ `b_edg = s·[−(γ₃/6)·u·φ + (γ₄/24)·(u²−1)·φ]` рахується З НАШИХ `(m,C)`,
   без жодного семпла. Якщо він тягне ту саму інформацію — метод реалізовний.

🔒 ПЛОМБИ: (1) λ=0 мусить відтворити базову косинусну похибку
           (2) 🔒 КОНТРОЛЬ «ЧУЖА»: d_pred іншої мережі мусить дати λ*→0
           (3) вирок ЛИШЕ по кросу: λ* підбирається на A, міряється на B

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - оракульний d_pred у КОСИНУСНІЙ метриці дасть >×1.15:        45%
   - cos(b_реальне, b_Еджворт) > 0.5 хоч на глибоких шарах:      40%
   - табличний d_edg дасть крос >×1.05:                          20%
   - усе разом ⟹ реалізовний ×1.2:                               15%
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
G3 = [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398]
G4 = [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220]
G3T = np.interp(np.arange(DEPTH), GX, G3)
G4T = np.interp(np.arange(DEPTH), GX, G4)
LAM = [round(-2.0 + 0.05 * k, 3) for k in range(81)]
OUT = "/home/trim/wblogs/equil104b.json"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту — пінування зламане")
    sys.exit(2)
state = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    with open(OUT + ".tmp", "w") as fh:
        json.dump(state, fh)
    os.replace(OUT + ".tmp", OUT)


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


def step(m, C, w, i, want_edg=False):
    cpm = w.T @ C @ w
    if i > KSIG:
        cpm = cpm * KAP
    s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
    s = cp.sqrt(s2)
    mp = m @ w
    u = GAM * (mp / s)
    Ph, ph = ndtr(u), phi(u)
    mn = mp * Ph + s * ph
    D = Ph + u * ph * (1.0 - GAM)
    edg = None
    if want_edg:                       # 🔑 рахується З НАШИХ (m,C), без істини
        edg = s * (-(G3T[i] / 6.0) * u * ph
                   + (G4T[i] / 24.0) * (u * u - 1.0) * ph)
    if i == DEPTH - 1:
        return mn, cpm, D, edg
    Co = (cpm * cp.outer(Ph, Ph)
          + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
    d = (mp * mp + s2) * Ph + (mp * s) * ph - mn * mn
    Co = Co + cp.diag(d - cp.diagonal(Co))
    return mn, Co, D, edg


def one_net(net, Wn, T):
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
    mt = S1 / N
    Ct = S2 / N - mt[:, :, None] * mt[:, None, :]
    Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
    out_t = O1 / N
    del S1, S2
    cp.get_default_memory_pool().free_all_blocks()

    # b_l ОРАКУЛЬНЕ (з істинних входів) + b_l ЕДЖВОРТ на тих самих входах
    B, Dl, E = [], [], []
    for i in range(DEPTH):
        mn, _, D, eg = step(mt[i], Ct[i], W64[i], i, True)
        B.append(mn - (out_t if i == DEPTH - 1 else mt[i + 1]))
        Dl.append(D)
        E.append(eg)
    cb = [float((B[i] @ E[i]) / cp.maximum(cp.linalg.norm(B[i])
                                           * cp.linalg.norm(E[i]), 1e-300))
          for i in range(DEPTH)]

    # наша рекурсія + ЕДЖВОРТ на НАШИХ моментах (реалізовна версія)
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    Eo, Do = [], []
    for i in range(DEPTH):
        m, C, D, eg = step(m, C, W64[i], i, True)
        Eo.append(eg)
        Do.append(D)

    def transport(src, jac):
        d = cp.zeros(W_, dtype=cp.float64)
        for i in range(DEPTH):
            if i > 0:
                d = jac[i] * (d @ W64[i])
            d = d + src[i]
        return d

    d_or = transport(B, Dl)          # оракульний (потребує істини)
    d_eg = transport(Eo, Do)         # РЕАЛІЗОВНИЙ: усе з наших моментів
    TT = float(T @ T)

    # 🚨 зберігаємо ВЕКТОРИ: контроль «чужа» через підміну СКАЛЯРНИХ добутків
    #    ламає нерівність Коші-Шварца (mm,mT свої, md,dd,dT чужі) ⟹ фальшиві нулі
    r = {"m": [float(x) for x in m], "T": [float(x) for x in T],
         "or": [float(x) for x in d_or], "eg": [float(x) for x in d_eg],
         "cb": cb,
         "sd": float(cp.linalg.norm(out_t - T) / cp.linalg.norm(T))}
    del Wg, W64, B, Dl, E, Eo, Do, Ct, mt
    cp.get_default_memory_pool().free_all_blocks()
    return r


print(f"equil104b · {NNET} мереж × {2*NPAIRS} семплів", flush=True)
t0 = time.time()
for k, (wn, tn) in enumerate(stream(NNET)):
    if str(k) in state:
        continue
    state[str(k)] = one_net(k, wn, cp.asarray(tn))
    save()
    print(f"  ✅ мережа {k} ({time.time()-t0:.1f} с)", flush=True)

# ------------------------------------------------------------------ аналіз
ns = sorted(state, key=int)
NN = len(ns)
HA, HB = ns[:NN // 2], ns[NN // 2:]


V = {k: {f: np.array(state[k][f]) for f in ("m", "T", "or", "eg")}
     for k in state}


def cerr(sub, mode, lam, shift=0):
    """косинусна похибка для m − λ·d; shift>0 ⟹ 🔒 КОНТРОЛЬ «ЧУЖА»
    (вектор d береться з ІНШОЇ мережі — саме вектор, не скалярні добутки)"""
    tot = tt = 0.0
    for a, k in enumerate(sub):
        m, T = V[k]["m"], V[k]["T"]
        d = V[sub[(a + shift) % len(sub)] if shift else k][mode]
        x = m - lam * d
        xx = float(x @ x)
        tot += float(T @ T) - (float(x @ T) ** 2) / max(xx, 1e-300)
        tt += float(T @ T)
    return math.sqrt(max(tot, 0.0) / tt)


print(f"\n🔒 ПЛОМБА ПРИЛАДУ: ‖МК − final_means‖/‖T‖ = "
      f"{100*sum(state[k]['sd'] for k in ns)/NN:.4f}%")
b0 = cerr(ns, "or", 0.0)
print(f"🔒 ПЛОМБА (1): λ=0 ⟹ {100*b0:.4f}% (база, косинусна)")

print(f"\n{'варіант':<26}{'λ*(A)':>7}{'in-sample':>11}{'×':>8}"
      f"{'КРОС A→B':>10}{'×':>8}")
for mode, name in (("or", "ОРАКУЛЬНИЙ b (з істини)"),
                   ("eg", "ЕДЖВОРТ b (реалізовний)")):
    bA = cerr(HA, mode, 0.0)
    bB = cerr(HB, mode, 0.0)
    best = min((cerr(HA, mode, l), l) for l in LAM)
    xb = cerr(HB, mode, best[1])
    print(f"{name:<26}{best[1]:>7.2f}{100*best[0]:>10.4f}%"
          f"{bA/best[0]:>8.4f}{100*xb:>9.4f}%{bB/xb:>8.4f}")
    sh = min((cerr(HA, mode, l, 1), l) for l in LAM)
    print(f"{'  🔒 контроль «чужа»':<26}{sh[1]:>7.2f}{100*sh[0]:>10.4f}%"
          f"{bA/sh[0]:>8.4f}")

print(f"\n{'шар':>4} {'cos(b_реальне, b_Еджворт)':>26}")
for i in (1, 2, 4, 8, 12, 16, 20, 24, 28, 31):
    print(f"{i:>4} {sum(state[k]['cb'][i] for k in ns)/NN:>25.4f}")
print("\n🔒 (в) якщо cos малий — `b` НЕ є еджвортівським членом ⟹ реалізовної")
print("      оцінки `b` не існує, і гілка мертва незалежно від (а)")
