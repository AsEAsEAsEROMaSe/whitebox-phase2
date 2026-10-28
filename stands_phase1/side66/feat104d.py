"""feat104d — END-TO-END: вчити поправку ПРЯМО на фінальній косинусній похибці.

ЄДИНИЙ НЕПЕРЕВІРЕНИЙ ПУНКТ АУДИТУ (п.5). Досі модель училась на ПРОКСІ-цілі
`b_l` (похибка кроку), і розрив між R² і виграшем спостерігався ДВІЧІ:
    канал m: R²=0.353 ⟹ ×1.0112
    канал σ: R²=0.836 ⟹ ×1.0000   ← передбачувана майже ідеально, виграш НУЛЬ
⟹ проксі-ціль не є тим, що важить. Тут ціль — САМЕ фінальна косинусна похибка,
градієнт чисельний (центральні різниці) через усі 32 кроки рекурсії.

🔑 ЧОМУ ЦЕ МОЖЕ ДАТИ БІЛЬШЕ: система похибок ЗБАЛАНСОВАНА (сьомий доказ —
c_s*=0 при R²=0.84). Навчання на `b` тягне поправку до «правильної» похибки
кроку, яка системі НЕ потрібна. Навчання на фінальній метриці саме знайде ту
комбінацію, що виживає після взаємного гасіння — якщо вона існує.

МОДЕЛЬ: b̂ = (ознаки·w + b0), застосовується як `m ← крок(·) − c·b̂·s` при
i>KSIG. Параметри: w (14 або 28 з квадратами), b0, κ, gam — ВСІ разом,
тобто спільна калібровка вбудована в оптимізацію (закон С100).

🔒 ПЛОМБИ: (1) w=0, (κ,gam)=(0.9985,1.048) мусить дати рівно базу
           (2) вирок ЛИШЕ на TEST-мережах, оптимізація на TRAIN
           (3) 🔒 «ЧУЖА»: навчена модель із переставленими мережами
           (4) стеля гілки ×1.2436, попереднє найкраще ×1.024

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - end-to-end поб'є навчання на проксі (×1.024):        55%
   - дасть на TEST > ×1.10:                               20%
   - лінійна і квадратична зійдуться до того самого:      45%
   - 🔒 «чужа» зламається (×<1.0):                        85%
"""
import math
import os
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG = 8
NF = 14
GX = [0, 1, 4, 8, 16, 24, 31]
G3T = np.interp(np.arange(DEPTH), GX,
                [0.003, 0.211, 0.302, 0.331, 0.411, 0.412, 0.398])
G4T = np.interp(np.arange(DEPTH), GX,
                [0.005, 0.049, 0.133, 0.125, 0.231, 0.239, 0.220])
SRC = "/home/trim/wblogs/feat104c.npz"
if not os.path.exists(SRC):
    print("🚨 немає feat104c.npz — спершу треба зібрати дані")
    sys.exit(2)
if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту")
    sys.exit(2)

z = np.load(SRC)
XA, WA, TA = z["X"], z["W"], z["T"]
NN = XA.shape[0]
TR, TE = list(range(NN // 2)), list(range(NN // 2, NN))
mu = cp.asarray(XA.reshape(-1, NF).mean(0).astype(np.float64))
sd = cp.asarray(XA.reshape(-1, NF).std(0).astype(np.float64) + 1e-12)
del XA
WG = {k: cp.asarray(WA[k]) for k in range(NN)}
TG = {k: cp.asarray(TA[k]) for k in range(NN)}
del WA
print(f"feat104d · мереж {NN} · TRAIN {len(TR)} / TEST {len(TE)}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def loss(idx, th, quad, perm=None):
    """косинусна похибка батчу idx при параметрах th = [w..., b0, κ, gam]"""
    P = len(idx)
    nw = (NF * 2 if quad else NF)
    w = cp.asarray(th[:nw])[None, None, :]
    b0, kap, gam = float(th[nw]), float(th[nw + 1]), float(th[nw + 2])
    Wb = cp.stack([WG[k] for k in idx])
    m = cp.zeros((P, W_), dtype=cp.float64)
    C = cp.empty((P, W_, W_), dtype=cp.float32)
    C[:] = cp.eye(W_, dtype=cp.float32)
    dg = cp.arange(W_)
    for i in range(DEPTH):
        ww = Wb[:, i]
        cpm = cp.matmul(cp.matmul(ww.transpose(0, 2, 1), C), ww).astype(
            cp.float64)
        if i > KSIG:
            cpm = cpm * kap
        s2 = cp.maximum(cpm[:, dg, dg], 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, ww.astype(cp.float64))
        u = gam * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        if i > KSIG:
            tr = cpm[:, dg, dg].sum(1)
            er = tr * tr / cp.maximum((cpm * cpm).sum((1, 2)), 1e-30)
            rw = cp.sqrt((cpm * cpm).sum(2))
            one = cp.ones((P, W_), dtype=cp.float64)
            f = cp.stack([
                u, Ph, ph, u * u, s / cp.mean(s, 1, keepdims=True),
                one * (i / (DEPTH - 1.0)), one * cp.mean(u, 1, keepdims=True),
                one * cp.std(u, 1, keepdims=True),
                s2 / cp.mean(s2, 1, keepdims=True),
                rw / cp.maximum(cp.sqrt((cpm * cpm).sum((1, 2)))[:, None],
                                1e-30),
                (cpm * cpm).sum(2) / cp.maximum(s2 * tr[:, None], 1e-30),
                one * (er / 256.0)[:, None], one * 0.0,
                (-(G3T[i] / 6.0) * u * ph
                 + (G4T[i] / 24.0) * (u * u - 1.0) * ph)], axis=2)
            f = (f - mu) / sd
            if quad:
                f = cp.concatenate([f, f * f], axis=2)
            if perm is not None:                     # 🔒 КОНТРОЛЬ «ЧУЖА»
                f = f[perm]
            mn = mn - ((f * w).sum(2) + b0) * s
        if i == DEPTH - 1:
            break
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
    tot = tt = 0.0
    for a, k in enumerate(idx):
        t = TG[k]
        x = mn[a]
        tot += float(t @ t) - float(x @ t) ** 2 / max(float(x @ x), 1e-300)
        tt += float(t @ t)
    del Wb, C, cpm
    v = math.sqrt(max(tot, 0.0) / tt)
    return v if (v == v and v < 1e3) else 1e3   # 🚨 nan/вибух ⟹ великий штраф


for quad in (False, True):
    nw = NF * 2 if quad else NF
    th = np.zeros(nw + 3)
    th[nw + 1], th[nw + 2] = 0.9985, 1.048
    base_tr, base_te = loss(TR, th, quad), loss(TE, th, quad)
    if not quad:
        print(f"\n🔒 ПЛОМБА (1) w=0 ⟹ TRAIN {100*base_tr:.4f}% · "
              f"TEST {100*base_te:.4f}%")
    h = np.array([0.0004] * nw + [0.0004, 0.0004, 0.0015])
    M = np.zeros_like(th)
    V = np.zeros_like(th)
    t0 = time.time()
    for it in range(90):
        g = np.zeros_like(th)
        for q in range(len(th)):
            tp, tm = th.copy(), th.copy()
            tp[q] += h[q]
            tm[q] -= h[q]
            g[q] = (loss(TR, tp, quad) - loss(TR, tm, quad)) / (2 * h[q])
        M = 0.9 * M + 0.1 * g
        V = 0.999 * V + 0.001 * g * g
        st = 0.0002 * (M / (1 - 0.9 ** (it + 1))) / (
            np.sqrt(V / (1 - 0.999 ** (it + 1))) + 1e-9)
        st[nw + 1] *= 0.15
        st[nw + 2] *= 0.4
        th -= np.clip(st, -0.001, 0.001)
        nrm = np.linalg.norm(th[:nw])          # 🚨 кліп: без нього поправка
        if nrm > 0.03:                          #    вибухає і рекурсія дає nan
            th[:nw] *= 0.03 / nrm
        th[nw + 1] = min(max(th[nw + 1], 0.99), 1.01)
        th[nw + 2] = min(max(th[nw + 2], 1.00), 1.10)
        if it % 20 == 19:
            print(f"    it {it+1:>3}  TRAIN {100*loss(TR, th, quad):.4f}%  "
                  f"‖w‖={np.linalg.norm(th[:nw]):.4f}", flush=True)
    etr, ete = loss(TR, th, quad), loss(TE, th, quad)
    pm = cp.asarray(np.roll(np.arange(len(TE)), 1))
    ech = loss(TE, th, quad, pm)
    nm = "квадратична (28+3)" if quad else "лінійна (14+3)"
    print(f"\n{nm} · {time.time()-t0:.0f} с · κ*={th[nw+1]:.4f} "
          f"gam*={th[nw+2]:.4f} · ‖w‖={np.linalg.norm(th[:nw]):.4f}")
    print(f"  TRAIN {100*etr:.4f}%  ×{base_tr/etr:.4f}")
    print(f"  TEST  {100*ete:.4f}%  ×{base_te/ete:.4f}   ← ВИРОК")
    print(f"  🔒 «чужа» {100*ech:.4f}%  ×{base_te/ech:.4f}")

print("\n🔒 (4) стеля гілки ×1.2436 · навчання на проксі-цілі дало ×1.011…1.024")
