"""mix102 — ЧИ ВІДТВОРЮЄ ЗАМИКАННЯ *ФОРМУ* ЗМІШУВАННЯ, І ЧИ Є З НЕЇ РУЧКА

ІДЕЯ ЮЗЕРА (С102): «форма змішування каже нам те, як входи працюють разом.
і в нас є точні числа результату роботи» — тобто змішування є ДЕТЕРМІНОВАНОЮ
функцією ваг, а не властивістю вибірки. rank102b зміряв його на ІСТИНІ:
    eff_rank 165.8 → 2.24 (піврозпад 4.83 шару) · frozen 0.318 → 0.943
    cos(z,z′) 0.319 → 0.979
Але жодного разу не питали: а що дає НАША рекурсія на тих самих шарах?

ДВА ПИТАННЯ В ОДНОМУ ПРОГОНІ:

(1) ДІАГНОСТИКА — де саме ламається опис змішування:
    напрямки:  cos(v̂_k, v_k), k=1..4  +  підпростірний cos топ-4
    швидкість: eff_rank(Ĉ) проти eff_rank(C) по шарах
    спектр:    λ̂_1/λ_1  ·  tr Ĉ/tr C  ·  frozen наш проти істинного
    ⟹ якщо НАПРЯМКИ точні, а eff_rank ні — помилка суто у СПЕКТРІ,
      і це ручка з механізмом, а не ще одна калібрувальна константа.

(2) РУЧКА (тестується одразу, бо коштує один матмул):
    C ← C + c·C²/tr(C),  ПОТІМ слід відновлюється: C ← C·tr(C_old)/tr(C_new)
    власні вектори не змінюються, власні значення λ → λ + cλ²/trC
    c>0 ⟹ великі λ ростуть сильніше ⟹ eff_rank ПАДАЄ (сильніше змішано)
    🔑 відновлення сліду ОБОВ'ЯЗКОВЕ: інакше це просто масштаб, який `_ALPHA`
       бере даром (закон С101-в: поправку до m міряти на КОСИНУСНІЙ похибці)
    ✅ гейт узгодженості C пройдено за побудовою: C² чіпає діагональ і
       позадіагональ РАЗОМ, вони не роз'їжджаються

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 c=0 відтворить базу біт у біт (пломба)                         95%
 (б) cos(v̂_1,v_1) > 0.95 на всіх шарах — головний напрямок ми знаємо   70%
 (в) наше eff_rank БІЛЬШЕ за істинне (недозмішуємо) на глибині         60%
 (г) оптимум c ≠ 0 і має ОДИН знак на всіх мережах                     45%
 (д) виграш на косинусній похибці > ×1.02                              30%

ВИРОК: якщо (б) вірне, а (в) ні — форма змішування в нас правильна, і гілка
закрита як об'єкт. Якщо (в) вірне і (г) теж — маємо другу ручку типу κ.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
CS = [-0.30, -0.15, -0.05, 0.0, 0.05, 0.15, 0.30, 0.60]
LSIG = int(sys.argv[3]) if len(sys.argv) > 3 else 8
PROBE = [2, 4, 8, 12, 16, 20, 24, 28, 31]

if cp.cuda.runtime.getDeviceCount() != 1:
    print("ПІНУВАННЯ НЕ СПРАЦЮВАЛО", flush=True)
    sys.exit(2)

tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"mix102 · {NNET} мереж · істина {NTRUE} · ручка з шару {LSIG}", flush=True)


def draw(seed, npair):
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
    cov = (z.T @ z) / z.shape[0]
    wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
    return (z @ wh).astype(cp.float32)


def collect(Wg, seed0, ntot):
    S = cp.zeros((DEPTH + 1, W_), dtype=cp.float64)
    Q = cp.zeros((DEPTH + 1, W_, W_), dtype=cp.float64)
    n = 0
    while n < ntot // 2:
        nb = min(4096, ntot // 2 - n)
        h = draw(seed0 + n, nb)
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
            h64 = h.astype(cp.float64)
            S[i + 1] += h64.sum(0)
            Q[i + 1] += h64.T @ h64
        n += nb
    m = S / ntot
    return m, Q / ntot - cp.einsum('li,lj->lij', m, m)


def step(m, C, W):
    mp = m @ W
    Cp = W.T @ C @ W
    s2 = cp.maximum(cp.diag(Cp), 1e-300)
    s = cp.sqrt(s2)
    t = mp / s
    Ph = ndtr(t)
    pf = cp.exp(-0.5 * t * t) / SQ2PI
    m_new = s * (pf + t * Ph)
    e2 = s2 * ((t * t + 1.0) * Ph + t * pf)
    var = cp.maximum(e2 - m_new * m_new, 1e-300)
    Cn = Cp * cp.outer(Ph, Ph) + (Cp * Cp / (2.0 * cp.outer(s, s))) * cp.outer(pf, pf)
    idx = cp.arange(W_)
    Cn[idx, idx] = var
    return m_new, Cn


def knob(C, c):
    """C ← C + c·C²/tr(C) зі СТАЛИМ слідом ⟹ міняється лише форма спектра"""
    if c == 0.0:
        return C
    tr0 = cp.trace(C)
    Cn = C + (c / cp.maximum(tr0, 1e-300)) * (C @ C)
    return Cn * (tr0 / cp.maximum(cp.trace(Cn), 1e-300))


def effrank(C):
    tr = cp.trace(C)
    return float(tr * tr / cp.maximum((C * C).sum(), 1e-300))


def cerr(mh, mt):
    c2 = float((mh * mt).sum() ** 2 / ((mh * mh).sum() * (mt * mt).sum()))
    return 100.0 * math.sqrt(max(1.0 - c2, 0.0))


DIAG = np.zeros((NNET, len(PROBE), 8))
SCAN = np.zeros((NNET, len(CS)))
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    Wg64 = [w.astype(cp.float64) for w in Wg]
    mt, Ct = collect(Wg, 810000 + net * 7919, NTRUE)

    # --- (1) діагностика на чистій гаусовій рекурсії ---
    m, C = mt[1].copy(), Ct[1].copy()
    for l in range(1, DEPTH):
        m, C = step(m, C, Wg64[l])
        if (l + 1) in PROBE:
            pi = PROBE.index(l + 1)
            Cr = Ct[l + 1]
            wh, vh = cp.linalg.eigh(C)
            wr, vr = cp.linalg.eigh(Cr)
            vh, vr = vh[:, ::-1], vr[:, ::-1]
            wh, wr = wh[::-1], wr[::-1]
            cs = [abs(float((vh[:, k] * vr[:, k]).sum())) for k in range(4)]
            P = vh[:, :4].T @ vr[:, :4]
            sub = float((P * P).sum()) / 4.0
            trh, trr = float(cp.trace(C)), float(cp.trace(Cr))
            nm2h = float((m * m).sum())
            nm2r = float((mt[l + 1] * mt[l + 1]).sum())
            DIAG[net, pi] = [effrank(C), effrank(Cr), cs[0], cs[3], sub,
                             float(wh[0]) / max(float(wr[0]), 1e-300),
                             trh / max(trr, 1e-300),
                             nm2h / (nm2h + trh) - nm2r / (nm2r + trr)]

    # --- (2) скан ручки ---
    for ci, c in enumerate(CS):
        m, C = mt[1].copy(), Ct[1].copy()
        for l in range(1, DEPTH):
            m, C = step(m, C, Wg64[l])
            if l + 1 > LSIG:
                C = knob(C, c)
        SCAN[net, ci] = cerr(m, mt[DEPTH])
    b = CS[int(np.argmin(SCAN[net]))]
    print(f"  мережа {net}: база {SCAN[net, CS.index(0.0)]:.4f}% "
          f"найкраще c={b:+.2f} {SCAN[net].min():.4f}%", flush=True)
    del Wg, Wg64
    cp.get_default_memory_pool().free_all_blocks()

D = DIAG.mean(0)
print("\n" + "=" * 84)
print("(1) ЧИ ВІДТВОРЮЄ ЗАМИКАННЯ ФОРМУ ЗМІШУВАННЯ")
print(f"{'шар':>4} {'eff_наш':>9} {'eff_іст':>9} {'×':>7} {'cos v1':>8} "
      f"{'cos v4':>8} {'top4':>7} {'λ1̂/λ1':>8} {'trÎ/tr':>8} {'Δfrozen':>9}")
print("-" * 84)
for pi, l in enumerate(PROBE):
    print(f"{l:>4} {D[pi,0]:>9.2f} {D[pi,1]:>9.2f} {D[pi,0]/D[pi,1]:>7.3f} "
          f"{D[pi,2]:>8.4f} {D[pi,3]:>8.4f} {D[pi,4]:>7.4f} {D[pi,5]:>8.4f} "
          f"{D[pi,6]:>8.4f} {D[pi,7]:>+9.4f}")

S = SCAN.mean(0)
i0 = CS.index(0.0)
print("\n" + "=" * 84)
print(f"(2) РУЧКА C ← C + c·C²/tr(C) при шарі > {LSIG}, слід СТАЛИЙ")
print(f"{'c':>7} {'косинусна':>11} {'×виграш':>9} {'знаковий':>10}")
print("-" * 84)
for ci, c in enumerate(CS):
    win = int((SCAN[:, ci] < SCAN[:, i0]).sum())
    print(f"{c:>+7.2f} {S[ci]:>10.4f}% {S[i0]/S[ci]:>9.4f} {win:>7}/{NNET}")
print("-" * 84)
bi = int(np.argmin(S))
print(f"🔒 ПЛОМБА: c=0 має бути РІВНО базою · оптимум c={CS[bi]:+.2f} "
      f"⟹ ×{S[i0]/S[bi]:.4f}, знаковий {int((SCAN[:,bi]<SCAN[:,i0]).sum())}/{NNET}")
print(f"ОДНАКОВИЙ ЗНАК ОПТИМУМУ: "
      f"{sum(1 for n in range(NNET) if CS[int(np.argmin(SCAN[n]))] > 0)}/{NNET} додатних")
