"""rank102 — ЧИ ПЕРЕНОСИТЬСЯ ПОКРАЩЕННЯ КРОКУ В НАКОПИЧЕНУ ПОХИБКУ `m`?

ІДЕЯ ЮЗЕРА (С102): «для якоїсь ваги всі інші ваги слою є функцією. нам фактично
треба провести розклад на функції, знаючи точні ваги».
Формально:  p_a = h·w_a = mp_a + Σ_k β[k,a]·ξ_k + e_a
    ξ_k = r головних напрямків центрованого h — СПІЛЬНІ для всього шару («функції»)
    β   = √λ̂_k · (u_kᵀ W)  — коефіцієнти, які дає КОНКРЕТНА вага (точна!)
    e   = залишок, гаусів за побудовою (сума 256−r напрямків)

ПІДСТАВА: lowrank101 зміряв це на ОДНОМУ КРОЦІ при точних моментах —
    шар 31: r=0 0.1620% · r=4 0.1339% (×1.21) · r=16 0.1013% (×1.60)
    🔒 чужа r=4 = 0.1831% > 0.1620% ⟹ зламала як треба, сигнал реальний
    виграш РОСТЕ з глибиною (шар 1: ×1.002 при ранзі 165; шар 31: ранг 2.8)

ЧОГО НЕ ЗМІРЯНО І ЧОМУ ЦЕ ВИРІШУЄ ГІЛКУ: С101 показала, що 2/3 похибки `m`
робить ПЕРЕНОС, а не крок (ідеальна локальна m = ×1.111, істинна m = ×3.109).
⟹ покращення кроку може ПОВНІСТЮ загубитись у накопиченні. Якщо загубиться —
гілка мертва при будь-якій ціні, і жодну реалізацію писати не треба.

СХЕМА (форма ОРАКУЛЬНА, моменти НАШІ — це верхня межа):
    старт: істинні (m_1, C_1) з великої вибірки
    крок l=1..31: несемо (m̂,Ĉ); напрямки U_r і семпли ξ беремо з ІСТИННОЇ C_l,
                  але їхню ВАГУ λ̂_k = u_kᵀ Ĉ u_k — з НАШОЇ Ĉ
    ⟹ при r=0 формула ТОТОЖНО дорівнює гаусовому замиканню (вбудована пломба)
    C оновлюється Мелером K=2 з коефіцієнтами c1,c2, усередненими по ξ

🚨 ЗАКОН С101-в: поправку до `m` міряти на КОСИНУСНІЙ похибці, бо масштаб
   `_ALPHA` бере даром. Щілина сира/косинусна = ×1.9. Друкуємо ОБИДВІ.
🔒 СІДИ ІСТИНИ Й ЯДРА РОЗДІЛЕНІ (різні seed) — спільного шуму немає.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 r=0 відтворить гаусову рекурсію до 6 знаків (пломба)            95%
 (б) r=4 зніме >15% накопиченої СИРОЇ похибки m                        45%
 (в) на КОСИНУСНІЙ виграш буде ВДВІЧІ менший, ніж на сирій             60%
 (г) 🔒 чужа зламає (гірше за r=0)                                     80%
 (д) виграш буде МЕНШИЙ за крокові 17.4%, бо перенос домінує           65%

ВИРОК: якщо r=4 знімає <8% КОСИНУСНОЇ накопиченої похибки ⟹ гілка мертва
(за ситом С99 це <×1.02 у балі при ціні щонайменше 1% бюджету).
Якщо >20% на косинусній ⟹ живе, і тоді міряти ціну несення ядра.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")          # ПОСТІЙНИЙ шлях: /tmp чиститься

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
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 524288
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
RMAX = 16
RANKS = [0, 1, 2, 4, 8, 16]

if cp.cuda.runtime.getDeviceCount() != 1:
    print("ПІНУВАННЯ НЕ СПРАЦЮВАЛО — видно не одну карту", flush=True)
    sys.exit(2)

tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"rank102 · {NNET} мереж · істина {NTRUE} · ядро {NSUB} · ранги {RANKS}",
      flush=True)


def draw(seed, npair):
    """антитетичні пари + вибілювання — точні перші два моменти вибірки"""
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
    cov = (z.T @ z) / z.shape[0]
    wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
    return (z @ wh).astype(cp.float32)


def collect(Wg, seed0, ntot, want_states):
    """один прохід: суми h_l і h_lᵀh_l для l=1..32.
    want_states=True ⟹ ще й повертає самі стани (для проєкції на ядро)"""
    S = cp.zeros((DEPTH + 1, W_), dtype=cp.float64)
    Q = cp.zeros((DEPTH + 1, W_, W_), dtype=cp.float64)
    states = [] if want_states else None
    n = 0
    while n < ntot // 2:
        nb = min(4096, ntot // 2 - n)
        h = draw(seed0 + n, nb)
        cur = []
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
            h64 = h.astype(cp.float64)
            S[i + 1] += h64.sum(0)
            Q[i + 1] += h64.T @ h64
            if want_states:
                cur.append(h.copy())
        if want_states:
            states.append(cur)
        n += nb
    m = S / ntot
    C = Q / ntot - cp.einsum('li,lj->lij', m, m)
    return m, C, states


def step(m, C, W, Ur, xi, r):
    """один крок рекурсії; r=0 ⟹ ТОЧНО гаусове замикання"""
    mp = m @ W
    Cp = W.T @ C @ W
    s2 = cp.maximum(cp.diag(Cp), 1e-300)
    s = cp.sqrt(s2)
    if r == 0:
        t = mp / s
        Ph = ndtr(t)
        pf = cp.exp(-0.5 * t * t) / SQ2PI
        m_new = s * (pf + t * Ph)
        e2 = s2 * ((t * t + 1.0) * Ph + t * pf)
    else:
        U = Ur[:, :r]                                   # (256,r) з ІСТИННОЇ C_l
        lam = cp.maximum(((C @ U) * U).sum(0), 0.0)     # вага — з НАШОЇ Ĉ
        beta = cp.sqrt(lam)[:, None] * (U.T @ W)        # (r,256)
        v = (beta * beta).sum(0)
        se2 = cp.maximum(s2 - v, 0.01 * s2)
        se = cp.sqrt(se2)
        te = (mp[None, :] + xi[:, :r] @ beta) / se[None, :]
        Phi = ndtr(te)
        ph = cp.exp(-0.5 * te * te) / SQ2PI
        m_new = (se[None, :] * (ph + te * Phi)).mean(0)
        Ph = Phi.mean(0)
        pf = ph.mean(0) * (se / s)
        e2 = (se2[None, :] * ((te * te + 1.0) * Phi + te * ph)).mean(0)
    var = cp.maximum(e2 - m_new * m_new, 1e-300)
    Cn = Cp * cp.outer(Ph, Ph) + (Cp * Cp / (2.0 * cp.outer(s, s))) * cp.outer(pf, pf)
    idx = cp.arange(W_)
    Cn[idx, idx] = var
    return m_new, Cn


def errs(mh, mt):
    d = mh - mt
    raw = float(cp.sqrt((d * d).sum() / (mt * mt).sum()))
    c2 = float((mh * mt).sum() ** 2 / ((mh * mh).sum() * (mt * mt).sum()))
    return 100.0 * raw, 100.0 * math.sqrt(max(1.0 - c2, 0.0))


KER, TRUE = {}, {}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    mt, Ct, _ = collect(Wg, 810000 + net * 7919, NTRUE, False)
    # ядро — ОКРЕМИЙ сід (закон: сіди істини й оракула розділяти)
    _, _, st = collect(Wg, 930000 + net * 6113, NSUB, True)
    U = cp.zeros((DEPTH, W_, RMAX), dtype=cp.float64)
    XI = cp.zeros((DEPTH, NSUB, RMAX), dtype=cp.float32)
    for l in range(1, DEPTH):
        w_, v_ = cp.linalg.eigh(Ct[l])
        Ul = v_[:, ::-1][:, :RMAX]
        lam = cp.maximum(w_[::-1][:RMAX], 1e-300)
        U[l] = Ul
        hs = cp.concatenate([b[l - 1] for b in st], 0)[:NSUB].astype(cp.float64)
        x = (hs - mt[l][None, :]) @ Ul / cp.sqrt(lam)[None, :]
        x = (x - x.mean(0)) / cp.maximum(x.std(0), 1e-300)
        XI[l] = x.astype(cp.float32)
    KER[net] = (U, XI)
    TRUE[net] = (mt, Ct)
    del Wg, st
    cp.get_default_memory_pool().free_all_blocks()
    print(f"  мережа {net}: ядро готове", flush=True)

RES = np.zeros((NNET, len(RANKS) + 1, 2))
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    Wg64 = [w.astype(cp.float64) for w in Wg]
    mt, Ct = TRUE[net]
    for ri, r in enumerate(RANKS + ["CHU"]):
        rr = 4 if r == "CHU" else r
        src = KER[(net + 1) % NNET] if r == "CHU" else KER[net]
        U, XI = src
        m, C = mt[1].copy(), Ct[1].copy()
        for l in range(1, DEPTH):
            m, C = step(m, C, Wg64[l], U[l], XI[l].astype(cp.float64), rr)
        RES[net, ri] = errs(m, mt[DEPTH])
    del Wg, Wg64
    cp.get_default_memory_pool().free_all_blocks()
    print(f"  мережа {net}: рекурсія готова  сира r=0 {RES[net,0,0]:.4f}% "
          f"r=4 {RES[net,3,0]:.4f}%", flush=True)

M = RES.mean(0)
print("\n" + "=" * 78)
print("НАКОПИЧЕНА похибка m на шарі 32 (старт — істинні m_1,C_1)")
print(f"{'ранг':>6} {'СИРА':>10} {'×виграш':>9} {'КОСИНУСНА':>11} {'×виграш':>9}")
print("-" * 78)
for ri, r in enumerate(RANKS + ["CHU r=4"]):
    lab = str(r)
    print(f"{lab:>6} {M[ri,0]:>9.4f}% {M[0,0]/M[ri,0]:>9.4f} "
          f"{M[ri,1]:>10.4f}% {M[0,1]/M[ri,1]:>9.4f}")
print("-" * 78)
i4 = RANKS.index(4)
cut = 100.0 * (1.0 - M[i4, 1] / M[0, 1])
print(f"🔒 ПЛОМБА r=0 — це РІВНО гаусова рекурсія")
print(f"🔒 ЧУЖА мусить бути ГІРШОЮ за r=0: {M[-1,1]:.4f}% проти {M[0,1]:.4f}%")
print(f"ВИРОК: r=4 знімає {cut:.1f}% КОСИНУСНОЇ похибки "
      f"({'МЕРТВО' if cut < 8 else 'ЖИВЕ' if cut > 20 else 'СІРА ЗОНА'})")
