"""rank102b — ДЕ САМЕ ЛАМАЄТЬСЯ, І ЯК ШВИДКО МЕРЕЖА «ЗМІШУЄ ФАРБУ»

ЩО ЗМУСИЛО (rank102): оракульне ранг-r ядро в рекурсії ПСУЄ накопичену `m`
монотонно з рангом — косинусна 0.4064%(r=0) → 0.4502%(r=4) → 0.6627%(r=16),
🔒 пломба масштабування NSUB×4 залишила числа на місці ⟹ це НЕ шум.
Протилежно до lowrank101, де те саме ядро покращувало ОДИН КРОК у ×1.21.

ПИТАННЯ 1 — ДЕ ПЕРЕХІД: вмикаємо ядро лише з шару LFROM і далі.
   LFROM=32 ⟹ чиста гаусова рекурсія (пломба, має дати r=0)
   LFROM=31 ⟹ ЛИШЕ останній крок — тут крокове покращення ще не встигло
              ні накопичитись, ні розузгодитись
   якщо LFROM=31 ПОКРАЩУЄ, а LFROM=1 псує ⟹ винне НАКОПИЧЕННЯ (розузгодження
   каналів), а не сама формула кроку. Це різні вироки і різні наслідки.

ПИТАННЯ 2 — ІДЕЯ ЮЗЕРА «ЦЕ ЯК КОЛОТИТИ ФАРБУ»: чи є метрика змішування і
як вона йде по шарах. Міряємо на ІСТИННИХ (m_l, C_l) — ціна нуль, бо вони
вже пораховані:
   eff_rank = (tr C)²/‖C‖_F²        скільки незалежних «кольорів» лишилось
   top1                             частка найбільшого власного значення
   frozen = ‖m‖²/(‖m‖²+tr C)        частка стану, що ВЖЕ НЕ ЗАЛЕЖИТЬ від входу
   cosxy                            сер. косинус між h(z) і h(z′) для НЕЗАЛЕЖНИХ z
   halflife                         за скільки шарів eff_rank падає вдвічі

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 LFROM=32 відтворить r=0 біт у біт (пломба)                     95%
 (б) LFROM=31 (один крок) ПОКРАЩИТЬ косинусну                          55%
 (в) існує LFROM*, де крива перевертається, і він у діапазоні 24…30    50%
 (г) eff_rank падає ГЕОМЕТРИЧНО (сталий множник на шар ±20%)           70%
 (д) frozen монотонно росте і на шарі 31 перевищить 0.90               65%
 (е) cosxy на шарі 31 буде ≈0.974 (відтворить записане число)          80%

ВИРОК: якщо жодне LFROM не дає виграшу ⟹ ранг-ядро мертве не через ціну
і не через накопичення, а тому що покращення кроку НЕСУМІСНЕ з нашими (m,C).
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

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
RMAX = 16
LFROMS = [1, 8, 16, 24, 28, 30, 31, 32]
RS = [4, 16]

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
print(f"rank102b · {NNET} мереж · істина {NTRUE} · ядро {NSUB}", flush=True)


def draw(seed, npair):
    rng = cp.random.default_rng(seed)
    z = rng.standard_normal((npair, W_), dtype=cp.float32)
    z = cp.concatenate([z, -z], axis=0).astype(cp.float64)
    cov = (z.T @ z) / z.shape[0]
    wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
    return (z @ wh).astype(cp.float32)


def collect(Wg, seed0, ntot, want_states):
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
        U = Ur[:, :r]
        lam = cp.maximum(((C @ U) * U).sum(0), 0.0)
        beta = cp.sqrt(lam)[:, None] * (U.T @ W)
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


RES = np.zeros((NNET, len(RS), len(LFROMS), 2))
MIX = np.zeros((NNET, DEPTH + 1, 4))
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    Wg64 = [w.astype(cp.float64) for w in Wg]
    mt, Ct, _ = collect(Wg, 810000 + net * 7919, NTRUE, False)
    _, _, st = collect(Wg, 930000 + net * 6113, NSUB, True)

    # --- метрика змішування («як колотять фарбу») ---
    za = draw(550000 + net, 2048)
    zb = draw(660000 + net, 2048)
    ha, hb = za, zb
    for l in range(1, DEPTH + 1):
        C = Ct[l]
        tr = float(cp.trace(C))
        fro = float((C * C).sum())
        ev = cp.linalg.eigvalsh(C)
        nm2 = float((mt[l] * mt[l]).sum())
        ha = cp.maximum(ha @ Wg[l - 1], 0.0)
        hb = cp.maximum(hb @ Wg[l - 1], 0.0)
        na = ha / cp.maximum(cp.linalg.norm(ha, axis=1, keepdims=True), 1e-30)
        nb = hb / cp.maximum(cp.linalg.norm(hb, axis=1, keepdims=True), 1e-30)
        MIX[net, l] = [tr * tr / max(fro, 1e-300),
                       float(ev[-1]) / max(tr, 1e-300),
                       nm2 / max(nm2 + tr, 1e-300),
                       float((na * nb).sum(1).mean())]

    U = cp.zeros((DEPTH, W_, RMAX), dtype=cp.float64)
    XI = cp.zeros((DEPTH, NSUB, RMAX), dtype=cp.float64)
    for l in range(1, DEPTH):
        w_, v_ = cp.linalg.eigh(Ct[l])
        Ul = v_[:, ::-1][:, :RMAX]
        lam = cp.maximum(w_[::-1][:RMAX], 1e-300)
        U[l] = Ul
        hs = cp.concatenate([b[l - 1] for b in st], 0)[:NSUB].astype(cp.float64)
        x = (hs - mt[l][None, :]) @ Ul / cp.sqrt(lam)[None, :]
        XI[l] = (x - x.mean(0)) / cp.maximum(x.std(0), 1e-300)
    del st
    cp.get_default_memory_pool().free_all_blocks()

    for ri, r in enumerate(RS):
        for li, lf in enumerate(LFROMS):
            m, C = mt[1].copy(), Ct[1].copy()
            for l in range(1, DEPTH):
                m, C = step(m, C, Wg64[l], U[l], XI[l], r if l >= lf else 0)
            RES[net, ri, li] = errs(m, mt[DEPTH])
    print(f"  мережа {net}: r=4 LFROM1 {RES[net,0,0,1]:.4f}% "
          f"LFROM31 {RES[net,0,6,1]:.4f}% LFROM32 {RES[net,0,7,1]:.4f}%",
          flush=True)
    del Wg, Wg64, U, XI
    cp.get_default_memory_pool().free_all_blocks()

M = RES.mean(0)
X = MIX.mean(0)
print("\n" + "=" * 80)
print("ПИТАННЯ 1 — З ЯКОГО ШАРУ ВМИКАТИ ЯДРО (косинусна похибка m на шарі 32)")
print(f"{'LFROM':>7} {'r=4':>10} {'×':>8} {'r=16':>10} {'×':>8}   (LFROM=32 — чиста база)")
print("-" * 80)
for li, lf in enumerate(LFROMS):
    print(f"{lf:>7} {M[0,li,1]:>9.4f}% {M[0,7,1]/M[0,li,1]:>8.4f} "
          f"{M[1,li,1]:>9.4f}% {M[1,7,1]/M[1,li,1]:>8.4f}")
print("-" * 80)
best = max(range(len(LFROMS)), key=lambda i: M[0, 7, 1] / M[0, i, 1])
print(f"НАЙКРАЩЕ r=4: LFROM={LFROMS[best]} ⟹ ×{M[0,7,1]/M[0,best,1]:.4f}")

print("\n" + "=" * 80)
print("ПИТАННЯ 2 — ЯК МЕРЕЖА «КОЛОТИТЬ ФАРБУ» (істинні моменти, ціна нуль)")
print(f"{'шар':>5} {'eff_rank':>10} {'×/шар':>8} {'top1':>8} {'frozen':>8} {'cos(z,z′)':>10}")
print("-" * 80)
prev = None
for l in [1, 2, 4, 8, 12, 16, 20, 24, 28, 31, 32]:
    rr = X[l, 0]
    rat = f"{rr/prev:>8.3f}" if prev else f"{'—':>8}"
    prev = rr
    print(f"{l:>5} {rr:>10.2f} {rat} {X[l,1]:>8.4f} {X[l,2]:>8.4f} {X[l,3]:>10.4f}")
print("-" * 80)
r1, r31 = X[1, 0], X[31, 0]
per = (r31 / r1) ** (1.0 / 30.0)
print(f"eff_rank {r1:.1f} → {r31:.2f} за 30 шарів ⟹ ×{per:.4f} на шар, "
      f"період піврозпаду {math.log(0.5)/math.log(per):.2f} шару")
print(f"🔒 передбачення (е): cos на шарі 31 ≈ 0.974 · зміряно {X[31,3]:.4f}")
