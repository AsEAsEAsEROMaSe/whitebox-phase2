"""hyb101 — ГІБРИД: замикання до шару K, далі МК З ГАУСІАНА `N(m_K, C_K)`.

АРИФМЕТИКА, ЗАРАДИ ЯКОЇ ЦЕ РОБИТЬСЯ:
    повний форвард = 32 матвектори на семпл
    гібрид від K   = (32−K) матвекторів      ⟹ при K=24 семплів у **×4** більше
    шум МК ~ 1/√N  ⟹ V падає у **×4**;  а `V` = 91% балу і `MSE ≤ V` СТРУКТУРНО

ЦІНА ПОМИЛКИ: реальний `h_K` підмінюється ГАУСОВИМ із тими самими `(m,C)`.
Додається (1) похибка наших `(m_K,C_K)` і (2) негаусовість `h_K`.
`[орієнтир early100]` точний стан на шарі 24 вартий ×3.7, на шарі 1 — нічого
⟹ бити треба глибоко, і саме там гібрид найдешевший.

🚨 ЦЕ ВИБІРКОВА ІДЕЯ ⟹ конкурує з ПРЯМОЮ МК (0.1454% / V=1.9227e-06), а не із
   замиканням. Гейт відомий заздалегідь.

ЯК МІРЯЄМО ЧЕСНО:
  · пряма МК з `N` семплів (антитетика + вибілювання) — база
  · гібрид від шару K з `N·32/(32−K)` семплами — ТА САМА ціна у флопах
  · ціль — `final_means` (1e9 семплів) ⟹ незалежна, шум може лише ЗАНИЖУВАТИ
  · обидві гілки — на тій самій мережі, той самий сід ⟹ парне порівняння

🔒 ПЛОМБА K=0: гібрид від шару 0 = `N(0,I)` = РІВНО пряма МК ⟹ мусить дати те
   саме число в межах шуму. Якщо ні — зламана реалізація семплювання.
🔒 ПЛОМБА ОРАКУЛА: гібрид з ІСТИННИМИ `(m_K,C_K)` (з семплів) — верхня межа;
   різниця з нашим `(m_K,C_K)` показує, скільки коштує похибка замикання.
🔒 МАСШТАБУВАННЯ: два `N` ⟹ якщо перевага росте з `N`, це шум, а не метод.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒 K=0 відтворить пряму МК в межах 3%                            85%
 (б) гібрид K=24 БУДЕ ГІРШИМ за пряму МК (негаусовість h_24 з'їсть ×4) 65%
 (в) якщо і виграє, то лише на K=28…30                                40%
 (г) оракульний гібрид виграє на всіх K ≥ 8                           55%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG = 8
KAPPA = 0.9985
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BASE_N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096      # семплів для K=0
KS = [0, 8, 16, 24, 28, 30]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)
print(f"hyb101 · {NNET} мереж · база {BASE_N} семплів (K=0), "
      f"бюджет сталий у флопах", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure_to(Wg, K):
    """наша рекурсія до шару K; повертає (m, C) активацій ПІСЛЯ шару K−1"""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(K):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = Co + cp.diag(d - cp.diagonal(Co))
    return m, C


def truth_at(Wg, K, ns=131072):
    """істинні (m,C) активацій після шару K−1 — для 🔒 пломби оракула"""
    rng = cp.random.default_rng(31337)
    S1 = cp.zeros(W_, dtype=cp.float64)
    S2 = cp.zeros((W_, W_), dtype=cp.float64)
    n = 0
    while n < ns:
        nb = min(8192, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(K):
            a = cp.maximum(a @ Wg[i], 0.0)
        a64 = a.astype(cp.float64)
        S1 += a64.sum(0)
        S2 += a64.T @ a64
        n += nb
    N = 2 * ns
    m = S1 / N
    C = S2 / N - cp.outer(m, m)
    return m, 0.5 * (C + C.T)


def mc_from(Wg, K, m0, C0, nsamp, seed):
    """МК з N(m0,C0) через шари K..31; K=0 ⟹ (0,I) = пряма МК"""
    if K > 0:
        ev, U = cp.linalg.eigh(C0)
        L = U * cp.sqrt(cp.maximum(ev, 0.0))[None, :]
        L = L.astype(cp.float32)
        m0f = m0.astype(cp.float32)
    rng = cp.random.default_rng(seed)
    acc = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < nsamp:
        nb = min(8192, nsamp - n)
        g = rng.standard_normal((nb, W_), dtype=cp.float32)
        g = cp.concatenate([g, -g], axis=0)               # антитетика
        a = (m0f[None, :] + g @ L.T) if K > 0 else g
        for i in range(K, DEPTH):
            p = a @ Wg[i]
            a = cp.maximum(p, 0.0) if i < DEPTH - 1 else p
        acc += cp.maximum(a, 0.0).sum(0, dtype=cp.float64)
        n += nb
    return acc / (2 * nsamp)


res = {K: [0.0, 0.0] for K in KS}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    T = cp.asarray(FM[net].astype(np.float64))
    for K in KS:
        nsamp = BASE_N * DEPTH // max(DEPTH - K, 1)      # СТАЛИЙ бюджет флопів
        if K == 0:
            mo, Co = None, None
        else:
            mo, Co = closure_to(W64, K)
        v = mc_from(Wg, K, mo, Co, nsamp, 4242 + net)
        res[K][0] += float(cp.mean((v - T) ** 2))
        if K > 0:
            mt, Ct = truth_at(Wg, K)
            vt = mc_from(Wg, K, mt, Ct, nsamp, 4242 + net)
            res[K][1] += float(cp.mean((vt - T) ** 2))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'K':>4s}{'семплів':>10s}{'MSE наш (m,C)':>16s}{'×проти K=0':>12s}"
      f"{'🔒MSE оракул':>15s}{'×оракул':>10s}")
print("=" * 68)
b = res[0][0] / NNET
for K in KS:
    v = res[K][0] / NNET
    o = res[K][1] / NNET if K > 0 else v
    print(f"{K:>4d}{BASE_N*DEPTH//max(DEPTH-K,1):>10d}{v:>16.6e}{b/v:>12.4f}"
          f"{o:>15.6e}{b/o:>10.4f}", flush=True)
print(f"\n🔒 ПЛОМБА K=0 — це пряма МК. Записане V (грейдер, 50 мереж) = 1.9227e-06")
print("🚨 гейт: вибіркова ідея конкурує з ПРЯМОЮ МК, тобто зі стовпцем K=0")
