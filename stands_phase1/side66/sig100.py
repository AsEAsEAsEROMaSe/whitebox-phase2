"""sig100 — ЗАМИКАННЯ ЗАНИЖУЄ σ НА 10.6%. ЧИ КУПУЄ ЩОСЬ ПРОСТИЙ МНОЖНИК?

`[ЗМІРЯНО walk100, мережа 0, 32768 семплів]` медіана `s_h/s_t − 1` росте
монотонно: −0.08%(шар1) · −1.35%(8) · −4.24%(16) · −6.90%(24) · −10.56%(31).
Найкращий СПІЛЬНИЙ множник знімає −16.2%, лишаючи 9.58% форми.

ЧОМУ ЦЕ НЕ МЕРТВЕ ЗА ОДНОРІДНІСТЮ: масштабна однорідність замикання каже, що
`(m,C) → (c·m, c²·C)` виносить `c` на вихід ⟹ пошаровий скаляр НА ОБОХ = те,
що вже робить `_ALPHA`. Тут масштабується **лише C, при незмінному m** ⟹
змінюється `t = m/s`, і зміна проходить через `Φ,φ` НЕЛІНІЙНО. Інша ступінь
свободи. Ціна — нуль флопів.

ГЕЙТИ (усі до прогону):
  ціна        0 флопів                                   ✅
  узгодженість  множник чіпає діаг І позадіаг РАЗОМ      ✅
  збіжність     не ряд                                   ✅ (не застосовний)
  сито С99      двомоментний клас, стеля ×2.52 у балі    ⚠️ не претендуємо вище
  масштаб точки не розклад                               ✅

ЦІЛЬ — ІСТИНА БАНКУ `final_means` (N=1e9), тобто шуму МК немає взагалі.
БАЗА — те саме замикання при κ=1. Тест ПАРНИЙ (ті самі мережі, та сама істина).

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) оптимум κ буде > 1 (треба ПІДНЯТИ σ), бо замикання її занижує     70%
 (б) виграш після перепідбору α буде < ×1.15                          65%
 (в) оптимум на зрізі A втримається на зрізі B (закон «підбір/вирок»)  50%
 (г) залежний від глибини κ(L) поб'є константний щонайменше вдвічі     45%
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH, W_ = 32, 256
SEED, PILOT = 12345, 128
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def closure(Wc, kap):
    """замикання sub99 з множником kap(L) на C ПЕРЕД обчисленням s."""
    m = np.zeros(Wc[0].shape[0])
    C = np.eye(Wc[0].shape[0])
    for i in range(DEPTH):
        w = Wc[i]
        cp = w.T @ C @ w
        k = kap[i]
        cpk = cp * k                                # ← ЄДИНА ЗМІНА
        s = np.sqrt(np.maximum(np.diagonal(cpk), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        if i == DEPTH - 1:
            break
        C = cpk * np.outer(Ph, Ph) + (0.5 * (cpk * cpk)) * np.outer(ph / s,
                                                                    ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diagonal(C))
    return m


def prune(Wf):
    rng = np.random.default_rng(SEED)
    zp = rng.standard_normal((PILOT, W_)).astype(np.float32)
    xp = np.concatenate([zp, -zp], axis=0)
    alive, a = [], xp
    for i in range(DEPTH):
        h = a @ Wf[i].astype(np.float32)
        if i < DEPTH - 1:
            alive.append(np.max(h, axis=0) >= 0.0)
        a = np.maximum(h, 0.0)
    out = []
    for i in range(DEPTH):
        wi = Wf[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < DEPTH - 1:
            wi = wi[:, alive[i]]
        out.append(np.ascontiguousarray(wi, dtype=np.float64))
    return out


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
x = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(x.type):
    x = x.flatten()
Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, W_, W_)
y = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(y.type):
    y = y.flatten()
FM = np.asarray(y).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)

NETS = [prune([np.ascontiguousarray(Wall[i][L], dtype=np.float64)
               for L in range(DEPTH)]) for i in range(NNET)]
TRU = [FM[i].astype(np.float64) for i in range(NNET)]
print(f"sig100 · {NNET} мереж · ціль = final_means банку (N=1e9), ШУМУ НЕМА",
      flush=True)


def mse_of(kap, idx):
    """MSE з ПЕРЕПІДІБРАНОЮ α (спільною на зрізі — як робить сабміт)."""
    M = np.array([closure(NETS[i], kap) for i in idx])
    T = np.array([TRU[i] for i in idx])
    a = float((M * T).sum() / max((M * M).sum(), 1e-300))
    return float((((a * M) - T) ** 2).mean()), a


ALL = list(range(NNET))
base, a0 = mse_of(np.ones(DEPTH), ALL)
print(f"🔒 БАЗА (κ=1), α перепідібрана = {a0:.6f} : MSE = {base:.6e}")
print(f"   для орієнтиру: грейдер на цій же гілці дає 2.3385e-05\n")

print("СКАН КОНСТАНТНОГО κ (множник на C щошару)")
print(f"{'κ':>8s}{'MSE':>15s}{'×база':>9s}{'α':>10s}")
best, bk = base, 1.0
for k in (0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10, 1.20, 1.35, 1.50):
    v, a = mse_of(np.full(DEPTH, k), ALL)
    mark = "  ←" if v < best else ""
    if v < best:
        best, bk = v, k
    print(f"{k:>8.2f}{v:>15.6e}{base/v:>9.4f}{a:>10.6f}{mark}")

print(f"\nнайкраще константне κ = {bk} ⟹ ×{base/best:.4f}")

# ── κ, залежне від глибини: κ(L) = 1 + c·(L/31) ──
print("\nСКАН ЛІНІЙНОГО κ(L) = 1 + c·(L/31)")
print(f"{'c':>8s}{'MSE':>15s}{'×база':>9s}")
Ls = np.arange(DEPTH) / (DEPTH - 1.0)
bestc, bc = base, 0.0
for c in (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
    v, _ = mse_of(1.0 + c * Ls, ALL)
    mark = "  ←" if v < bestc else ""
    if v < bestc:
        bestc, bc = v, c
    print(f"{c:>8.2f}{v:>15.6e}{base/v:>9.4f}{mark}")
print(f"\nнайкраще c = {bc} ⟹ ×{base/bestc:.4f}")

# ── 🔒 ЗАКОН: ПІДБІР НА ЗРІЗІ A, ВИРОК НА ЗРІЗІ B ──
print("\n🔒 ПІДБІР НА A, ВИРОК НА B (закон, порушення якого коштувало 8 переворотів)")
A_, B_ = ALL[0::2], ALL[1::2]
for nm, grid, mk in (("κ конст.", (0.90, 0.95, 0.98, 1.0, 1.02, 1.05, 1.10,
                                   1.20, 1.35, 1.50),
                      lambda k: np.full(DEPTH, k)),
                     ("κ(L) лін.", (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05,
                                    0.10, 0.20, 0.30, 0.50),
                      lambda c: 1.0 + c * Ls)):
    bA, pA = None, None
    for p in grid:
        v, _ = mse_of(mk(p), A_)
        if bA is None or v < bA:
            bA, pA = v, p
    bB0, _ = mse_of(mk(grid[list(grid).index(1.0 if "конст" in nm else 0.0)]),
                    B_)
    bB, _ = mse_of(mk(pA), B_)
    print(f"  {nm:<11s} оптимум на A: {pA:>6.2f}  ⟹  на B ×{bB0/bB:.4f}"
          f"   (in-sample на A ×{mse_of(mk(1.0 if 'конст' in nm else 0.0), A_)[0]/bA:.4f})")
