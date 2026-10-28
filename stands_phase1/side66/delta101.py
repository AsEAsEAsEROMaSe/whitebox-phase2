"""delta101 — ЧИ Є ОДНОШАГОВА ПОХИБКА `E[ReLU]` УНІВЕРСАЛЬНОЮ ФУНКЦІЄЮ `δ(t)`?

ЛАНЦЮГ, ЩО ПРИВІВ СЮДИ (С101):
  · `allm` (істинні m,C щокроку) = 0.0913% ⟹ стеля класу = #5 місце
  · `mm` (істинна m, наша C) ×2.491 · `full` (істинна C) ×1.006 ⟹ УСЕ В `m`
  · одношагова похибка `E[ReLU]` при ТОЧНИХ (m,s) = 0.107…0.34%
  · наша база 0.5386% ⟹ **ПІДСИЛЕННЯ 4.57**
⟹ винна не форма й не моменти, а ПІДСИЛЕННЯ малої похибки. А оператор
підсилення ми ЗНАЄМО явно: `∂m_{i+1}[j]/∂m_i[k] = W[k,j]·Φ(t_j)`.

ІДЕЯ: якщо систематична частина `δ = E[ReLU]_істинне − (mΦ+sφ)` є УНІВЕРСАЛЬНОЮ
функцією безрозмірного `t = m/s` (у одиницях `s`), її можна виміряти ОДИН РАЗ
на банку і ПРОПАГУВАТИ через відомий оператор. Ціна — матвектор на шар.
🚨 ЦЕ НЕ ЕДЖВОРТ: там РЯД із гейтом збіжності (`a₂ = 2.07` при t=3.5, sub110
дав ×0.80). Тут — ЕМПІРИЧНА таблиця без ряду. І ми жодного разу не ПРОПАГУВАЛИ
корекцію — усі 100 сесій виправляли крок ЛОКАЛЬНО.

МІРЯЄМО: `d_j = (relu_true_j − relu_gauss_j)/s_j` проти `t_j`, і питаємо:
  (1) чи лягає на ОДНУ криву для різних ШАРІВ?
  (2) чи лягає на одну криву для різних МЕРЕЖ?
  (3) яку частку дисперсії `d` пояснює крива `δ(t)` (це і є стеля корекції)
🔒 КОНТРОЛЬ: та сама R² для ПЕРЕМІШАНИХ `t` ⟹ мусить дати ≈0.
🔒 ПЛОМБА ШАРУ 0: там `p` гаусів ТОЧНО ⟹ `d ≈ 0` (шум), і крива мусить бути
   плоскою біля нуля — інакше ми міряємо артефакт вибірки, а не негаусовість.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒 шар 0 дасть |d| у ~10× менше за шар 16                        80%
 (б) крива δ(t) буде схожою МІЖ МЕРЕЖАМИ (розкид < 20%)               65%
 (в) але РІЗНОЮ між шарами (бо γ₃ росте з глибиною)                   60%
 (г) R² від t на глибокому шарі > 0.5                                 45%
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

W_ = 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
CHUNK = 16384
LAYERS = [0, 2, 8, 16, 24, 31]
BINS = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, 32, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"delta101 · {NNET} мереж × {2*NS} семплів (антитетика)", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def one_net(net):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(32)]
    rng = cp.random.default_rng(2024 + net)
    S1 = cp.zeros((32, W_), dtype=cp.float64)
    S2 = cp.zeros((32, W_), dtype=cp.float64)
    RL = cp.zeros((32, W_), dtype=cp.float64)
    n = 0
    while n < NS:
        nb = min(CHUNK, NS - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(32):
            p = a @ Wg[i]
            p64 = p.astype(cp.float64)
            S1[i] += p64.sum(0)
            S2[i] += (p64 * p64).sum(0)
            RL[i] += cp.maximum(p64, 0.0).sum(0)
            if i < 31:
                a = cp.maximum(p, 0.0)
        n += nb
    N = 2 * NS
    m, r = S1 / N, RL / N
    s = cp.sqrt(cp.maximum(S2 / N - m * m, 1e-30))
    t = m / s
    g = m * ndtr(t) + s * phi(t)
    return t, (r - g) / s                      # безрозмірна похибка δ


T, D = [], []
for net in range(NNET):
    t, d = one_net(net)
    T.append(t)
    D.append(d)
    print(f"  мережа {net} готова", flush=True)

print(f"\n── δ(t) ПО СМУГАХ `t`, у одиницях `s` (×1e3) ──")
hdr = "".join(f"{f'[{BINS[i]},{BINS[i+1]})':>12s}" for i in range(len(BINS) - 1))
print(f"{'шар':>4s}{hdr}")
for L in LAYERS:
    cells = []
    for i in range(len(BINS) - 1):
        vals = []
        for k in range(NNET):
            msk = (T[k][L] >= BINS[i]) & (T[k][L] < BINS[i + 1])
            if int(msk.sum()) > 0:
                vals.append(float(D[k][L][msk].mean()))
        cells.append(f"{1e3*np.mean(vals):>12.3f}" if vals else f"{'—':>12s}")
    print(f"{L:>4d}" + "".join(cells), flush=True)

print(f"\n── РОЗКИД МІЖ МЕРЕЖАМИ (sd/|сер| по смузі t∈[1,2)) ──")
for L in LAYERS:
    vals = []
    for k in range(NNET):
        msk = (T[k][L] >= 1.0) & (T[k][L] < 2.0)
        if int(msk.sum()) > 0:
            vals.append(float(D[k][L][msk].mean()))
    if len(vals) > 1:
        mu, sd = np.mean(vals), np.std(vals, ddof=1)
        print(f"  шар {L:>2d}: сер {1e3*mu:>8.3f}e-3 · sd {1e3*sd:>8.3f}e-3 "
              f"· розкид {abs(sd/mu) if abs(mu)>1e-12 else float('nan'):>6.2f}")

print(f"\n── R²: скільки дисперсії `δ` пояснює ГЛАДКА функція t (кубічний фіт) ──")
rng = np.random.default_rng(5)
for L in LAYERS:
    tt = np.concatenate([cp.asnumpy(T[k][L]) for k in range(NNET)])
    dd = np.concatenate([cp.asnumpy(D[k][L]) for k in range(NNET)])
    A = np.column_stack([np.ones_like(tt), tt, tt ** 2, tt ** 3])
    b, *_ = np.linalg.lstsq(A, dd, rcond=None)
    r2 = 1.0 - np.sum((dd - A @ b) ** 2) / max(np.sum((dd - dd.mean()) ** 2), 1e-30)
    sh = tt[rng.permutation(len(tt))]
    As = np.column_stack([np.ones_like(sh), sh, sh ** 2, sh ** 3])
    bs, *_ = np.linalg.lstsq(As, dd, rcond=None)
    r2s = 1.0 - np.sum((dd - As @ bs) ** 2) / max(np.sum((dd - dd.mean()) ** 2), 1e-30)
    print(f"  шар {L:>2d}: R² = {r2:>7.4f}   🔒перемішаний контроль {r2s:>7.4f}",
          flush=True)
