"""switch101 — СКІЛЬКИ НЕЙРОНІВ РЕАЛЬНО ПЕРЕМИКАЮТЬСЯ? (перевірка суперечності)

СУПЕРЕЧНІСТЬ У НАШИХ ЗАПИСАХ:
  «медіана |t| = 3.50» ⟹ типовий нейрон майже завжди по ОДИН бік нуля
     (для гаусового `p` це P(перемикання) ≈ 0.02%)
  «мертвих нейронів 11.2%» ⟹ решта 89% таки відкриваються
  «R² лінійної апроксимації = 0.222» ⟹ мережа ДАЛЕКО не лінійна
Якщо маски майже фіксовані, `R²` мав би бути ≈1. Отже медіана `|t|` НЕ описує
розподіл: має бути широкий хвіст малих `|t|`.

ЧОМУ ЦЕ ВАЖЛИВО (гейт «тіло чи хвіст», С101): якщо перемикачів МАЛО, мережа
розкладається на ЛІНІЙНУ частину (фіксована маска) + поправку від жменьки
нейронів ⟹ це СТРУКТУРНА редукція, а не моментна, і вона поза класом, чия
стеля щойно виміряна як #5.

МІРЯЄМО (на кожному шарі):
    q_j = P(p_j > 0)  для кожного нейрона
    частки: q<0.01 (мертвий) · q>0.99 (завжди живий) · 0.01≤q≤0.99 (ПЕРЕМИКАЧ)
    внесок перемикачів у дисперсію виходу — через «заморожування» маски:
        out_frozen = мережа з маскою, зафіксованою на найімовірнішому значенні
    🔒 і скільки цієї дисперсії пояснює лінійна модель з ЗАМОРОЖЕНОЮ маскою

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) частка перемикачів (0.01…0.99) буде 40…80%                       55%
 (б) медіана |t| виявиться НЕ 3.5, а помітно меншою                    45%
 (в) заморожена маска дасть похибку виходу > 20%                       70%
 (г) розподіл |t| буде важкохвостим (сер |t| >> медіана або навпаки)    60%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

W_ = 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 65536
CHUNK = 8192
LAYERS = [0, 1, 2, 4, 8, 16, 24, 30, 31]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, 32, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"switch101 · {NNET} мереж × {NS} семплів", flush=True)

accQ = {L: cp.zeros(4) for L in LAYERS}
accT = {L: [] for L in LAYERS}
frz = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(32)]
    rng = cp.random.default_rng(101 + net)
    OP = cp.zeros((32, W_), dtype=cp.float64)
    S1 = cp.zeros((32, W_), dtype=cp.float64)
    S2 = cp.zeros((32, W_), dtype=cp.float64)
    n = 0
    while n < NS:
        nb = min(CHUNK, NS - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = z
        for i in range(32):
            p = a @ Wg[i]
            OP[i] += (p > 0).sum(0, dtype=cp.float64)
            p64 = p.astype(cp.float64)
            S1[i] += p64.sum(0)
            S2[i] += (p64 * p64).sum(0)
            if i < 31:
                a = cp.maximum(p, 0.0)
        n += nb
    q = OP / NS
    m = S1 / NS
    s = cp.sqrt(cp.maximum(S2 / NS - m * m, 1e-30))
    t = cp.abs(m / s)
    for L in LAYERS:
        qq = q[L]
        accQ[L] += cp.asarray([float((qq < 0.01).mean()),
                               float((qq > 0.99).mean()),
                               float(((qq >= 0.01) & (qq <= 0.99)).mean()),
                               float(((qq >= 0.1) & (qq <= 0.9)).mean())])
        accT[L].append((float(cp.median(t[L])), float(cp.mean(t[L])),
                        float((t[L] < 1.0).mean())))
    # 🔒 заморожена маска: беремо D = (q > 0.5), ганяємо ЛІНІЙНУ мережу
    D = (q > 0.5).astype(cp.float32)
    rng = cp.random.default_rng(101 + net)
    num = den = 0.0
    n = 0
    while n < min(NS, 16384):
        nb = min(CHUNK, min(NS, 16384) - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = z
        b = z
        for i in range(32):
            p = a @ Wg[i]
            pf = b @ Wg[i]
            if i < 31:
                a = cp.maximum(p, 0.0)
                b = pf * D[i]
        o = cp.maximum(p, 0.0).mean(1)
        of = cp.maximum(pf * D[31], 0.0).mean(1)
        num += float(((o - of) ** 2).sum())
        den += float((o ** 2).sum())
        n += nb
    frz.append(math.sqrt(num / max(den, 1e-30)))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'шар':>4s}{'мертві':>9s}{'завжди':>9s}{'ПЕРЕМИКАЧІ':>12s}"
      f"{'q∈[.1,.9]':>11s}{'мед |t|':>9s}{'сер |t|':>9s}{'|t|<1':>8s}")
print("=" * 71)
for L in LAYERS:
    v = accQ[L] / NNET
    tt = np.array(accT[L]).mean(0)
    print(f"{L:>4d}{100*float(v[0]):>8.2f}%{100*float(v[1]):>8.2f}%"
          f"{100*float(v[2]):>11.2f}%{100*float(v[3]):>10.2f}%"
          f"{tt[0]:>9.3f}{tt[1]:>9.3f}{100*tt[2]:>7.1f}%")
print("\n🔒 ЗАМОРОЖЕНА МАСКА (лінійна мережа з D=q>0.5): відносна похибка виходу "
      f"{100*float(np.mean(frz)):.2f}%")
