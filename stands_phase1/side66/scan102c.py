"""scan102c — 2D СКАН (gam, kap) НА 1000 МЕРЕЖАХ. ВИРОК ДЛЯ НОВОЇ РУЧКИ

ЗНАХІДКА scan102b: `gam` — множник на АРГУМЕНТ воріт, `Φ(g·t)`, `φ(g·t)`,
при незмінних множниках `mp` і `s`:
    m = mp·Φ(g·t) + s·φ(g·t)
оптимум g = 1.050, крос ×1.0246 / ×1.0210, знаковий 136/200, ціна НУЛЬ.

🔑 ЧОМУ ЦЕ НЕ κ І НЕ `_ALPHA`:
   κ множить `C` ⟹ міняє `s` І в `t`, І в множнику `s·φ` ⟹ це масштаб + форма
   `_ALPHA` множить готове `m` ⟹ ЧИСТИЙ масштаб, знімається косинусною метрикою
   `gam` міняє ЛИШЕ аргумент Φ,φ ⟹ рухає РЕЖИМ ВОРІТ, не чіпаючи ваги членів
   ⟹ третя незалежна ступінь свободи; при t≈4 `Φ(gt)−Φ(t)` мізерне, а
     `φ(gt)/φ(t) = exp(−t²(g²−1)/2)` — множник ~0.44 при g=1.05, t=4
   ⟹ фактично це РУЧКА НА ВАГУ ДРУГОГО ЧЛЕНА, але залежна від `t` НЕЛІНІЙНО

🚨 ЗАКОН С100: κ і `gam` МУСЯТЬ підбиратись РАЗОМ — калібрувальні константи
   не переносяться між конфігураціями (переніс α коштував 3.11e-05 замість
   2.21e-05). Тому 2D, а не два 1D.
🚨 Вирок — КРОС A→B на 1000 мережах, метрика КОСИНУСНА.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 (g=1.0, k=0.9985) відтворить 0.5032% (число scan102 на 200)      70%
     (на 1000 мережах популяція інша ⟹ допуск ±3%)
 (б) спільний оптимум дасть крос > ×1.03                                 45%
 (в) оптимальна κ ЗСУНЕТЬСЯ від 0.9985 (бо gam забирає частину роботи)    65%
 (г) знаковий буде > 600/1000                                            70%
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
KSIG = 8
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
GAMS = [1.000, 1.025, 1.040, 1.050, 1.060, 1.075, 1.100]
KAPS = [0.9955, 0.9985, 1.0000, 1.0020]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
Ws, Ts = [], []
for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
    if len(Ws) >= NNET:
        break
    tb = pa.ipc.open_stream(pa.memory_map(fp, "rb")).read_all()
    xw = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xw.type):
        xw = xw.flatten()
    Wa = np.asarray(xw).reshape(tb.num_rows, DEPTH, W_, W_)
    xt = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(xt.type):
        xt = xt.flatten()
    Ta = np.asarray(xt).reshape(tb.num_rows, W_)
    for r in range(tb.num_rows):
        if len(Ws) >= NNET:
            break
        Ws.append(np.ascontiguousarray(Wa[r], dtype=np.float32))
        Ts.append(Ta[r].astype(np.float64))
    del Wa, Ta
NNET = len(Ws)
print(f"scan102c · {NNET} мереж · сітка {len(GAMS)}×{len(KAPS)}", flush=True)
Tg = cp.asarray(np.array(Ts))


def closure(Wg, g, k):
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        deep = i > KSIG
        if deep:
            cpm = cpm * k
        s2 = cp.maximum(cp.diagonal(cpm), 1e-30)
        s = cp.sqrt(s2)
        mp = m @ w
        t = mp / s
        tg = t * g if deep else t
        Ph = ndtr(tg)
        ph = cp.exp(-0.5 * tg * tg) / SQ2PI
        m_new = mp * Ph + s * ph
        C = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        d = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        C = C + cp.diag(d - cp.diagonal(C))
        m = m_new
    return m


E = {}
for gi, g in enumerate(GAMS):
    for k in KAPS:
        acc = []
        for n in range(NNET):
            Wg = [cp.asarray(Ws[n][i]).astype(cp.float64) for i in range(DEPTH)]
            acc.append(closure(Wg, g, k))
            del Wg
        M = cp.stack(acc)
        num = (M * Tg).sum(1) ** 2
        den = (M * M).sum(1) * (Tg * Tg).sum(1)
        E[(g, k)] = cp.asnumpy(cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0)))
        del acc, M
        cp.get_default_memory_pool().free_all_blocks()
    print(f"  gam={g} готова", flush=True)

IA, IB = np.arange(0, NNET, 2), np.arange(1, NNET, 2)
B = (1.000, 0.9985)
base = float(E[B].mean())
print("\n" + "=" * 76)
print(f"КОСИНУСНА похибка, база (g=1.000, k=0.9985) = {100*base:.4f}%")
print(f"{'gam\\kap':>9}" + "".join(f"{k:>12.4f}" for k in KAPS))
print("-" * 76)
for g in GAMS:
    print(f"{g:>9.3f}" + "".join(f"{100*float(E[(g,k)].mean()):>11.4f}%" for k in KAPS))
print("-" * 76)
bestA = min(E, key=lambda kk: E[kk][IA].mean())
bestB = min(E, key=lambda kk: E[kk][IB].mean())
bestAll = min(E, key=lambda kk: E[kk].mean())
e_ab = float(E[bestA][IB].mean())
e_ba = float(E[bestB][IA].mean())
win = int((E[bestAll] < E[B]).sum())
print(f"in-sample оптимум {bestAll}  {100*float(E[bestAll].mean()):.4f}%  "
      f"×{base/float(E[bestAll].mean()):.4f}  знаковий {win}/{NNET}")
print(f"крос A→B: підбір {bestA} ⟹ {100*e_ab:.4f}%  "
      f"×{float(E[B][IB].mean())/e_ab:.4f}")
print(f"крос B→A: підбір {bestB} ⟹ {100*e_ba:.4f}%  "
      f"×{float(E[B][IA].mean())/e_ba:.4f}")
gcr = 0.5 * (float(E[B][IB].mean()) / e_ab + float(E[B][IA].mean()) / e_ba)
V, E2 = 1.9227e-06, 2.3385e-05
new = V * (E2 / gcr ** 2) / (V + E2 / gcr ** 2)
print(f"\nУ БАЛІ при кросі ×{gcr:.4f}: {V*E2/(V+E2):.4e} → {new:.4e} "
      f"= ×{(V*E2/(V+E2))/new:.4f}")
