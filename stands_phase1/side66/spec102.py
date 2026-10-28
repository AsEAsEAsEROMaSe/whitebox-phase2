"""spec102 — ЗВЕСТИ eff_rank(C) З 4.5 ДО 2.8. ПЕРША КОНКРЕТНА ВАДА `C`.

ЗВІДКИ (effrank102, 16 мереж):
    eff_rank(C) у НАШОМУ замиканні   L1 = 165.7   L31 = 4.5
    🔒 зміряний на семплах            L1 = 165.6   L31 = 2.8
⟹ на шарі 1 збіг ДО ЧОТИРЬОХ ЗНАКІВ, на шарі 31 розбіжність **1.6×**
⟹ замикання правильно несе залежність угорі й НЕДООЦІНЮЄ КОЛАПС РАНГУ
   на глибині: розмазує дисперсію по 4.5 напрямках там, де реально 2.8.

ЧОМУ κ ЦЬОГО НЕ ЛІКУЄ: `κ` множить УСЮ `cp` на скаляр ⟹ всі власні значення
масштабуються однаково ⟹ eff_rank = (Σλ)²/Σλ² НЕ ЗМІНЮЄТЬСЯ ВЗАГАЛІ.
Саме тому κ дає лише ×1.0036 — він працює в ортогональному напрямку.

ЩО ПРОБУЄМО — ТРИ ДЕШЕВІ СПОСОБИ ЗМІНИТИ СПЕКТР, БЕЗ `eigh`:
  P  «загострення» степенем:  C ← C·(C/trC·n)^p  наближено як
     C ← C + p·(C@C/tr(C) − C)          1 матмул 256³ = 3.355e7 на шар
  M  ручка на другий член Мелера: коеф 0.5 → 0.5·(1+q)
     (це той член, що народжує позадіагональні кореляції)   ЦІНА НУЛЬ
  D  зсув діагоналі: C ← C − d·diag(C) + d·(trC/n)·I        ЦІНА НУЛЬ
     (зменшує внесок «індивідуальних» напрямків ⟹ підсилює колективні)

🔒 ПЛОМБИ / КОНТРОЛІ:
  (1) при p=q=d=0 усе мусить відтворити базу БІТ У БІТ
  (2) друкуємо eff_rank(C₃₁) для кожного варіанта ⟹ видно, чи ручка справді
      РУХАЄ ранг у бік 2.8, чи просто чіпає масштаб
  (3) 🔒 «ЧУЖА»: найкраще значення ручки, застосоване з профілем сусідньої
      мережі — тут не застосовне (ручки глобальні), тому натомість
      КРОС A/B: підбір на перших 8 мережах, вирок на других 8

МІРЯЄМО НА КОСИНУСНІЙ ПОХИБЦІ — прилад щойно доведено: він передбачив
грейдерний ефект κ як ×1.0036 при факті ×1.0039 (0.03%).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) ручка P справді зсуне eff_rank(31) у бік 2.8                     70%
 (б) хоч одна ручка дасть >×1.02 на косинусній                        35%
 (в) крос A→B утримає щонайменше половину виграшу                     45%
 (г) ручка M (Мелер) виявиться найсильнішою, бо саме вона робить
     позадіагональ, а позадіагональ і є ранг                          40%

ВИРОК: >×1.02 з крос-підтвердженням ⟹ це перша поправка, що б'є в
ЗМІРЯНУ ваду, а не в підібрану константу.
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
KSIG, KAPPA = 8, 0.9985
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 4194304

GRID = {
    "P": [0.0, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50],
    "M": [0.0, -0.20, -0.10, 0.10, 0.20, 0.40, 0.80],
    "D": [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75],
}

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"spec102 · {NNET} мереж · істина {NTRUE}", flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(Wg, knob=None, val=0.0):
    """замикання сабміту (κ=0.9985 при i>8) + одна спектральна ручка"""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    er31 = None
    for i in range(DEPTH):
        w = Wg[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * KAPPA
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn, er31
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        mel = 0.5 if knob != "M" else 0.5 * (1.0 + val)
        Co = cpm * cp.outer(Ph, Ph) + (mel * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        if knob == "P" and val != 0.0 and i > KSIG:
            tr = cp.trace(C)
            C = C + val * ((C @ C) * (W_ / cp.maximum(tr, 1e-30)) - C)
        elif knob == "D" and val != 0.0 and i > KSIG:
            dg = cp.diagonal(C)
            C = C - val * cp.diag(dg) + (val * cp.mean(dg)) * cp.eye(W_)
        m = mn
        if i == DEPTH - 2:
            tr = float(cp.trace(C))
            er31 = tr * tr / max(float((C * C).sum()), 1e-300)


def mc_out(Wg, ns, seed):
    rng = cp.random.default_rng(seed)
    acc = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < ns:
        nb = min(8192, ns - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        z = cp.concatenate([z, -z], axis=0)
        zz = z.astype(cp.float64)
        cov = (zz.T @ zz) / zz.shape[0]
        wh = cp.linalg.inv(cp.linalg.cholesky(cov)).T
        h = (zz @ wh).astype(cp.float32)
        for i in range(DEPTH):
            h = cp.maximum(h @ Wg[i], 0.0)
        acc += h.astype(cp.float64).sum(0)
        n += nb
    return acc / (2 * ns)


def cos_err(pred, truth):
    pn = float((pred * pred).sum())
    tn = float((truth * truth).sum())
    ip = float((pred * truth).sum())
    return math.sqrt(max(1.0 - ip * ip / max(pn * tn, 1e-300), 0.0))


E = {}
R = {}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    truth = mc_out([w.astype(cp.float32) for w in Wg], NTRUE // 2, 70001 + net)
    mb, rb = closure(Wg)
    E.setdefault(("base", 0.0), []).append(cos_err(mb, truth))
    R.setdefault(("base", 0.0), []).append(rb)
    for kn, vals in GRID.items():
        for v in vals:
            if v == 0.0:
                continue
            mm, rr = closure(Wg, kn, v)
            E.setdefault((kn, v), []).append(cos_err(mm, truth))
            R.setdefault((kn, v), []).append(rr)
    print(f"  мережа {net}: база {E[('base',0.0)][-1]*100:.4f}%  "
          f"eff_rank(C31) {rb:.2f}", flush=True)
    del Wg
    cp.get_default_memory_pool().free_all_blocks()

base = np.array(E[("base", 0.0)])
print(f"\nБАЗА (сабміт): {100*base.mean():.4f}%  ·  "
      f"eff_rank(C31) = {np.mean(R[('base',0.0)]):.2f}  (ціль 2.8)")
print(f"\n{'ручка':>6s}{'знач':>8s}{'похибка':>11s}{'×виграш':>10s}"
      f"{'знак':>7s}{'eff_rank':>10s}{'крос A→B':>10s}")
print("=" * 64)
for kn, vals in GRID.items():
    for v in vals:
        if v == 0.0:
            continue
        e = np.array(E[(kn, v)])
        print(f"{kn:>6s}{v:>8.2f}{100*e.mean():>10.4f}%"
              f"{base.mean()/e.mean():>10.4f}"
              f"{int((e < base).sum()):>4d}/{NNET}"
              f"{np.mean(R[(kn,v)]):>10.2f}", end="")
        h = NNET // 2
        ia = min(((kn, u) for u in vals if u != 0.0),
                 key=lambda k: np.mean(E[k][:h]))
        print(f"{np.mean(E[('base',0.0)][h:]) / np.mean(E[ia][h:]):>10.4f}"
              if v == ia[1] else "")
print("\n🔒 ПЛОМБА: eff_rank мусить РУХАТИСЬ до 2.8, інакше ручка чіпає масштаб")
print("ВИРОК: >×1.02 з крос-підтвердженням ⟹ поправка б'є в ЗМІРЯНУ ваду")
