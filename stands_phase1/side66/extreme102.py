"""extreme102 — ІДЕЯ ЮЗЕРА «МІРЯТИ НА ЕКСТРЕМАЛЬНИХ / ПОЛАМАНИХ СЕМПЛАХ».

КОНКРЕТИЗАЦІЯ. Однорідність `out(cz) = c·out(z)` (🔒7.45e-15) означає, що
ДОВЖИНА `z` не створює екстремальності — лише НАПРЯМОК. Тому «поламаний семпл»
має сенс у трьох формах, і всі три вимірювані:

  A. ХВІСТ РОЗПОДІЛУ `out(z)`: чи домінують кілька семплів у дисперсії МК?
     Якщо так — робастна оцінка (вінзоризація) може виграти в MSE, попри зсув.
     Досі НЕ пробувано жодного разу. `out` — СЕРЕДНЄ, воно хвосто-чутливе.
  B. ЕКСТРЕМАЛЬНІ НАПРЯМКИ: `z` вздовж власних векторів `Cov(out)` і вздовж
     рядків `W₀` — чи там замикання ламається сильніше?
  C. ДОМЕН: записано, що замикання «за доменом МОВЧКИ бреше» (`diag(Cp)` стає
     від'ємною при `t≈11…22`). Чи досягають реальні семпли цієї зони?

ЩО ЦЕ МОЖЕ ДАТИ (кожне зі своїм вироком):
  A → якщо ексцес великий і топ-1% дає >20% дисперсії ⟹ вінзоризація/усічення
      знижує V сильніше, ніж додає зсуву  ⟹ ПРЯМИЙ виграш у 91% балу
  B → карта, де саме народжується похибка (доповнює back101: «біля виходу»)
  C → перевірка, чи наша реалізація взагалі коли-небудь бреше на робочих даних

🔒 ПЛОМБИ:
  (1) гаусів еталон: для N(0,1) ексцес 0, топ-1% дає рівно 1%·(запас)
  (2) 🔒 антитетичні пари вже вбивають НЕПАРНІ моменти ⟹ асиметрія має бути ≈0
  (3) вінзоризація з порогом ∞ мусить ТОЧНО відтворити звичайне середнє

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) ексцес проєкції out на t̂ буде < 1.0 (тобто майже гаусів)          70%
 (б) топ-1% семплів дасть < 8% дисперсії (гаусів рівень ~6%)           65%
 (в) вінзоризація НЕ виграє (зсув з'їсть виграш)                        75%
 (г) реальні семпли НЕ досягають доменної зони `t>11`                   90%

ВИРОК: (б) < 10% ⟹ хвоста немає, робастні оцінки мертві, і «поламані семпли»
як клас закрито. (б) > 20% ⟹ є прямий важіль на 91% балу.
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
NNET = min(NNET, tb.num_rows)
print(f"extreme102 · {NNET} мереж · {NS} семплів", flush=True)

res = []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    rng = cp.random.default_rng(31337 + net)
    outs, tmax = [], 0.0
    n = 0
    while n < NS // 2:
        nb = min(8192, NS // 2 - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        z = cp.concatenate([z, -z], axis=0)
        h = z
        for i in range(DEPTH):
            p = h @ Wg[i]
            if i in (8, 16, 31):                     # C: чи є доменна зона t>11
                s = p.std(0) + 1e-30
                tmax = max(tmax, float(cp.abs(p.mean(0) / s).max()))
            h = cp.maximum(p, 0.0)
        outs.append(h.astype(cp.float64))
        n += nb
    O = cp.concatenate(outs, axis=0)
    mu = O.mean(0)
    th = mu / cp.sqrt((mu * mu).sum())
    x = cp.asnumpy(O @ th)                            # проєкція виходу, скаляр
    d = x - x.mean()
    v = d.var()
    ku = float((d ** 4).mean() / (v * v) - 3.0)       # A: ексцес
    sk = float((d ** 3).mean() / v ** 1.5)            # 🔒 має бути ≈0 (антитет.)
    a = np.sort(d * d)[::-1]
    top1 = a[:max(1, len(a) // 100)].sum() / a.sum()  # A: частка дисперсії топ-1%
    top5 = a[:max(1, len(a) // 20)].sum() / a.sum()
    # 🔒 гаусів еталон на тій самій кількості точок
    gg = np.random.default_rng(7).standard_normal(len(d))
    ag = np.sort(gg * gg)[::-1]
    g1 = ag[:max(1, len(ag) // 100)].sum() / ag.sum()
    g5 = ag[:max(1, len(ag) // 20)].sum() / ag.sum()
    # A: чи виграє вінзоризація — на підвибірках бюджету сабміту
    NP, NR = 3072, 40
    err0, errw = [], []
    idx = np.random.default_rng(11 + net).integers(0, len(x), size=(NR, 2 * NP))
    Onp = cp.asnumpy(O)
    for r in range(NR):
        sub = Onp[idx[r]]
        m0 = sub.mean(0)
        pr = sub @ cp.asnumpy(th)
        lo, hi = np.percentile(pr, [0.5, 99.5])
        keep = (pr >= lo) & (pr <= hi)
        mw = sub[keep].mean(0)
        tru = cp.asnumpy(mu)
        err0.append(((m0 - tru) ** 2).mean())
        errw.append(((mw - tru) ** 2).mean())
    res.append(dict(net=net, ku=ku, sk=sk, t1=top1, t5=top5, g1=g1, g5=g5,
                    tmax=tmax, w=np.mean(err0) / np.mean(errw)))
    print(f"  мережа {net}: ексцес={ku:+.3f} асим={sk:+.4f} топ1%={100*top1:.2f}% "
          f"(гаус {100*g1:.2f}%) макс|t|={tmax:.1f} вінзор ×{res[-1]['w']:.4f}",
          flush=True)
    del Wg, O, outs
    cp.get_default_memory_pool().free_all_blocks()

g = lambda k: np.array([r[k] for r in res])
print("\n" + "=" * 70)
print(f"(а) ексцес проєкції out           {g('ku').mean():+.4f} ± {g('ku').std():.4f}")
print(f"🔒  асиметрія (антитет. ⟹ ≈0)     {g('sk').mean():+.6f}")
print(f"(б) частка дисперсії в топ-1%     {100*g('t1').mean():.2f}%   "
      f"🔒 гаусів еталон {100*g('g1').mean():.2f}%")
print(f"    частка дисперсії в топ-5%     {100*g('t5').mean():.2f}%   "
      f"🔒 гаусів еталон {100*g('g5').mean():.2f}%")
print(f"(в) вінзоризація 0.5%/99.5%       ×{g('w').mean():.4f} "
      f"(>1 = виграє)")
print(f"(г) макс |t| на шарах 8/16/31     {g('tmax').max():.1f}   "
      f"(доменна зона починається ~11)")
print("\nВИРОК: топ-1% < 10% ⟹ хвоста немає ⟹ робастні оцінки мертві")
