"""fitreal96 — ФІТ λ НА СПРАВЖНІХ ОЗНАКАХ ЕСТИМАТОРА (не на репліці).

🚨 ПРИЧИНА: репліка в `fit96.py` дала ваги, які в естиматорі ЗРОБИЛИ ГІРШЕ
(2.0034e-06 проти 1.9265e-06) ⟹ ознаки репліки ≠ ознаки естиматора.
Тут ознаки виписані САМИМ естиматором (`dump107.py`, ksig=30) під час
`whest run`, тож розбіжності бути не може ЗА ПОБУДОВОЮ.
Розділ train/val — ПО МЕРЕЖАХ (закон С89: підбір на зрізі A, вирок на B).
"""
import glob
import sys

import numpy as np
import pyarrow as pa

DUMP = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
        "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/dump")
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
ALPHA, W = -0.00751, 0.0883

truth = {}
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    x = t.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
    fm = np.asarray(t.column("final_means").combine_chunks()
                    .flatten()).reshape(t.num_rows, 256)
    for i in range(t.num_rows):
        truth[tuple(np.round(w[i, 0, :2, :2].astype(np.float64), 10).ravel())] \
            = fm[i].astype(np.float64)
    if len(truth) > 1200:
        break
print(f"істина завантажена: {len(truth)} мереж")

REC = []
miss = 0
for fn in sorted(glob.glob(DUMP + "/*.npz")):
    d = np.load(fn)
    k = tuple(np.round(d["key"].astype(np.float64), 10).ravel())
    if k not in truth:
        miss += 1
        continue
    REC.append((d["mc"], d["v0"], d["md"], d["B"], truth[k]))
print(f"зіставлено {len(REC)} мереж, не знайдено {miss}")
if not REC:
    sys.exit("немає збігів — ключ не той")

nb = REC[0][3].shape[0]
print(f"базисних векторів: {nb} (= 4 × {nb // 4} шарів)")


def pred(rec, lam, ncols):
    mc, v0, md, B, y = rec
    ours = mc - v0 - (lam @ B[:ncols] if lam is not None else 0.0)
    return ours + W * ((1.0 + ALPHA) * md - ours)


def mse(idx, lam, ncols):
    tot = tn = 0.0
    for k in idx:
        r = REC[k]
        e = pred(r, lam, ncols) - r[4]
        tot += float((e * e).sum())
        tn += float((r[4] * r[4]).sum())
    return tot / tn


def fit(idx, rg, ncols):
    G = np.zeros((ncols, ncols))
    h = np.zeros(ncols)
    for k in idx:
        mc, v0, md, B, y = REC[k]
        Bs = B[:ncols] * (1.0 - W)
        r0 = pred(REC[k], None, ncols) - y
        G += Bs @ Bs.T
        h += Bs @ r0
    G += rg * np.eye(ncols) * np.trace(G) / ncols
    return np.linalg.solve(G, h)


n = len(REC)
tr, va = list(range(0, n, 2)), list(range(1, n, 2))
base = mse(va, None, nb)
print(f"\n🔒 БАЗА (λ=0, тобто МК+бленд без родин): val {base:.4e} "
      f"({base**0.5*100:.4f}%)")
print(f"{'шарів':>6s}{'гребінь':>9s}{'train':>11s}{'VAL':>11s}{'MSE ×':>8s}"
      f"{'бал':>12s}")
best = None
for nl in (8, 16, 24, 30):
    nc = 4 * nl
    for rg in (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0):
        w = fit(tr, rg, nc)
        a, b = mse(tr, w, nc), mse(va, w, nc)
        sc = 1.86762e-07 * b / base
        if best is None or b < best[0]:
            best = (b, nl, rg, w)
        print(f"{nl:>6d}{rg:>9.0e}{a**0.5*100:>10.4f}%{b**0.5*100:>10.4f}%"
              f"{base/b:>8.3f}{sc:>12.4e}")
b, nl, rg, w = best
print(f"\n🏆 НАЙКРАЩЕ: {nl} шарів, гребінь {rg:.0e} ⟹ val MSE ×{base/b:.3f}, "
      f"бал ≈ {1.86762e-07 * b / base:.4e}")
wm = w.reshape(nl, 4)
for fam, k in (("_LAM_S", 0), ("_LAM_A", 1), ("_LAM_Q", 2), ("_LAM_G", 3)):
    print(f"{fam} = (" + ", ".join(f"{wm[i, k]:.7f}" for i in range(nl)) + ")")

# ── ЧЕСНЕ ПОРІВНЯННЯ НА ВАЛІДАЦІЙНІЙ ПОЛОВИНІ ────────────────────────────
import re as _re
_src = open('/mnt/win_d/llm/White-Box/whest-starterkit/estimator_sub99.py').read()
_nat = []
for _k in ("_LAM_S", "_LAM_A", "_LAM_Q", "_LAM_G"):
    _m = _re.search(_k + r" = \(([^)]*)\)", _src, flags=_re.S)
    _nat.append([float(x) for x in _m.group(1).replace('\n', ' ').split(',')
                 if x.strip()])
_native = np.zeros(32)
for _i in range(8):
    for _f in range(4):
        _native[4 * _i + _f] = _nat[_f][_i]
print("\n══ ВІЧ-НА-ВІЧ на ВАЛІДАЦІЙНІЙ половині (50 мереж, фіт їх не бачив) ══")
_mn = mse(va, _native, 32)
print(f"  рідні ваги сабміту (8 шарів):   {_mn:.4e}  ({_mn**0.5*100:.4f}%)")
for _nl, _rg in ((8, 1.0), (16, 0.3), (30, 0.1)):
    _w = fit(tr, _rg, 4 * _nl)
    _mv = mse(va, _w, 4 * _nl)
    print(f"  фіт {_nl:2d} шарів (гребінь {_rg:.0e}):     {_mv:.4e}  "
          f"({_mv**0.5*100:.4f}%)   проти рідних ×{_mn/_mv:.3f}")
