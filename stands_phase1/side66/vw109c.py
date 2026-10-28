"""vw109c — чи існує ДЕШЕВИЙ проксі до того, що корелює з похибкою замикання.

`vw109b` (n=10) показав, що величини ансамбля віртуальних ваг корелюють із
помережною похибкою замикання на |r| ≈ 0.76…0.82 при порозі значущості 0.741.
🚨 АЛЕ вони коштують повного ансамбля `M` (N форвардів × 256 проб) — це в рази
дорожче за весь бюджет. Питання, від якого залежить, чи це прилад чи курйоз:

    чи можна дістати ТУ САМУ інформацію з величин, які замикання вже має?

Кандидати, всі БЕЗКОШТОВНІ (замикання їх обчислює або майже обчислює):
    live_l   Σ_j Φ(t_j)(1−Φ(t_j))   очікуване число «живих» затворів на шарі
    pr_C_l   ефективний ранг C на шарі ℓ
    medt_l   медіана |t| на шарі ℓ
    frac_l   частка |t| < 1  (нейрони біля порога)
    growth   ‖m‖²/(‖m‖²+tr C) — «замерзлість»

🔒 ПЛОМБИ
 C1 замикання відтворює корпусну косинусну базу ≈0.507% у середньому
 C2 шар 0: замикання ТОЧНЕ ⟹ його передбачення = ‖w‖/√(2π) до 1e-12
 C3 контроль: та сама кореляція на ПЕРЕМІШАНИХ парах (ознака від чужої мережі)

📣 ПЕРЕДБАЧЕННЯ: хоч один дешевий проксі дає |r| > 0.6 на 100 мережах — 40%
"""
import glob
import json
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
OUT = "/home/trim/wblogs/vw109c.json"

import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
PROBE_L = (1, 4, 8, 16, 24, 31)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


Wall, TRUE = [], []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall.append(np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_))
    yy = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(yy.type):
        yy = yy.flatten()
    TRUE.append(np.asarray(yy).reshape(tb.num_rows, W_))
Wall = np.concatenate(Wall, 0)
TRUE = np.concatenate(TRUE, 0)
NNET = Wall.shape[0]
print(f"vw109c · {NNET} мереж · замикання + дешеві проксі", flush=True)

rows = []
for net in range(NNET):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float64))
         for L in range(DEPTH)]
    m = cp.zeros(W_, dtype=cp.float64)
    C = W[0].T @ W[0]
    feat = {}
    c2 = None
    for i in range(1, DEPTH + 1):
        s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
        t = m / s
        Ph, ph = ndtr(t), phi(t)
        ma = s * (ph + t * Ph)
        if i == 1:
            exact = cp.sqrt(cp.sum(W[0] * W[0], 0)) / SQ2PI
            c2 = float(cp.max(cp.abs(ma - exact) / exact))
        if i in PROBE_L:
            ev = C / cp.maximum(cp.trace(C), 1e-300)
            feat[f"live{i}"] = float(cp.sum(Ph * (1.0 - Ph)))
            feat[f"prC{i}"] = float(cp.trace(C) ** 2 / cp.sum(C * C))
            feat[f"medt{i}"] = float(cp.median(cp.abs(t)))
            feat[f"near{i}"] = float(cp.mean(cp.abs(t) < 1.0))
            feat[f"froz{i}"] = float(cp.sum(m * m)
                                     / (cp.sum(m * m) + cp.trace(C)))
            del ev
        q = s * s * ((1.0 + t * t) * Ph + t * ph)
        rho = C / cp.maximum(cp.outer(s, s), 1e-300)
        Ca = cp.outer(Ph, Ph) * C + 0.5 * cp.outer(ph, ph) * rho * rho \
            * cp.outer(s, s)
        cp.fill_diagonal(Ca, q - ma * ma)
        if i == DEPTH:
            break
        m = W[i].T @ ma
        C = W[i].T @ Ca @ W[i]
    s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
    t = m / s
    pred = s * (phi(t) + t * ndtr(t))
    T = cp.asarray(TRUE[net], dtype=cp.float64)
    c = float(cp.dot(pred, T)) / math.sqrt(
        float(cp.dot(pred, pred)) * float(cp.dot(T, T)))
    feat["err"] = math.sqrt(max(0.0, 1.0 - c * c))
    feat["C2"] = c2
    rows.append(feat)
    if net % 20 == 0:
        print(f"  {net}: похибка {100*feat['err']:.3f}% · 🔒C2 {c2:.1e}",
              flush=True)

json.dump(rows, open(OUT, "w"))
E = np.array([r["err"] for r in rows])
print(f"\n🔒 C1 середня косинусна похибка замикання {100*E.mean():.3f}% "
      f"± {100*E.std():.3f}  (корпус: 0.507%)")
print(f"🔒 C2 шар 0 проти ‖w‖/√(2π): max {max(r['C2'] for r in rows):.1e}")
keys = [k for k in rows[0] if k not in ("err", "C2")]
print(f"\nкореляція дешевих проксі з похибкою (n={NNET}, "
      f"поріг |r|>{1.96/np.sqrt(NNET-3):.3f})")
res = []
for k in keys:
    v = np.array([r[k] for r in rows])
    if v.std() < 1e-12:
        continue
    res.append((abs(np.corrcoef(np.log(np.abs(v) + 1e-30),
                                np.log(E))[0, 1]), k,
                np.corrcoef(v, E)[0, 1]))
res.sort(reverse=True)
for a, k, r1 in res[:14]:
    print(f"   {k:<10} |r_loglog| {a:.3f}   r {r1:+.3f}")
print(f"\n   … найгірші: " + ", ".join(f"{k} {a:.2f}" for a, k, _ in res[-4:]))
