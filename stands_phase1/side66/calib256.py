"""calib256 — 🔒 ПЛОМБА №1: чи відтворює МІЙ стенд відомі числа С88 при n=256?

Без цієї пломби вся калібровка ширини безглузда: якщо при n=256 мій стенд дає
не ті числа, я не дізнаюсь, що винне — ширина чи протокол.

ДЖЕРЕЛА (навмисно РІЗНІ, закон «сіди істини й оракула розділяти»):
  оракул  — `keenanpepper/arc-whestbench-higher-moments-2026`, `mean`+`M11`,
            пошарово, N=1e8
  істина  — публічний датасет, `final_means`, N=1e9
  ваги    — публічний датасет, `weights`
  адресація — `mlp_id == global_index` (пастка С97: `mlp_name` НЕ збігається)

ЦІЛЬОВІ ЧИСЛА (С88, 32 мережі): m ×0.912 · діаг ×0.243 · поза ×0.634 ·
                                 C повна ×0.725 · m+C ×20.97

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: пломба пройде (усі п'ять у межах ×1.5)  —  35%
"""
import glob
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib99 import run_variant, loo_alpha_err, VARIANTS, DEPTH   # noqa: E402

SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/")
# 🚨 спліт `full`, НЕ `mini`: датасет вищих моментів адресує саме його
# (перевірено прямо — `mini` рядок 0 відрізняється від `mlp_00000` на 95%)
SRC = sorted(glob.glob(SRC + "*"))[0] + "/prepared/full"
HM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hm/full")

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
# режим шарда: `calib256.py N START OUT.npz` — рахує мережі [START, START+N)
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
DUMP = sys.argv[3] if len(sys.argv) > 3 else None

print(f"calib256 · пломба №1 · {NNET} мереж · оракул N=1e8, істина N=1e9",
      flush=True)
t0 = time.time()

P = {v: [] for v, _ in VARIANTS}
T = []
seal3 = []
seal_align = []

for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    x = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, 256, 256)
    y = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(y.type):
        y = y.flatten()
    FM = np.asarray(y).reshape(tb.num_rows, 256)
    ids = tb.column("mlp_id").to_pylist()
    for i in range(tb.num_rows):
        if len(T) >= NNET:
            break
        if ids[i] < START or ids[i] >= START + NNET:
            continue
        p = os.path.join(HM, f"mlp_{ids[i]:05d}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        assert int(d["global_index"]) == ids[i], "адресація!"
        TM = [d["mean"][L].astype(np.float64) for L in range(DEPTH)]
        M11 = d["M11"].astype(np.float64)
        TC = [M11[L] - np.outer(TM[L], TM[L]) for L in range(DEPTH)]
        TC = [0.5 * (c + c.T) for c in TC]
        W = [np.ascontiguousarray(Wall[i][L], dtype=np.float64)
             for L in range(DEPTH)]
        tm = FM[i].astype(np.float64)
        sc = float(np.linalg.norm(tm))
        # 🔒 узгодженість двох джерел істини (N=1e8 проти N=1e9)
        seal_align.append(float(np.linalg.norm(TM[DEPTH - 1] - tm) / sc))
        for name, chan in VARIANTS:
            P[name].append(run_variant(W, 256, chan, TM, TC))
        fo = run_variant(W, 256, {"m", "diag", "off"}, TM, TC, upto=DEPTH)
        seal3.append(float(np.linalg.norm(fo - TM[DEPTH - 1]) / sc))
        T.append(tm)
        print(f"  мережа {len(T):>2d} (id {ids[i]:>3d})  "
              f"{time.time()-t0:.0f} с", flush=True)
    if len(T) >= NNET:
        break

if DUMP:
    np.savez_compressed(DUMP, T=np.array(T), seal3=np.array(seal3),
                        seal_align=np.array(seal_align),
                        **{f"P{j}": np.array(P[v])
                           for j, (v, _) in enumerate(VARIANTS)})
    print(f"  збережено {DUMP}  ({len(T)} мереж, {time.time()-t0:.0f} с)")
    sys.exit(0)

print("\n" + "=" * 92)
print(f"🔒 ПЛОМБА №1 · n = 256 · {len(T)} мереж")
print("=" * 92)
print(f"🔒P3 повний оракул (мусить бути 0): {max(seal3):.3e}")
print(f"🔒 узгодженість N=1e8 vs N=1e9 (шум істини): "
      f"{100*float(np.median(seal_align)):.4f}%")
raw = float(np.median([float(np.linalg.norm(p - t) / np.linalg.norm(t))
                       for p, t in zip(P["база"], T)]))
print(f"🔒 сира база БЕЗ α: {100*raw:.4f}%  (anchor97 дав 0.9138%)")
base_med, base_rms, _ = loo_alpha_err(P["база"], T)
TARGET = {"оракул m": 0.912, "оракул діаг C": 0.243,
          "оракул позадіаг C": 0.634, "оракул C повна": 0.725,
          "оракул m + C": 20.97}
print(f"\n{'варіант':>20s}{'медіана':>11s}{'RMS':>11s}"
      f"{'k(мед)':>10s}{'k(RMS)':>10s}{'ціль С88':>11s}{'промах':>10s}")
for name, chan in VARIANTS:
    med, rms, _ = loo_alpha_err(P[name], T)
    km = base_med / max(med, 1e-300)
    kr = base_rms / max(rms, 1e-300)
    tg = TARGET.get(name)
    s1 = f"{tg:.3f}" if tg else "—"
    s2 = f"×{max(km,tg)/min(km,tg):.2f}" if tg else "—"
    print(f"{name:>20s}{100*med:>10.4f}%{100*rms:>10.4f}%"
          f"{km:>10.3f}{kr:>10.3f}{s1:>11s}{s2:>10s}", flush=True)
print(f"\n  час {time.time()-t0:.0f} с")
