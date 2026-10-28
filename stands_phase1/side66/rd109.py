"""rd109 — ШВИДКІСТЬ-СПОТВОРЕННЯ: скільки бітів про `W` споживає відповідь.

НОВА ВІСЬ СТЕЛІ. Усі попередні стелі мали форму «підстав ТОЧНЕ значення
внутрішньої величини». Ця — ортогональна: «обмеж, скільки інформації про `W`
методу дозволено прочитати».

    D(b) = ‖E[out(W_b)] − E[out(W)]‖ / ‖E[out(W)]‖ ,   W_b = W у b бітах

Це НИЖНЯ МЕЖА для будь-якого методу, що бачить лише `b` біт про кожну вагу —
незалежно від того, що він із ними робить. Перетворює наш емпіричний провал
«дешевого дескриптора мережі не існує» (гіпермережа, corr = 0.0012) на межу
з числом.

🔑 ПАРНІСТЬ ОБОВ'ЯЗКОВА: обидві мережі женуться на ТИХ САМИХ `z`, тож шум МК
   майже цілком скорочується і `D` міряється далеко нижче за шумову підлогу
   окремого прогону (0.29% при N=8192).

ДВА КВАНТУВАЧІ, і різниця між ними — це і є «наскільки далеко RTN від R(D)»:
   rtn   округлення до найближчого, шкала = max|w| по стовпцю (стандарт)
   opt   те саме, але з обрізанням, підібраним під МІНІМУМ MSE ваг

🔒 ПЛОМБИ
 S1 b = 24 ⟹ D мусить бути ≈0 (квантування практично тотожне)
 S2 РОЗДІЛЬНІСТЬ: для кожного b рахується власна стандартна похибка
    se = sqrt(Σ Var(out_q − out)/N)/‖E[out]‖. Вирок лише там, де D ≫ se.
 S3 відносна похибка ваг мусить падати як 2^(−b)

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (з переносу збурення 1% → 0.018% на шар)
    D(8) ≈ 0.04%  ·  D(4) ≈ 0.6%  ·  D(2) ≈ 2.5%
    точка, де D дорівнює похибці замикання (0.53%), лежить при b ≈ 4     55%
    `opt` б'є `rtn` щонайменше на один біт                               60%
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, "/home/trim/gpupkg")
os.environ.setdefault("CUPY_GPU_MEMORY_LIMIT", "1073741824")   # 1 ГБ, карти зайняті

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
BITS = (2, 3, 4, 5, 6, 8, 10, 12, 24)
CLIPS = (0.35, 0.45, 0.55, 0.70, 0.85, 1.00)
OUT = f"/home/trim/wblogs/rd109_N{NSAMP}.json"

import cupy as cp                                             # noqa: E402

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 ВИДНО НЕ ОДНУ КАРТУ — вихід.", flush=True)
    sys.exit(2)
free, tot = cp.cuda.runtime.memGetInfo()
print(f"rd109 · {NNET} мереж · N={NSAMP} · вільно на карті "
      f"{free/2**20:.0f} МБ із {tot/2**20:.0f}", flush=True)

Wall = []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall.append(np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_))
Wall = np.concatenate(Wall, 0)


def quant(w, b, clip):
    """симетричне RTN по СТОВПЦЯХ (стовпець = входи одного нейрона)."""
    lv = 2 ** (b - 1) - 1
    s = clip * cp.max(cp.abs(w), 0, keepdims=True) / lv
    s = cp.maximum(s, 1e-300)
    return cp.clip(cp.round(w / s), -lv, lv) * s


def fwd(W, z):
    a = z
    for i in range(DEPTH):
        a = cp.maximum(a @ W[i], 0.0)
    return a


rows = []
for net in range(NNET):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float64))
         for L in range(DEPTH)]
    z = cp.random.default_rng(4400 + net).standard_normal(
        (NSAMP, W_), dtype=cp.float64)
    out = fwd(W, z)
    mo = out.mean(0)
    nrm = float(cp.sqrt(cp.sum(mo * mo)))
    r = {"rtn": {}, "opt": {}, "se": {}, "werr": {}, "clip": {}}
    for b in BITS:
        best = None
        for c in CLIPS:
            Wq = [quant(W[L], b, c) for L in range(DEPTH)]
            we = float(sum(float(cp.sum((Wq[L] - W[L]) ** 2))
                           for L in range(DEPTH))
                       / sum(float(cp.sum(W[L] ** 2)) for L in range(DEPTH)))
            oq = fwd(Wq, z)
            d = float(cp.sqrt(cp.sum((oq.mean(0) - mo) ** 2))) / nrm
            se = float(cp.sqrt(cp.sum((oq - out).var(0)) / NSAMP)) / nrm
            if c == 1.00:
                r["rtn"][str(b)] = d
                r["werr"][str(b)] = math.sqrt(we)
                r["se"][str(b)] = se
            if best is None or we < best[0]:
                best = (we, d, c)
            del Wq, oq
        r["opt"][str(b)] = best[1]
        r["clip"][str(b)] = best[2]
    rows.append(r)
    print(f"  ✅ {net}: D(8) {100*r['rtn']['8']:.4f}% · D(4) "
          f"{100*r['rtn']['4']:.3f}% · D(2) {100*r['rtn']['2']:.2f}% · "
          f"🔒b=24 {100*r['rtn']['24']:.2e}% · se(4) {100*r['se']['4']:.4f}%",
          flush=True)
    del W, z, out
    cp.get_default_memory_pool().free_all_blocks()

json.dump(rows, open(OUT, "w"))
g = lambda k, b: np.array([r[k][str(b)] for r in rows])       # noqa: E731
print("\n  b     D(b) RTN        D(b) з підібраним обрізанням   роздільність se"
      "     похибка ваг")
for b in BITS:
    print(f" {b:>3}   {100*g('rtn',b).mean():9.4f}%   {100*g('opt',b).mean():9.4f}%"
          f"  (clip {np.median(g('clip',b)):.2f})   {100*g('se',b).mean():8.4f}%"
          f"   {100*g('werr',b).mean():7.3f}%")
print("\n🔒 S1  b=24 ⟹ D = %.2e%% (мусить бути ≈0)" % (100 * g('rtn', 24).mean()))
w = g('werr', 4).mean() / g('werr', 8).mean()
print(f"🔒 S3  похибка ваг b=4 проти b=8: ×{w:.1f} (для 2^(−b) очікується ×16)")
print("\n📏 для порівняння: косинусна похибка нашого замикання 0.53%")
