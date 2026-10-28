"""base104 — ЧОМУ БАЗА СТЕНДУ 0.4384% ПРОТИ ГРЕЙДЕРНИХ 0.5070% (пункт 2 аудиту).

Розбіжність 13.5% псує КОЖНЕ переведення «стенд → бал» цієї сесії. Дві гіпотези:
   (П) ПОПУЛЯЦІЯ — 24 перші мережі `full` проти 50 мереж публічного спліту
   (К) КОНСТАНТИ — orac101 (С101) не мав `gam`, доданого у С102

Розділяємо прямо: та сама конфігурація на ДВОХ сплітах × ДВІ конфігурації.
   mini 100 мереж   (публічний спліт грейдера — його підмножина)
   full 100 мереж   (перші 100, з них 24 давали 0.4384%)
   gam = 1.048 (наш сабміт)   проти   gam = 1.0 (конфігурація до С102)

🔒 ПЛОМБИ: (1) mini з gam=1.048 має дати ≈0.5070%, якщо винна ПОПУЛЯЦІЯ
           (2) full з gam=1.0 має дати ≈0.5070%, якщо винні КОНСТАНТИ
           (3) розкид ПО МЕРЕЖАХ друкуємо — якщо він великий, різниця
               між сплітами може бути просто вибірковим шумом
"""
import glob
import math
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG, KAP = 8, 0.9985
ROOT = ("/home/trim/.cache/huggingface/hub/"
        "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
        "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 100
PACK = 20

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def stream(split, nmax):
    got = 0
    for fp in sorted(glob.glob(ROOT + split + "/data-*.arrow")):
        if got >= nmax:
            break
        for b in pa.ipc.open_stream(pa.memory_map(fp, "rb")):
            if got >= nmax:
                break
            xw = b.column("weights")
            while not pa.types.is_primitive(xw.type):
                xw = xw.flatten()
            xt = b.column("final_means")
            while not pa.types.is_primitive(xt.type):
                xt = xt.flatten()
            nb = b.num_rows
            wv = np.asarray(xw).reshape(nb, DEPTH, W_, W_)
            tv = np.asarray(xt).reshape(nb, W_)
            for j in range(min(nb, nmax - got)):
                yield np.array(wv[j], dtype=np.float32), \
                    np.array(tv[j], dtype=np.float64)
            got += min(nb, nmax - got)
            del xw, xt, wv, tv


def recur(Wb, gam):
    P = Wb.shape[0]
    m = cp.zeros((P, W_), dtype=cp.float64)
    C = cp.empty((P, W_, W_), dtype=cp.float32)
    C[:] = cp.eye(W_, dtype=cp.float32)
    dg = cp.arange(W_)
    for i in range(DEPTH):
        w = Wb[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w).astype(
            cp.float64)
        if i > KSIG:
            cpm = cpm * KAP
        s2 = cp.maximum(cpm[:, dg, dg], 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w.astype(cp.float64))
        u = gam * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn
        pv = (ph / s).astype(cp.float32)
        P32 = Ph.astype(cp.float32)
        c32 = cpm.astype(cp.float32)
        C2 = c32 * c32
        C2 *= 0.5
        C2 *= pv[:, :, None]
        C2 *= pv[:, None, :]
        c32 *= P32[:, :, None]
        c32 *= P32[:, None, :]
        c32 += C2
        del C2
        c32[:, dg, dg] = ((mp * mp + s2) * Ph + (mp * s) * ph
                          - mn * mn).astype(cp.float32)
        C, m = c32, mn
    return m


print(f"base104 · {NNET} мереж на спліт · пачка {PACK}", flush=True)
t0 = time.time()
RES = {}
for split in ("mini", "full"):
    per = {1.0: [], 1.048: []}
    bw, bt = [], []
    for wn, tn in stream(split, NNET):
        bw.append(wn)
        bt.append(tn)
        if len(bw) < PACK:
            continue
        Wb = cp.asarray(np.stack(bw))
        Tb = cp.asarray(np.stack(bt))
        for gam in (1.0, 1.048):
            mm = recur(Wb, gam)
            for a in range(len(bw)):
                t, x = Tb[a], mm[a]
                per[gam].append((float(t @ t)
                                 - float(x @ t) ** 2 / max(float(x @ x), 1e-300),
                                 float(t @ t)))
            del mm
        del Wb, Tb
        bw, bt = [], []
        cp.get_default_memory_pool().free_all_blocks()
    RES[split] = per
    print(f"  ✅ {split} ({time.time()-t0:.1f} с)", flush=True)

print(f"\n{'спліт':>6}{'gam':>7}{'мереж':>7}{'косинусна':>12}"
      f"{'медіана по мережах':>20}{'sd(log) по мережах':>20}")
for split in ("mini", "full"):
    for gam in (1.0, 1.048):
        p = RES[split][gam]
        agg = math.sqrt(sum(a for a, _ in p) / sum(b for _, b in p))
        ind = np.array([math.sqrt(a / b) for a, b in p])
        print(f"{split:>6}{gam:>7.3f}{len(p):>7}{100*agg:>11.4f}%"
              f"{100*np.median(ind):>19.4f}%{np.std(np.log(ind)):>20.4f}")

print("\n🔒 ПЛОМБА: грейдер на 50 мережах публічного спліту дає 0.5070%")
print("   якщо mini/gam=1.048 ≈ 0.507% ⟹ винна ПОПУЛЯЦІЯ")
print("   якщо full/gam=1.0   ≈ 0.507% ⟹ винні КОНСТАНТИ")
a = math.sqrt(sum(x for x, _ in RES['mini'][1.048])
              / sum(y for _, y in RES['mini'][1.048]))
b = math.sqrt(sum(x for x, _ in RES['full'][1.048])
              / sum(y for _, y in RES['full'][1.048]))
print(f"\nmini vs full при однакових константах: {100*a:.4f}% проти "
      f"{100*b:.4f}% ⟹ ×{a/b:.4f}")
