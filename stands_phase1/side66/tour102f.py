"""tour102f — ШУКАЄМО САМУ ФУНКЦІЮ `h(t)`, А НЕ ЇЇ ДВА НАБЛИЖЕННЯ

СПОСТЕРЕЖЕННЯ, ЩО ПРИВЕЛО СЮДИ: `gam` і `eB` діють в один бік, але з різною
формою по `t` — обидва множать член `s·φ` на спадну функцію:
    gam=1.048 ⟹ φ(gt)/φ(t) = exp(−t²(g²−1)/2)   → f(4) = 0.456
    eB=−0.002 ⟹ (1 + eB·t²)                      → f(4) = 0.968
⟹ двома параметрами ми намацуємо ОДНУ невідому функцію.

🔑 АЛЕ ПРАВИЛЬНА ПАРАМЕТРИЗАЦІЯ — НЕ `f(t)·s·φ(t)`, А ЗСУВ АРГУМЕНТУ:
   при t=4 зміна Φ дає внесок 7e-5·s, зміна φ дає 7.3e-5·s — ОДНОГО ПОРЯДКУ
   ⟹ Φ і φ мусять лишатися УЗГОДЖЕНИМИ, інакше ламається сам сенс кроку.
   Тому шукаємо монотонну `u = h(t)`, а далі все як було:
       m = mp·Φ(u) + s·φ(u),  C, d — теж через Φ(u), φ(u)
   `h(t) = t` — база · `h(t) = gam·t` — sub127f · таблиця — цей стенд.

ПАРАМЕТРИ: kap + зсуви δ_k у вузлах t_k (лінійна інтерполяція, сталі краї).
Тобто `u = t + interp(t; t_k, δ_k)`.

ЩО МІРЯЄМО (жодного «мертво/живо»):
  1) TRAIN/TEST hold-out на мережах, яких пошук не бачив
  2) саму знайдену форму δ(t) — вона цікава сама по собі: вона каже, ДЕ
     замикання систематично помиляється по режиму воріт
  3) порівняння з `gam`-наближенням: скільки таблиця додає понад нього
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_, KSIG = 32, 256, 8
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/"
       "snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 400
NODES = cp.asarray([-3.0, -1.0, 0.0, 1.5, 3.0, 4.5], dtype=cp.float32)
NN = int(NODES.shape[0])

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
        Ws.append(Wa[r])
        Ts.append(Ta[r])
    del Wa, Ta
NNET = len(Ws)
Wg = cp.asarray(np.array(Ws, dtype=np.float32))
Tg = cp.asarray(np.array(Ts, dtype=np.float64))
IDX = cp.arange(W_)
IA = cp.arange(0, NNET // 2)
IB = cp.arange(NNET // 2, NNET)
print(f"tour102f · {NNET} мереж · TRAIN {len(IA)} / TEST {len(IB)} · "
      f"вузли {[float(x) for x in cp.asnumpy(NODES)]}", flush=True)


def run(kap, dlt):
    """dlt — зсуви у вузлах (numpy, довжина NN); u = t + interp(t)"""
    d = cp.asarray(np.asarray(dlt, dtype=np.float32))
    B = Wg.shape[0]
    m = cp.zeros((B, W_), dtype=cp.float32)
    C = cp.broadcast_to(cp.eye(W_, dtype=cp.float32), (B, W_, W_)).copy()
    for i in range(DEPTH):
        w = Wg[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w)
        deep = i > KSIG
        if deep:
            cpm = cpm * float(kap)
        s2 = cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w)
        t = mp / s
        u = t + cp.interp(t, NODES, d) if deep else t
        Ph = ndtr(u)
        ph = cp.exp(-0.5 * u * u) / SQ2PI
        m_new = mp * Ph + s * ph
        pv = ph / s
        C = (cpm * (Ph[:, :, None] * Ph[:, None, :])
             + (0.5 * (cpm * cpm)) * (pv[:, :, None] * pv[:, None, :]))
        C[:, IDX, IDX] = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        m = m_new
    M = m.astype(cp.float64)
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


nodes_np = cp.asnumpy(NODES)
zero = np.zeros(NN, dtype=np.float32)
gam_eq = (1.048 - 1.0) * nodes_np          # h(t)=1.048·t у вигляді зсувів
eb = run(0.9985, zero)
eg = run(0.9985, gam_eq)
print(f"\n{'':>28} {'TRAIN':>10} {'TEST':>10}")
print(f"{'база (κ, h=t)':>28} {100*float(eb[IA].mean()):>9.4f}% {100*float(eb[IB].mean()):>9.4f}%")
print(f"{'gam=1.048 як таблиця':>28} {100*float(eg[IA].mean()):>9.4f}% {100*float(eg[IB].mean()):>9.4f}%")
print("🔒 другий рядок мусить збігтися з прямим gam-прогоном (та сама функція)",
      flush=True)

KAPS = [0.9955, 0.9970, 0.9980, 0.9985, 1.0000]
STEPS = [0.24, 0.12, 0.06, 0.03]
kap, dlt = 0.9985, gam_eq.copy()
best = float(run(kap, dlt)[IA].mean())
for st in STEPS:
    for _ in range(2):
        for k in range(NN):
            for sg in (-1.0, 1.0):
                q = dlt.copy()
                q[k] += sg * st
                e = float(run(kap, q)[IA].mean())
                if e < best:
                    best, dlt = e, q
        for kv in KAPS:
            e = float(run(kv, dlt)[IA].mean())
            if e < best:
                best, kap = e, kv
    print(f"  крок {st}: TRAIN {100*best:.4f}%  kap={kap:.4f}  "
          f"δ=[{', '.join(f'{x:+.3f}' for x in dlt)}]", flush=True)

et = run(kap, dlt)
print(f"\nЗНАЙДЕНО: kap={kap:.4f}  δ=[{', '.join(f'{x:+.4f}' for x in dlt)}]")
print(f"  h(t) у вузлах: " + "  ".join(
    f"t={n:+.1f}→u={n+d:+.3f}" for n, d in zip(nodes_np, dlt)))
print(f"  TRAIN {100*float(et[IA].mean()):.4f}%  vs gam ×{float(eg[IA].mean())/float(et[IA].mean()):.4f}")
print(f"  TEST  {100*float(et[IB].mean()):.4f}%  vs gam ×{float(eg[IB].mean())/float(et[IB].mean()):.4f}"
      f"  vs база ×{float(eb[IB].mean())/float(et[IB].mean()):.4f}")
print(f"  знаковий на TEST проти gam: {int((et[IB] < eg[IB]).sum())}/{len(IB)}")
V, E2 = 1.9227e-06, 2.3385e-05
for lab, x in (("понад gam", float(eg[IB].mean()) / float(et[IB].mean())),
               ("понад базу", float(eb[IB].mean()) / float(et[IB].mean()))):
    e2 = E2 / x ** 2
    print(f"  у балі {lab}: ×{(V*E2/(V+E2))/(V*e2/(V+e2)):.4f}")
