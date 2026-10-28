"""tour102x — ЧЕСНИЙ КРОС + КАРТА ОКОЛУ знахідки турніру.

tour102 підбирав параметри на ВСІХ мережах і там же міряв ⟹ in-sample.
Тут два питання, і жодне з них не «живе чи мертве»:
  (1) скільки з ×1.0154 переживає підбір на A → вимір на B (і навпаки)?
  (2) ЯКА ФОРМА в околу: де межа зони виграшу по кожній координаті?
      друкуємо ПОВНІ профілі, а не лише оптимум — нуль теж є даними.

ПРОСТІР: u = gam·t + del,  m = mp·Φ(u) + cB·(1+eB·t²)·s·φ(u),  C: κ при i>KSIG
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 200
BASE = np.array([0.9985, 1.0, 0.0, 1.0, 0.0])
GAMP = np.array([0.9985, 1.04, 0.0, 1.0, 0.0])
CAND = np.array([0.9974, 1.0475, 0.0, 1.0, -0.0050])
LO = np.array([0.9900, 0.930, -0.150, 0.800, -0.030])
HI = np.array([1.0050, 1.160, 0.150, 1.200, 0.030])
NAMES = ["kap", "gam", "del", "cB ", "eB "]

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
print(f"tour102x · {NNET} мереж · чесний крос + карта околу", flush=True)


def run(p):
    kap, gam, dl, cB, eB = [float(x) for x in p]
    B = Wg.shape[0]
    m = cp.zeros((B, W_), dtype=cp.float32)
    C = cp.broadcast_to(cp.eye(W_, dtype=cp.float32), (B, W_, W_)).copy()
    for i in range(DEPTH):
        w = Wg[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w)
        deep = i > KSIG
        if deep:
            cpm = cpm * kap
        s2 = cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w)
        t = mp / s
        u = (gam * t + dl) if deep else t
        Ph = ndtr(u)
        ph = cp.exp(-0.5 * u * u) / SQ2PI
        w2 = (cB * (1.0 + eB * t * t)) if deep else 1.0
        m_new = mp * Ph + w2 * (s * ph)
        pv = ph / s
        C = (cpm * (Ph[:, :, None] * Ph[:, None, :])
             + (0.5 * (cpm * cpm)) * (pv[:, :, None] * pv[:, None, :]))
        C[:, IDX, IDX] = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        m = m_new
    M = m.astype(cp.float64)
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


IA, IB = cp.arange(0, NNET, 2), cp.arange(1, NNET, 2)


def fit(idx, start, rounds=3):
    p = start.copy()
    best = float(run(p)[idx].mean())
    step = np.array([0.0015, 0.015, 0.015, 0.02, 0.004])
    for _ in range(rounds):
        for k in range(5):
            for d in (-1.0, 1.0):
                q = np.clip(p + d * step * np.eye(5)[k], LO, HI)
                e = float(run(q)[idx].mean())
                if e < best:
                    best, p = e, q
        step = step * 0.5
    return p


eb, eg, ec = run(BASE), run(GAMP), run(CAND)
print(f"\n{'':>26} {'усі':>10} {'A':>10} {'B':>10}")
for lab, e in (("база (κ)", eb), ("sub127f (gam=1.04)", eg), ("кандидат турніру", ec)):
    print(f"{lab:>26} {100*float(e.mean()):>9.4f}% {100*float(e[IA].mean()):>9.4f}% "
          f"{100*float(e[IB].mean()):>9.4f}%")

pA, pB = fit(IA, BASE), fit(IB, BASE)
eA, eB_ = run(pA), run(pB)
print("\n--- ЧЕСНИЙ КРОС (підбір на одній половині, вимір на іншій) ---")
print(f"підбір на A ({', '.join(f'{x:.4f}' for x in pA)})")
print(f"   ⟹ на B: {100*float(eA[IB].mean()):.4f}%  vs gam "
      f"×{float(eg[IB].mean())/float(eA[IB].mean()):.4f}  vs база "
      f"×{float(eb[IB].mean())/float(eA[IB].mean()):.4f}")
print(f"підбір на B ({', '.join(f'{x:.4f}' for x in pB)})")
print(f"   ⟹ на A: {100*float(eB_[IA].mean()):.4f}%  vs gam "
      f"×{float(eg[IA].mean())/float(eB_[IA].mean()):.4f}  vs база "
      f"×{float(eb[IA].mean())/float(eB_[IA].mean()):.4f}")
pm = 0.5 * (pA + pB)
em = run(pm)
print(f"\nсереднє двох підборів ({', '.join(f'{x:.4f}' for x in pm)})")
print(f"   усі: {100*float(em.mean()):.4f}%  vs gam ×{float(eg.mean())/float(em.mean()):.4f}"
      f"  знаковий проти gam {int((em < eg).sum())}/{NNET}")

print("\n--- КАРТА ОКОЛУ кандидата (профіль по кожній координаті) ---")
GRID = {
    0: [0.9940, 0.9955, 0.9970, 0.9985, 1.0000],
    1: [1.000, 1.020, 1.035, 1.048, 1.060, 1.080, 1.110],
    2: [-0.060, -0.030, 0.000, 0.030, 0.060],
    3: [0.94, 0.97, 1.00, 1.03, 1.06],
    4: [-0.020, -0.012, -0.005, 0.000, 0.005, 0.012],
}
for k, vals in GRID.items():
    row = []
    for v in vals:
        q = CAND.copy()
        q[k] = v
        row.append((v, float(run(q).mean())))
    bestv = min(row, key=lambda x: x[1])
    print(f"{NAMES[k]}: " + "  ".join(f"{v:+.4f}→{100*e:.4f}%" for v, e in row)
          + f"   мін при {bestv[0]:+.4f}")
