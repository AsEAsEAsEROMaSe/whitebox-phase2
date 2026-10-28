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


IA = cp.arange(0, NNET // 2)          # TRAIN — перші половина
IB = cp.arange(NNET // 2, NNET)       # TEST  — мережі, яких пошук НЕ БАЧИВ
NAMES = ["kap", "gam", "del", "cB ", "eB "]
GRID = {
    0: [0.9940, 0.9955, 0.9970, 0.9980, 0.9985, 1.0000],
    1: [1.000, 1.020, 1.035, 1.048, 1.060, 1.075, 1.100],
    4: [-0.020, -0.012, -0.008, -0.005, -0.002, 0.000, 0.005],
}


def profile_fit(idx, start, passes=3):
    """повне профілювання по сітці, а не координатний спуск (він застрягає)"""
    p = start.copy()
    for _ in range(passes):
        for k, vals in GRID.items():
            best, bv = 1e9, p[k]
            for v in vals:
                q = p.copy()
                q[k] = v
                e = float(run(q)[idx].mean())
                if e < best:
                    best, bv = e, v
            p[k] = bv
    return p


eb, eg = run(BASE), run(GAMP)
print(f"\nTRAIN {len(IA)} мереж · TEST {len(IB)} мереж (пошук їх НЕ БАЧИВ)")
print(f"{'':>26} {'TRAIN':>10} {'TEST':>10}")
for lab, e in (("база (κ)", eb), ("sub127f (gam=1.04)", eg)):
    print(f"{lab:>26} {100*float(e[IA].mean()):>9.4f}% {100*float(e[IB].mean()):>9.4f}%")

pt = profile_fit(IA, BASE.copy())
et = run(pt)
print(f"\nпідбір ПРОФІЛЮВАННЯМ на TRAIN: "
      f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:+.4f}, {pt[3]:.4f}, {pt[4]:+.4f})")
print(f"  TRAIN {100*float(et[IA].mean()):.4f}%  vs gam ×{float(eg[IA].mean())/float(et[IA].mean()):.4f}")
print(f"  TEST  {100*float(et[IB].mean()):.4f}%  vs gam ×{float(eg[IB].mean())/float(et[IB].mean()):.4f}"
      f"  vs база ×{float(eb[IB].mean())/float(et[IB].mean()):.4f}")
w = int((et[IB] < eg[IB]).sum())
print(f"  знаковий на TEST проти gam: {w}/{len(IB)}")
V, E2 = 1.9227e-06, 2.3385e-05
g = float(eg[IB].mean()) / float(et[IB].mean())
gg = float(eb[IB].mean()) / float(et[IB].mean())
for lab, x in (("понад sub127f", g), ("понад базу", gg)):
    e2 = E2 / x ** 2
    print(f"  у балі {lab}: ×{(V*E2/(V+E2))/(V*e2/(V+e2)):.4f}")
