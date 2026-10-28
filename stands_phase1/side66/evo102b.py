"""evo102 — РОЗШИРЕНИЙ ПРОСТІР: не 5 констант, а ЧОТИРИ МІСЦЯ РЕКУРСІЇ

ЗАУВАЖЕННЯ ЮЗЕРА: механізм AlphaGo потужніший за те, що я зробив — і це так.
tour102 сканував 5 констант, які Я САМ обрав; простір не розширювався. Тут
параметризуються ЧОТИРИ РІЗНІ МІСЦЯ кроку, з них два не мали жодного параметра
за 102 сесії:

    u    = gam·t                      аргумент воріт  (форма ВИЧЕРПАНА: лінійна)
    m'   = mp·Φ(u) + W(t)·s·φ(u)      W(t) = 1 + w1·t² + w2·t⁴
    C'ij = R(ρ)·[cp·ΦΦ + 0.5cp²·(φ/s)(φ/s)]   R(ρ) = 1 + r1·ρ + r2·ρ²   ← НОВЕ
    d'   = D(t)·d                     D(t) = 1 + d1·t²                   ← НОВЕ
  плюс kap (множник на cp при i>KSIG)

🔑 ЧОМУ `R(ρ)` — ГОЛОВНЕ: `κ` множить УСЮ `C` на СТАЛУ. Але похибка кроку в
позадіагоналі залежить від кореляції пари: при ρ→0 пара майже незалежна і
гаусова гіпотеза точна, при ρ→1 — це фактично один нейрон і Мелер K=2 обрізає
найбільше. Тобто поправка МУСИТЬ бути функцією ρ, а не числом. `orac101` казав,
що істинна позадіагональ дає ×1.149 — це стеля для цього місця.
🔒 ГЕЙТ УЗГОДЖЕНОСТІ C: `R` чіпає позадіагональ, `D` — діагональ, і вони
скануються РАЗОМ, тож узгодженість перевіряється прямо, а не постулюється.

БАЗА (усі поправки вимкнені) = поточне замикання з κ ⟹ вбудована пломба.
ВИРОК — ЛИШЕ hold-out на мережах, яких пошук не бачив.
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 300
# p = [kap, gam, w1, w2, r1, r2, d1]
BASE = np.array([0.9985, 1.000, 0.0, 0.0, 0.0, 0.0, 0.0])
GAMP = np.array([0.9985, 1.048, 0.0, 0.0, 0.0, 0.0, 0.0])
NAMES = ["kap", "gam", "w1 ", "w2 ", "r1 ", "r2 ", "d1 "]
GRID = [
    [0.9955, 0.9970, 0.9980, 0.9985, 1.0000],
    [1.000, 1.020, 1.035, 1.048, 1.060, 1.080],
    [-0.008, -0.006, -0.004, -0.002, 0.0, 0.002],
    [-4e-4, -2e-4, -1e-4, 0.0, 1e-4, 2e-4],
    [-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004, 0.008],
    [-0.004, -0.002, -0.001, 0.0, 0.001, 0.002, 0.004, 0.008],
    [-0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002],
]

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
print(f"evo102b (дрібна сітка + захист) · {NNET} мереж · TRAIN {len(IA)} / TEST {len(IB)}", flush=True)


def run(p):
    kap, gam, w1, w2, r1, r2, d1 = [float(x) for x in p]
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
        t2 = t * t
        u = t * gam if deep else t
        Ph = ndtr(u)
        ph = cp.exp(-0.5 * u * u) / SQ2PI
        Wt = (1.0 + w1 * t2 + w2 * (t2 * t2)) if deep else 1.0
        m_new = mp * Ph + Wt * (s * ph)
        pv = ph / s
        Cn = (cpm * (Ph[:, :, None] * Ph[:, None, :])
              + (0.5 * (cpm * cpm)) * (pv[:, :, None] * pv[:, None, :]))
        if deep and (r1 != 0.0 or r2 != 0.0):
            rho = cpm / (s[:, :, None] * s[:, None, :])
            fac = 1.0 + r1 * rho + r2 * (rho * rho)
            Cn = Cn * cp.maximum(fac, 0.05)   # захист від виродження C
        d = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        if deep and d1 != 0.0:
            d = d * cp.maximum(1.0 + d1 * t2, 0.05)
        Cn[:, IDX, IDX] = d
        C = Cn
        m = m_new
    M = m.astype(cp.float64)
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    e = cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))
    return cp.nan_to_num(e, nan=9.9, posinf=9.9, neginf=9.9)


eb, eg = run(BASE), run(GAMP)
print(f"\n{'':>24} {'TRAIN':>10} {'TEST':>10}")
print(f"{'база (κ)':>24} {100*float(eb[IA].mean()):>9.4f}% {100*float(eb[IB].mean()):>9.4f}%")
print(f"{'gam=1.048':>24} {100*float(eg[IA].mean()):>9.4f}% {100*float(eg[IB].mean()):>9.4f}%")

print("\n--- ПРОФІЛІ КОЖНОГО МІСЦЯ ОКРЕМО (від gam-точки, TRAIN) ---", flush=True)
for k in range(len(BASE)):
    row = []
    for v in GRID[k]:
        q = GAMP.copy()
        q[k] = v
        row.append((v, float(run(q)[IA].mean())))
    bv = min(row, key=lambda x: x[1])
    print(f"{NAMES[k]}: " + "  ".join(f"{v:+.4g}→{100*e:.4f}%" for v, e in row)
          + f"   мін {bv[0]:+.4g}", flush=True)

p = GAMP.copy()
best = float(run(p)[IA].mean())
for it in range(3):
    for k in range(len(BASE)):
        bv, be = p[k], best
        for v in GRID[k]:
            q = p.copy()
            q[k] = v
            e = float(run(q)[IA].mean())
            if e < be:
                be, bv = e, v
        p[k], best = bv, be
    print(f"  прохід {it+1}: TRAIN {100*best:.4f}%  "
          f"[{', '.join(f'{x:.4g}' for x in p)}]", flush=True)

et = run(p)
print(f"\nЗНАЙДЕНО [{', '.join(f'{x:.5g}' for x in p)}]")
print(f"  TRAIN {100*float(et[IA].mean()):.4f}%  vs gam ×{float(eg[IA].mean())/float(et[IA].mean()):.4f}")
print(f"  TEST  {100*float(et[IB].mean()):.4f}%  vs gam ×{float(eg[IB].mean())/float(et[IB].mean()):.4f}"
      f"  vs база ×{float(eb[IB].mean())/float(et[IB].mean()):.4f}")
print(f"  знаковий на TEST проти gam: {int((et[IB] < eg[IB]).sum())}/{len(IB)}")
V, E2 = 1.9227e-06, 2.3385e-05
for lab, x in (("понад gam", float(eg[IB].mean()) / float(et[IB].mean())),
               ("понад базу", float(eb[IB].mean()) / float(et[IB].mean()))):
    e2 = E2 / x ** 2
    print(f"  у балі {lab}: ×{(V*E2/(V+E2))/(V*e2/(V+e2)):.4f}")
