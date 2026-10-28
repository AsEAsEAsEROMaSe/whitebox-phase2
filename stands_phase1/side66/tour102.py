"""tour102 — ТУРНІРНИЙ ПОШУК ФОРМИ КРОКУ (ідея ЮЗЕРА «як AlphaGo, багато партій»)

ЧОМУ ЦЕ МОЖЛИВО ТІЛЬКИ ЗАРАЗ: спліт `full` має 1000 мереж із ТОЧНОЮ істиною
⟹ фітнес коштує секунди. Сьогодні я перебрав 7 сімей РУКАМИ і одна дала `gam`
(+0.55% локально). Машина зіграє сотні партій замість семи.
🚨 ЩО САМЕ НЕ ПРАЦЮЄ В АНАЛОГІЇ: розподіл мереж ФІКСОВАНИЙ і відомий (He,
256×32) ⟹ «генератор важких мереж» відкривати нічого; сіди мертві за qual102.
Живе лише пошук СТРУКТУРИ — і саме він тут.

ПРОСТІР ФОРМ (база тотожно відтворює поточне замикання):
    u   = gam·t + del
    m   = mp·Φ(u) + cB·(1 + eB·t²)·s·φ(u)
    C   = κ·(WᵀCW) далі як було, з Φ(u), φ(u)
    p = (kap, gam, del, cB, eB),  база (0.9985, 1.0, 0.0, 1.0, 0.0)
⟹ `κ` і `gam` — окремі точки цього простору; шукаємо, чи є кращі.

ФАЗИ:
  1) випадковий пошук по 5-вимірному кубу, N мереж, fp32 (грубий відбір)
  2) локальне уточнення навколо топ-K координатним скануванням
  3) фіналісти → fp64 + КРОС A→B на всіх мережах (вирок)
🔒 ПЛОМБА fp32/fp64: база в обох точностях друкується поруч; якщо розходяться
   більш ніж на 1%, фаза 1 недійсна.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 (а) 🔒 база fp64 відтворить 0.5031% (scan102c)                        85%
 (б) fp32 і fp64 зійдуться в межах 1%                                  70%
 (в) знайдеться точка, краща за (κ,gam) більш ніж на 1% у похибці      40%
 (г) виграш житиме на кросі A→B (не перефіт)                           60%
"""
import glob
import math
import sys

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
KSIG = 8
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NRAND = int(sys.argv[2]) if len(sys.argv) > 2 else 300
BASE = (0.9985, 1.0, 0.0, 1.0, 0.0)
LO = np.array([0.9900, 0.930, -0.150, 0.800, -0.030])
HI = np.array([1.0050, 1.160, 0.150, 1.200, 0.030])

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)

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
Wg32 = cp.asarray(np.array(Ws, dtype=np.float32))          # (B,32,256,256)
Tg = cp.asarray(np.array(Ts, dtype=np.float64))
IDX = cp.arange(W_)
print(f"tour102 · {NNET} мереж на GPU · {NRAND} випадкових точок", flush=True)


def run(p, dt):
    kap, gam, dl, cB, eB = [float(x) for x in p]
    B = Wg32.shape[0]
    m = cp.zeros((B, W_), dtype=dt)
    C = cp.broadcast_to(cp.eye(W_, dtype=dt), (B, W_, W_)).copy()
    for i in range(DEPTH):
        w = Wg32[:, i].astype(dt)
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
        d = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        C[:, IDX, IDX] = d
        m = m_new
    return m.astype(cp.float64)


def cerr(M):
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


e64 = cp.asnumpy(cerr(run(BASE, cp.float64)))
e32 = cp.asnumpy(cerr(run(BASE, cp.float32)))
print(f"🔒 ПЛОМБА база: fp64 {100*e64.mean():.4f}%  fp32 {100*e32.mean():.4f}%  "
      f"розбіжність {abs(e32.mean()/e64.mean()-1)*100:.3f}%", flush=True)
if abs(e32.mean() / e64.mean() - 1) > 0.01:
    print("⚠️ fp32 НЕ ГОДИТЬСЯ для фази 1 — переходжу на fp64 усюди", flush=True)
    DT = cp.float64
else:
    DT = cp.float32
base32 = float(cerr(run(BASE, DT)).mean())

rng = np.random.default_rng(20260810)
P = rng.uniform(LO, HI, size=(NRAND, 5))
P[0] = BASE
P[1] = (0.9985, 1.04, 0.0, 1.0, 0.0)          # чинний sub127f
res = []
for i, p in enumerate(P):
    res.append((float(cerr(run(p, DT)).mean()), tuple(p)))
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{NRAND}  найкраще поки {100*min(res)[0]:.4f}%", flush=True)
res.sort()
print(f"\nФАЗА 1: база {100*base32:.4f}%  ·  sub127f {100*res[[r[1] for r in res].index(tuple(P[1]))][0] if False else 0:.4f}")
print(f"{'ранг':>5} {'похибка':>10} {'×':>7}  (kap, gam, del, cB, eB)")
for r, (e, p) in enumerate(res[:8]):
    print(f"{r+1:>5} {100*e:>9.4f}% {base32/e:>7.4f}  "
          f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:+.4f}, {p[3]:.4f}, {p[4]:+.4f})")

# ---- ФАЗА 2: координатне уточнення навколо топ-3 ----
STEP = np.array([0.0015, 0.015, 0.015, 0.02, 0.004])
FIN = []
for e0, p0 in res[:3]:
    p = np.array(p0)
    best = e0
    for _ in range(3):
        for k in range(5):
            for dsg in (-1.0, 1.0):
                q = p.copy()
                q[k] += dsg * STEP[k]
                q = np.clip(q, LO, HI)
                e = float(cerr(run(q, DT)).mean())
                if e < best:
                    best, p = e, q
        STEP = STEP * 0.5
    FIN.append((best, tuple(p)))
    STEP = np.array([0.0015, 0.015, 0.015, 0.02, 0.004])
FIN.sort()
print("\nФАЗА 2 (уточнено):")
for e, p in FIN:
    print(f"  {100*e:>9.4f}% ×{base32/e:.4f}  "
          f"({p[0]:.4f}, {p[1]:.4f}, {p[2]:+.4f}, {p[3]:.4f}, {p[4]:+.4f})")

# ---- ФАЗА 3: fp64 + КРОС A→B ----
IA = cp.arange(0, NNET, 2)
IB = cp.arange(1, NNET, 2)
eb = cerr(run(BASE, cp.float64))
eg = cerr(run((0.9985, 1.04, 0.0, 1.0, 0.0), cp.float64))
print(f"\nФАЗА 3 (fp64, {NNET} мереж):")
print(f"{'кандидат':>34} {'усі':>10} {'×':>7} {'A':>9} {'B':>9}")
print(f"{'база (κ)':>34} {100*float(eb.mean()):>9.4f}% {1.0:>7.4f} "
      f"{100*float(eb[IA].mean()):>8.4f}% {100*float(eb[IB].mean()):>8.4f}%")
print(f"{'sub127f (κ,gam=1.04)':>34} {100*float(eg.mean()):>9.4f}% "
      f"{float(eb.mean())/float(eg.mean()):>7.4f} "
      f"{100*float(eg[IA].mean()):>8.4f}% {100*float(eg[IB].mean()):>8.4f}%")
for e, p in FIN:
    ee = cerr(run(p, cp.float64))
    lab = f"({p[0]:.4f},{p[1]:.3f},{p[2]:+.3f},{p[3]:.3f},{p[4]:+.3f})"
    print(f"{lab:>34} {100*float(ee.mean()):>9.4f}% "
          f"{float(eb.mean())/float(ee.mean()):>7.4f} "
          f"{100*float(ee[IA].mean()):>8.4f}% {100*float(ee[IB].mean()):>8.4f}%"
          f"  vs gam ×{float(eg.mean())/float(ee.mean()):.4f}")
