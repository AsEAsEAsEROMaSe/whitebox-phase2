"""mask100b — ЧИ Є МАСКОВА РОЗБІЖНІСТЬ ТИМ САМИМ ОБ'ЄКТОМ, ЩО ПОХИБКА ЗАМИКАННЯ?

`[ЗМІРЯНО mask100, 4 мережі]` реальна `P(обидва відкриті)` розходиться з гаусовим
передбаченням із ТИХ САМИХ `(m,C)` у **6.02×** над шумом на парі шарів (1,2), і
розбіжність НЕ спадає з відстанню між шарами ((8,12)=(8,16)=(8,24)≈3.5%).
🔒 пломба пройдена: шар 0 сам із собою (де `p` гаусів ТОЧНО) дав ×0.43, тобто шум.

ПИТАННЯ ЦЬОГО ПРОГОНУ: маскова розбіжність — це ТОЙ САМИЙ об'єкт, що похибка
замикання, чи просто красива діагностика?

📏 ЗАКОН С99, ЯКОГО ТУТ ТРЕБА ДОТРИМАТИСЬ: ознаку міряти проти `E_i²` ПІСЛЯ `α`,
   НЕ проти істини (сира похибка й `E_i²` після α ОРТОГОНАЛЬНІ, corr = −0.018).

🔒 КОНТРОЛІ (без них вимір нічого не вартий):
 (1) ПЕРЕМІШАНА ознака між мережами ⟹ corr мусить впасти в нуль
 (2) ПРОСТІ ознаки тієї ж природи (медіана |t|, частка живих, сер|ρ|) — якщо
     вони корелюють так само, то маскова розбіжність НЕ несе нічого свого
 (3) ЧАСТКОВА кореляція: чи виживає сигнал маски після зняття простих ознак

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) |corr(маскова розбіжність, E_i²)| > 0.30                        40%
 (б) сигнал ВИЖИВЕ після контролю на прості ознаки                   30%
 (в) прості ознаки теж корелюють із E_i² (тобто контроль потрібен)   60%
 (г) 🔒 перемішаний контроль дасть |corr| < 0.15                     90%
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
NPAIR = 3000
CUTS = [(1, 2), (8, 9), (16, 17), (24, 25), (8, 24)]


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def bivar_orthant(t1, t2, rho, nq=48):
    xs, ws = np.polynomial.legendre.leggauss(nq)
    lo, hi = -8.0, 8.0
    u = 0.5 * (hi - lo) * xs + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * ws
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    out = np.zeros_like(t1)
    for k in range(nq):
        uk = u[k]
        out += (w[k] * (math.exp(-0.5 * uk * uk) / SQ2PI)
                * ndtr((t2 + rho * uk) / q) * (uk > -t1))
    return out


def closure(W):
    m = np.zeros(W_)
    C = np.eye(W_)
    for i in range(DEPTH):
        w = W[i]
        cp = w.T @ C @ w
        if i > 8:
            cp = cp * 0.9985
        s = np.sqrt(np.maximum(np.diagonal(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        if i == DEPTH - 1:
            break
        C = cp * np.outer(Ph, Ph) + (0.5 * (cp * cp)) * np.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diagonal(C))
    return m


def corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 1e-300 else 0.0


def partial(y, x, ctrl):
    """кореляція y з x після зняття лінійного впливу ctrl (список ознак)"""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in ctrl])
    def resid(v):
        v = np.asarray(v, float)
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    return corr(resid(y), resid(x))


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
x = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(x.type):
    x = x.flatten()
Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, W_, W_)
y = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(y.type):
    y = y.flatten()
FM = np.asarray(y).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)

print(f"mask100b · {NNET} мереж × {2*NS} семплів · ціль = E_i² ПІСЛЯ α",
      flush=True)
rng0 = np.random.default_rng(20260809)
MD, MT, ML, MR, MC = [], [], [], [], []
for net in range(NNET):
    W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
         for L in range(DEPTH)]
    rng = np.random.default_rng(555 + net)
    z = rng.standard_normal((NS, W_))
    A = np.concatenate([z, -z], axis=0)
    N = A.shape[0]
    P = []
    for i in range(DEPTH):
        p = A @ W[i]
        P.append(p)
        if i < DEPTH - 1:
            A = np.maximum(p, 0.0)
    D = [(p > 0.0) for p in P]

    devs, rhos = [], []
    for (l1, l2) in CUTS:
        p1, p2 = P[l1], P[l2]
        m1, m2 = p1.mean(0), p2.mean(0)
        s1, s2 = p1.std(0), p2.std(0)
        t1 = m1 / np.maximum(s1, 1e-30)
        t2 = m2 / np.maximum(s2, 1e-30)
        ii = rng0.integers(0, W_, NPAIR)
        jj = rng0.integers(0, W_, NPAIR)
        rho = np.clip(((p1[:, ii] - m1[ii]) * (p2[:, jj] - m2[jj])).mean(0)
                      / np.maximum(s1[ii] * s2[jj], 1e-30), -0.999, 0.999)
        real = (D[l1][:, ii] & D[l2][:, jj]).mean(0)
        pred = bivar_orthant(t1[ii], t2[jj], rho)
        devs.append(np.abs(real - pred).mean() / max(np.abs(real).mean(), 1e-30))
        rhos.append(float(np.abs(rho).mean()))

    MD.append(float(np.mean(devs)))                    # маскова розбіжність
    MR.append(float(np.mean(rhos)))                    # сер |ρ|
    p16 = P[16]
    MT.append(float(np.median(np.abs(p16.mean(0)
                                     / np.maximum(p16.std(0), 1e-30)))))
    ML.append(float(np.mean([(d.max(0)).mean() for d in D[:-1]])))  # живі
    MC.append(closure(W))
    print(f"  {net+1}/{NNET}", flush=True)

MC = np.array(MC)
T = np.array([FM[i].astype(np.float64) for i in range(NNET)])
a = float((MC * T).sum() / (MC * MC).sum())
E2 = (((a * MC) - T) ** 2).mean(1)                     # 🔑 ціль: E_i² ПІСЛЯ α
MD, MT, ML, MR = map(np.array, (MD, MT, ML, MR))

print("\n" + "=" * 74)
print(f"глобальна α = {a:.6f} · середнє E_i² = {E2.mean():.6e}")
print(f"{'ознака':<34s}{'corr з E_i²':>14s}")
print("=" * 74)
for nm, v in (("МАСКОВА РОЗБІЖНІСТЬ", MD), ("медіана |t| @16", MT),
              ("частка живих", ML), ("сер |ρ| на зрізах", MR)):
    print(f"{nm:<34s}{corr(v, E2):>14.4f}")

print(f"\n🔒 КОНТРОЛЬ (1) перемішана маскова ознака (200 перестановок):")
rs = np.random.default_rng(7)
sh = [abs(corr(MD[rs.permutation(NNET)], E2)) for _ in range(200)]
print(f"    середнє |corr| = {np.mean(sh):.4f} · 95-й перцентиль "
      f"{np.percentile(sh, 95):.4f}  проти реального "
      f"{abs(corr(MD, E2)):.4f}")

print(f"\n🔒 КОНТРОЛЬ (3) ЧАСТКОВА кореляція маски після зняття простих ознак:")
print(f"    partial(E_i², маска | мед|t|, живі, |ρ|) = "
      f"{partial(E2, MD, [MT, ML, MR]):.4f}")
print(f"    для порівняння — сира                     = {corr(MD, E2):.4f}")
