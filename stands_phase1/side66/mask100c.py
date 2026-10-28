"""mask100c — ТЕ САМЕ, ЩО mask100b, АЛЕ НА ВИСОКІЙ ТОЧНОСТІ (вимога ЮЗЕРА).

`[ЗМІРЯНО mask100]` сигнал маскової розбіжності 3.5% при шумі 1.1% ⟹ ×3.2.
Для КОРЕЛЯЦІЙНОГО тесту цього мало. Тут `N` піднято вчетверо, а пам'ять тримається
сталою: усе рахується ЧАНКАМИ через накопичення сум, без зберігання активацій.

    m  = Σp/N ·  s² = Σp²/N − m²  ·  ρ = (Σp₁p₂/N − m₁m₂)/(s₁s₂)
    real = Σ[D₁∧D₂]/N

⟹ шум `P̂` = √(p(1−p)/N) падає як 1/√N; при N=262144 це ≈0.39% проти сигналу 3.5%,
тобто **×9** замість ×3.2.

🔒 КОНТРОЛІ ті самі: (1) перемішана ознака · (2) прості ознаки тієї ж природи ·
   (3) часткова кореляція · (4) 🆕 ПЛОМБА ШУМУ: половини вибірки мусять дати
   ту саму розбіжність (справжній сигнал стоїть, шум падає як 1/√N).

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ, ті самі, що в mask100b):
 (а) |corr(маскова розбіжність, E_i²)| > 0.30                        40%
 (б) сигнал ВИЖИВЕ після контролю на прості ознаки                   30%
 (в) прості ознаки теж корелюють із E_i²                             60%
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
NPAIRS_TOT = int(sys.argv[2]) if len(sys.argv) > 2 else 131072   # пар
CHUNK = 8192
NPAIR = 3000
CUTS = [(1, 2), (8, 9), (16, 17), (24, 25), (8, 24)]
LAYERS = sorted({l for c in CUTS for l in c} | {16})


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
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in ctrl])
    def resid(v):
        v = np.asarray(v, float)
        beta, *_ = np.linalg.lstsq(A, v, rcond=None)
        return v - A @ beta
    return corr(resid(y), resid(x))


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)
N = 2 * NPAIRS_TOT

print(f"mask100c · {NNET} мереж × {N} семплів (чанками по {2*CHUNK}) · "
      f"очікуваний шум P̂ ≈ {100*0.5/math.sqrt(N)/0.25:.3f}%", flush=True)
rng0 = np.random.default_rng(20260809)
II = {c: rng0.integers(0, W_, NPAIR) for c in CUTS}
JJ = {c: rng0.integers(0, W_, NPAIR) for c in CUTS}

MD, MT, ML, MR, MC, MH = [], [], [], [], [], []
for net in range(NNET):
    W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
         for L in range(DEPTH)]
    rng = np.random.default_rng(555 + net)
    # накопичувачі
    S1 = {L: np.zeros(W_) for L in LAYERS}
    S2 = {L: np.zeros(W_) for L in LAYERS}
    CP = {c: np.zeros(NPAIR) for c in CUTS}
    JOINT = {c: np.zeros(NPAIR) for c in CUTS}
    JHALF = {c: [np.zeros(NPAIR), np.zeros(NPAIR)] for c in CUTS}
    alive_acc = np.zeros(DEPTH - 1)
    done = 0
    half_n = [0, 0]
    while done < NPAIRS_TOT:
        nb = min(CHUNK, NPAIRS_TOT - done)
        z = rng.standard_normal((nb, W_))
        A = np.concatenate([z, -z], axis=0)
        keep = {}
        a = A
        for i in range(DEPTH):
            p = a @ W[i]
            if i in LAYERS:
                keep[i] = p
                S1[i] += p.sum(0)
                S2[i] += (p * p).sum(0)
            if i < DEPTH - 1:
                a = np.maximum(p, 0.0)
                alive_acc[i] += (p.max(0) >= 0).sum()
        h = 0 if done * 2 < NPAIRS_TOT else 1
        half_n[h] += 2 * nb
        for c in CUTS:
            l1, l2 = c
            ii, jj = II[c], JJ[c]
            CP[c] += (keep[l1][:, ii] * keep[l2][:, jj]).sum(0)
            j = ((keep[l1][:, ii] > 0) & (keep[l2][:, jj] > 0)).sum(0)
            JOINT[c] += j
            JHALF[c][h] += j
        done += nb
    devs, rhos, dev_half = [], [], [[], []]
    for c in CUTS:
        l1, l2 = c
        ii, jj = II[c], JJ[c]
        m1, m2 = S1[l1] / N, S1[l2] / N
        s1 = np.sqrt(np.maximum(S2[l1] / N - m1 * m1, 1e-30))
        s2 = np.sqrt(np.maximum(S2[l2] / N - m2 * m2, 1e-30))
        rho = np.clip((CP[c] / N - m1[ii] * m2[jj])
                      / np.maximum(s1[ii] * s2[jj], 1e-30), -0.999, 0.999)
        pred = bivar_orthant(m1[ii] / s1[ii], m2[jj] / s2[jj], rho)
        real = JOINT[c] / N
        devs.append(np.abs(real - pred).mean() / max(np.abs(real).mean(), 1e-30))
        rhos.append(float(np.abs(rho).mean()))
        for h in (0, 1):
            rh = JHALF[c][h] / max(half_n[h], 1)
            dev_half[h].append(np.abs(rh - pred).mean()
                               / max(np.abs(rh).mean(), 1e-30))
    MD.append(float(np.mean(devs)))
    MH.append((float(np.mean(dev_half[0])), float(np.mean(dev_half[1]))))
    MR.append(float(np.mean(rhos)))
    m16 = S1[16] / N
    s16 = np.sqrt(np.maximum(S2[16] / N - m16 * m16, 1e-30))
    MT.append(float(np.median(np.abs(m16 / s16))))
    ML.append(float(alive_acc.mean() / W_ / (NPAIRS_TOT / CHUNK)))
    MC.append(closure(W))
    print(f"  {net+1}/{NNET}  розбіжність {100*MD[-1]:.3f}%  "
          f"(половини {100*MH[-1][0]:.3f}% / {100*MH[-1][1]:.3f}%)", flush=True)

MC = np.array(MC)
T = np.array([FM[i].astype(np.float64) for i in range(NNET)])
a = float((MC * T).sum() / (MC * MC).sum())
E2 = (((a * MC) - T) ** 2).mean(1)
MD, MT, ML, MR = map(np.array, (MD, MT, ML, MR))
MH = np.array(MH)

print("\n" + "=" * 74)
print(f"глобальна α = {a:.6f} · середнє E_i² = {E2.mean():.6e}")
print(f"🔒 ПЛОМБА ШУМУ: половини вибірки  {100*MH[:,0].mean():.3f}% vs "
      f"{100*MH[:,1].mean():.3f}%  (сигнал мусить СТОЯТИ)")
print(f"   сама розбіжність (повна вибірка): {100*MD.mean():.3f}% "
      f"· очікуваний шум {100*0.5/math.sqrt(N)/0.25:.3f}%")
print(f"\n{'ознака':<34s}{'corr з E_i²':>14s}")
print("=" * 74)
for nm, v in (("МАСКОВА РОЗБІЖНІСТЬ", MD), ("медіана |t| @16", MT),
              ("частка живих", ML), ("сер |ρ| на зрізах", MR)):
    print(f"{nm:<34s}{corr(v, E2):>14.4f}")

rs = np.random.default_rng(7)
sh = [abs(corr(MD[rs.permutation(NNET)], E2)) for _ in range(400)]
print(f"\n🔒 КОНТРОЛЬ (1) перемішана ознака: |corr| сер {np.mean(sh):.4f} · "
      f"95%-перцентиль {np.percentile(sh, 95):.4f}  проти реального "
      f"{abs(corr(MD, E2)):.4f}")
print(f"🔒 КОНТРОЛЬ (3) ЧАСТКОВА кореляція (знято мед|t|, живі, |ρ|): "
      f"{partial(E2, MD, [MT, ML, MR]):.4f}   сира {corr(MD, E2):.4f}")
