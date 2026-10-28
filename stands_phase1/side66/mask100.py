"""mask100 — ЧИ НЕСУТЬ МАСКИ ReLU ЩОСЬ ПОНАД `(m,C)`? (ідея ЮЗЕРА, С100)

ЮЗЕР: «нелінійність на мінусових вагах фактично розділяє мережу на дві частини…
потрібно зрозуміти вплив мінусів на плюсові ваги… не просто послідовно, а як
попередні мінуси впливають на наступні плюси».

ФАКТ, НА ЯКОМУ ЦЕ СТОЇТЬ: для фіксованого `z` мережа Є добутком матриць
`W₃₁ D₃₀ W₃₀ … D₀ W₀`, де `D_l = diag(p_l > 0)`. Уся нелінійність — у виборі
маршруту `D` і в тому, як маршрути КОРЕЛЮЮТЬ між шарами.

🚨 ЗАПЕРЕЧЕННЯ ДО ПРОГОНУ: якщо `p` гаусів, то `P(D_l[i]=1, D_{l+1}[j]=1)`
ПОВНІСТЮ визначається `(m, C)` (orthant probability від `t_i,t_j,ρ`) ⟹ маски
не несуть НІЧОГО нового. Сигнал можливий ЛИШЕ у негаусовості.

ТЕСТ 1 — МІЖШАРОВА КОРЕЛЯЦІЯ МАСОК: реальна `corr(D_l[i], D_{l+1}[j])` проти
   гаусового передбачення з ТИХ САМИХ реальних `(m,C)`. Різниця = немоментний
   сигнал. 🔒 КОНТРОЛЬ: та сама перевірка ВСЕРЕДИНІ шару (там ρ відома точно)
   і на шарі 0 (де `p` гаусів ТОЧНО ⟹ розбіжність мусить бути НУЛЬ).

ТЕСТ 2 — ПРЯМО НА ФОРМУЛЮВАННЯ ЮЗЕРА: чи впливає «скільки закрилось раніше»
   на вихід? Для кожного семпла рахуємо `n_closed(l)` і дивимось, чи корелює
   профіль закритості з ПОХИБКОЮ замикання на цьому семплі.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) шар 0: розбіжність маскової кореляції = шум (🔒пломба)            90%
 (б) на глибині розбіжність буде НЕНУЛЬОВА, але < 5% від самої кореляції 60%
 (в) `n_closed` НЕ пояснить похибку виходу (|corr| < 0.1)               70%
 (г) міжшарова кореляція масок сильно спадає з відстанню шарів          80%
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
NPAIR = 4000          # скільки пар (i,j) брати на кожну пару шарів


def bivar_orthant(t1, t2, rho, nq=64):
    """P(X>-t1, Y>-t2) для стандартної двовимірної нормальної з кореляцією rho.
    Інтегруємо по одній змінній: P = ∫ φ(u) Φ((t2 + ρu)/√(1−ρ²)) du, u > −t1."""
    xs, ws = np.polynomial.legendre.leggauss(nq)
    lo, hi = -8.0, 8.0
    u = 0.5 * (hi - lo) * xs + 0.5 * (hi + lo)
    w = 0.5 * (hi - lo) * ws
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    out = np.zeros_like(t1)
    for k in range(nq):
        uk = u[k]
        msk = uk > -t1
        arg = (t2 + rho * uk) / q
        out += w[k] * (math.exp(-0.5 * uk * uk) / SQ2PI) * ndtr(arg) * msk
    return out


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

print(f"mask100 · {NNET} мереж × {2*NS} семплів · {NPAIR} пар (i,j) на зріз",
      flush=True)
rng0 = np.random.default_rng(20260809)

PAIRS = [(0, 1), (1, 2), (4, 5), (8, 9), (16, 17), (24, 25), (30, 31),
         (8, 12), (8, 16), (8, 24)]
acc = {p: [] for p in PAIRS}
selfacc = {}
prof = []

for net in range(NNET):
    W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
         for L in range(DEPTH)]
    rng = np.random.default_rng(555 + net)
    z = rng.standard_normal((NS, W_))
    A = np.concatenate([z, -z], axis=0)
    N = A.shape[0]

    P = []          # передактивації кожного шару
    for i in range(DEPTH):
        p = A @ W[i]
        P.append(p)
        if i < DEPTH - 1:
            A = np.maximum(p, 0.0)

    D = [(p > 0.0) for p in P]                     # маски
    nclosed = np.array([(~d).sum(1) for d in D[:-1]], dtype=float).T  # (N, 31)

    for (l1, l2) in PAIRS:
        p1, p2 = P[l1], P[l2]
        m1, m2 = p1.mean(0), p2.mean(0)
        s1 = p1.std(0)
        s2 = p2.std(0)
        t1all, t2all = m1 / np.maximum(s1, 1e-30), m2 / np.maximum(s2, 1e-30)
        ii = rng0.integers(0, W_, NPAIR)
        jj = rng0.integers(0, W_, NPAIR)
        c1 = p1[:, ii] - m1[ii]
        c2 = p2[:, jj] - m2[jj]
        rho = (c1 * c2).mean(0) / np.maximum(s1[ii] * s2[jj], 1e-30)
        rho = np.clip(rho, -0.999, 0.999)
        # реальна спільна ймовірність «обидва відкриті»
        real = (D[l1][:, ii] & D[l2][:, jj]).mean(0)
        # гаусове передбачення з ТИХ САМИХ (m,C)
        pred = bivar_orthant(t1all[ii], t2all[jj], rho)
        rel = np.abs(real - pred).mean() / max(np.abs(real).mean(), 1e-30)
        noise = float(np.sqrt(0.25 / N) / max(np.abs(real).mean(), 1e-30))
        acc[(l1, l2)].append((rel, noise, float(np.abs(rho).mean())))

    # 🔒 КОНТРОЛЬ: те саме ВСЕРЕДИНІ шару 0, де `p` гаусів ТОЧНО
    p0 = P[0]
    m0, s0 = p0.mean(0), p0.std(0)
    t0 = m0 / np.maximum(s0, 1e-30)
    ii = rng0.integers(0, W_, NPAIR)
    jj = rng0.integers(0, W_, NPAIR)
    c0i = p0[:, ii] - m0[ii]
    c0j = p0[:, jj] - m0[jj]
    r0 = np.clip((c0i * c0j).mean(0) / np.maximum(s0[ii] * s0[jj], 1e-30),
                 -0.999, 0.999)
    real0 = (D[0][:, ii] & D[0][:, jj]).mean(0)
    pred0 = bivar_orthant(t0[ii], t0[jj], r0)
    selfacc[net] = (float(np.abs(real0 - pred0).mean()
                          / max(np.abs(real0).mean(), 1e-30)),
                    float(np.sqrt(0.25 / N) / max(np.abs(real0).mean(), 1e-30)))

    # ТЕСТ 2: чи пояснює профіль закритості похибку виходу?
    outv = np.maximum(P[DEPTH - 1], 0.0)
    err = outv.mean(1) - outv.mean()                # відхилення семпла
    cc = [float(np.corrcoef(nclosed[:, L], err)[0, 1]) for L in range(31)]
    prof.append(cc)
    print(f"  мережа {net} готова", flush=True)

print("\n" + "=" * 78)
print("ТЕСТ 1 · МАСКИ: реальна P(обидва відкриті) проти ГАУСОВОГО передбачення")
print(f"{'пара шарів':>12s}{'розбіжність':>14s}{'шум МК':>10s}{'відн. до шуму':>15s}{'сер|ρ|':>9s}")
for p in PAIRS:
    a = np.array(acc[p])
    rel, noi, rho = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
    print(f"{str(p):>12s}{100*rel:>13.4f}%{100*noi:>9.4f}%{rel/noi:>15.2f}{rho:>9.4f}")

s0 = np.array([selfacc[n] for n in selfacc])
print(f"\n🔒 КОНТРОЛЬ (шар 0 сам із собою, `p` гаусів ТОЧНО): "
      f"{100*s0[:,0].mean():.4f}% при шумі {100*s0[:,1].mean():.4f}% "
      f"⟹ ×{s0[:,0].mean()/s0[:,1].mean():.2f}")

print("\nТЕСТ 2 · чи пояснює «скільки закрилось на шарі L» відхилення виходу?")
pr = np.array(prof)
for L in (0, 1, 4, 8, 16, 24, 30):
    print(f"  шар {L:>2d}: corr = {pr[:, L].mean():+.4f} "
          f"± {pr[:, L].std(ddof=1)/math.sqrt(len(pr)):.4f}")
print(f"  МАКСИМУМ |corr| по всіх 31 шарах: {np.abs(pr).max():.4f}")
