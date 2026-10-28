"""early100 — ЩО ДАСТЬ ТОЧНЕ ЗНАННЯ СТАНУ НА ПЕРШИХ ШАРАХ? (питання ЮЗЕРА, С100)

Замикання стартує з ТОЧНОГО стану (m=0, C=I) і псується з першим же ReLU.
Питання: якщо вкинути ІСТИННІ (m, C) після шару K і далі йти як звичайно —
скільки це купує? Скануємо K = 1, 2, 3, 4, 8, 16, 24.

🚨 ЦЕ ОРАКУЛ: істинні m,C беруться з МК високої точності. Доставки не існує —
вимір показує СТЕЛЮ, а не метод.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ, записано в діалозі):
 (а) K=1  ×1.03 у замиканні                                        60%
 (б) K=2  ×1.05                                                    60%
 (в) приріст РОСТЕ з K (узгоджено з varmap)                        85%
 (г) K=1,2 у балі дають ≤×1.01                                     80%

🔒 ПЛОМБИ:
 (1) K=0 (вкинути точний стан на вході) мусить дати РІВНО базу — стан там і так
     точний ⟹ Δ = 0.00e+00
 (2) шум оракула: половини вибірки МК мусять дати той самий результат
 (3) K=depth-1 ⟹ похибка мусить впасти майже до нуля (лишається один крок)
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
KS = [None, 1, 2, 4, 8, 24]


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def truth_state(W, NS, seed, half=None):
    """істинні (m, C) активацій ПІСЛЯ кожного шару, з МК."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((NS, W_))
    A = np.concatenate([z, -z], axis=0)
    if half == 0:
        A = A[:len(A) // 2]
    elif half == 1:
        A = A[len(A) // 2:]
    st = []
    for i in range(DEPTH - 1):
        A = np.maximum(A @ W[i], 0.0)
        m = A.mean(0)
        Ac = A - m
        st.append((m, (Ac.T @ Ac) / len(A)))
    return st


def closure_from(W, kap, inject=None, state=None):
    """замикання; якщо inject=K — після шару K стан замінюється на істинний."""
    m = np.zeros(W_)
    C = np.eye(W_)
    for i in range(DEPTH):
        w = W[i]
        cp = w.T @ C @ w
        if i > 8:
            cp = cp * kap
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
        if inject is not None and i == inject:
            m, C = state[i][0].copy(), state[i][1].copy()
    return m


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

print(f"early100 · {NNET} мереж · оракул із МК {2*NS} семплів · ціль = банк N=1e9",
      flush=True)
res = {k: [] for k in KS}
half = {0: [], 1: []}
T = []
for net in range(NNET):
    W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
         for L in range(DEPTH)]
    T.append(FM[net].astype(np.float64))
    st = truth_state(W, NS, 4242 + net)
    for k in KS:
        res[k].append(closure_from(W, 0.9985, inject=k, state=st))
    for h in (0, 1):
        sth = truth_state(W, NS, 4242 + net, half=h)
        half[h].append(closure_from(W, 0.9985, inject=2, state=sth))
    print(f"  {net+1}/{NNET}", flush=True)

T = np.array(T)


def mse(M):
    M = np.array(M)
    a = float((M * T).sum() / (M * M).sum())
    return float(((a * M - T) ** 2).mean())


base = mse(res[None])
print("\n" + "=" * 72)
print(f"{'вкидаємо істинний (m,C) після шару':<38s}{'MSE':>14s}{'×база':>10s}")
print("=" * 72)
print(f"{'— (база, чисте замикання)':<38s}{base:>14.6e}{1.0:>10.4f}")
for k in KS[1:]:
    v = mse(res[k])
    tag = "   🔒 ПЛОМБА: мусить бути Δ=0" if k == 0 else ""
    print(f"{'K = ' + str(k):<38s}{v:>14.6e}{base/v:>10.4f}{tag}")

print(f"\n🔒 ПЛОМБА (1) K=0: Δ = {abs(mse(res[0]) - base):.2e} "
      f"(стан на вході й так точний)")
print(f"🔒 ПЛОМБА (2) шум оракула, K=2 по половинах вибірки: "
      f"{base/mse(half[0]):.4f} vs {base/mse(half[1]):.4f}")

V, E2b = 1.9227e-06, 2.2158e-05
print(f"\nПЕРЕКЛАД У БАЛ (V={V:.4e}, модель проти моделі):")
m0 = V * E2b / (V + E2b)
for k in (1, 2, 4, 8, 16, 24):
    E2 = E2b / (base / mse(res[k]))
    print(f"  K={k:<3d} E² {E2:.4e}  MSE {V*E2/(V+E2):.4e}  "
          f"⟹ ×{m0/(V*E2/(V+E2)):.4f}")


# ═══ ПЛОМБА МАСШТАБУВАННЯ: виграш мусить РОСТИ з N оракула і вийти на плато ═══
print("\n" + "=" * 72)
print("🔒 ПЛОМБА МАСШТАБУВАННЯ ОРАКУЛА — шум падає як 1/√N, сигнал стоїть")
print(f"{'N семплів':>12s}" + "".join(f"{'K='+str(k):>11s}" for k in (1, 2, 4, 8, 24)))
for NSx in (4096, 16384, 65536):
    row = {k: [] for k in (1, 2, 4, 8, 24)}
    for net in range(NNET):
        W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
             for L in range(DEPTH)]
        st = truth_state(W, NSx, 4242 + net)
        for k in (1, 2, 4, 8, 24):
            row[k].append(closure_from(W, 0.9985, inject=k, state=st))
    print(f"{2*NSx:>12d}" + "".join(f"{base/mse(row[k]):>11.4f}"
                                    for k in (1, 2, 4, 8, 24)), flush=True)
