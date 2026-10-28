"""sweep99 — ЧИ Є ГАСІННЯ МЕХАНІЗМОМ, ЧИ АРТЕФАКТОМ КАЛІБРОВКИ (контроль
контролера, С99).

ПИТАННЯ: Мелер K=2 виглядає ТОЧНІШИМ за точні парні моменти (×4.14 у MSE, але
при `α=0` у квадратурній версії). Це гасіння похибок — чи просто `α`?

СПОСІБ, ЩО РОЗДІЛЯЄ: замінювати Мелера на квадратуру не «все або нічого», а на
ХВОСТІ шарів `L ≥ L0`. `α` ПЕРЕПІДБИРАЄТЬСЯ для кожної конфігурації (МНК-масштаб)
⟹ калібровка не може пояснити різницю МІЖ конфігураціями за побудовою.
```
гасіння накопичувальне (МЕХАНІЗМ) ⟹ похибка росте МОНОТОННО з числом замінених шарів
артефакт калібровки               ⟹ монотонності не буде, стрибок лише на «все»
```
🔒 ПЛОМБА: `L0 = 32` (жодного заміненого шару) мусить відтворити варіант «як у
сабміті без обрізання» = **2.354765e-05** `[decomp99, 50 мереж]`.

ПЕРЕДБАЧЕННЯ (A, до прогону): монотонність буде — 45%. Контролер: 65%.
ЗАПИСАНО ДО ПРОГОНУ, щоб не тлумачити заднім числом.
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
L0S = [32, 28, 24, 16, 8, 0]


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def psi(x):
    x = np.asarray(x, float)
    return phi(x) + x * ndtr(x)


def gl(a, b, npan, nnod):
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    return ((0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel(),
            (0.5 * (hi - lo) * ws[None, :]).ravel())


ZQ, WQ = gl(0.0, 40.0, 12, 20)


def pair_quad(t, rho, n, chunk=8192):
    tj = np.repeat(t, n)
    tk = np.tile(t, n)
    rr = rho.ravel()
    out = np.empty(n * n)
    q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-14))
    for o in range(0, n * n, chunk):
        sl = slice(o, min(o + chunk, n * n))
        x = ZQ[None, :] - tj[sl, None]
        u = (rr[sl, None] * x + tk[sl, None]) / q[sl, None]
        out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                   * psi(u)).sum(1)
    return out.reshape(n, n)


def closure(W, width, l0):
    """шлях sub99, але на шарах `L >= l0` позадіагональ береться КВАДРАТУРОЮ."""
    m = np.zeros(width)
    C = np.eye(width)
    for i in range(DEPTH):
        w = W[i]
        cp = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diagonal(cp), 1e-20))
        mp = m @ w
        t = mp / s
        ph = phi(t)
        Ph = ndtr(t)
        m = mp * Ph + s * ph
        if i == DEPTH - 1:
            break
        if i >= l0:
            rho = np.clip(cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
            C = pair_quad(t, rho, width) * (s[:, None] * s[None, :])
            C = C - np.outer(m, m)
        else:
            C = cp * np.outer(Ph, Ph) + (0.5 * (cp * cp)) * np.outer(ph / s,
                                                                     ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diagonal(C))
    return m


print(f"sweep99 · квадратура на хвості шарів · {NNET} мереж · істина N=1e9",
      flush=True)
M = {l: [] for l in L0S}
T = []
for fpath in sorted(glob.glob(SRC + "/data-00000-of-00002.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(fpath, "rb")).read_all()
    x = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, 256, 256)
    yv = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(yv.type):
        yv = yv.flatten()
    FM = np.asarray(yv).reshape(tb.num_rows, 256)
    for i in range(min(tb.num_rows, NNET)):
        W = [np.ascontiguousarray(Wall[i][L], dtype=np.float64)
             for L in range(DEPTH)]
        T.append(FM[i].astype(np.float64))
        for l in L0S:
            M[l].append(closure(W, 256, l))
        print(f"  {i+1}/{NNET}", flush=True)

T = np.array(T)
print("\n" + "=" * 78)
print(f"{'L0':>4s}{'замінено шарів':>16s}{'MSE (α перепідібр.)':>22s}"
      f"{'відн.':>10s}{'× до L0=32':>12s}")
print("=" * 78)
ref = None
for l in L0S:
    A = np.array(M[l])
    a = float((A * T).sum() / max((A * A).sum(), 1e-300))
    mse = float(((a * A - T) ** 2).mean())
    if ref is None:
        ref = mse
    tag = ""
    if l == 32:
        tag = f"  🔒 ПЛОМБА: decomp99 дав 2.354765e-05 ⟹ Δ {100*(mse/2.354765e-05-1):+.2f}%"
    print(f"{l:>4d}{DEPTH-1-l if l < DEPTH else 0:>16d}{mse:>22.6e}"
          f"{100*math.sqrt(mse/0.9097):>9.4f}%{mse/ref:>12.4f}{tag}")
print("\nМОНОТОННІСТЬ ⟹ механізм (накопичувальне гасіння)."
      "\nСТРИБОК ЛИШЕ НА КРАЮ ⟹ артефакт калібровки.")
