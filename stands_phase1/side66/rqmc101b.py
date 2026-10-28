"""rqmc101b — ВИРОК ПО RQMC. ПАРНО, ОДНАКОВІ МЕРЕЖІ, 8 ЗСУВІВ, ТІЛЬКИ N=6144.

`[rqmc101]` дав ×1.14, але при 48 вимірах шум ~14% ⟹ РІВНО на межі. Закон:
малі прогони не розсуджують. Тут: 24 мережі × 8 зсувів = 192 виміри на схему,
і ПАРНО — той самий зсув-сід для обох схем на тій самій мережі ⟹ спільна
компонента шуму скорочується.
🔒 знаковий тест по мережах — головний вирок (як для κ у С100).
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtri                         # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NSH = int(sys.argv[2]) if len(sys.argv) > 2 else 8
N = 6144

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_).astype(np.float64)
NNET = min(NNET, tb.num_rows)


def roberts_alpha(d):
    x = 1.5
    for _ in range(200):
        f = x ** (d + 1) - x - 1.0
        fp = (d + 1) * x ** d - 1.0
        x -= f / fp
        if x <= 1.0:
            x = 1.0 + 1e-12
    return np.array([x ** (-(j + 1)) for j in range(d)])


AL = cp.asarray(roberts_alpha(W_))
print(f"rqmc101b · {NNET} мереж × {NSH} зсувів · N={N} · ПАРНО", flush=True)


def pts(kind, n, seed):
    rng = cp.random.default_rng(seed)
    if kind == "mcw":
        z = rng.standard_normal((n, W_), dtype=cp.float64)
    else:
        i = cp.arange(1, n + 1, dtype=cp.float64)[:, None]
        sh = rng.random((1, W_), dtype=cp.float64)
        z = ndtri(cp.clip(cp.mod(i * AL[None, :] + sh, 1.0), 1e-12, 1 - 1e-12))
    z = z - z.mean(0, keepdims=True)
    cov = (z.T @ z) / z.shape[0]
    L = cp.linalg.cholesky(cov + 1e-12 * cp.eye(W_))
    z = z @ cp.linalg.inv(L).T
    return cp.concatenate([z, -z], axis=0).astype(cp.float32)


def fwd(Wg, a):
    for i in range(DEPTH):
        p = a @ Wg[i]
        a = cp.maximum(p, 0.0) if i < DEPTH - 1 else p
    return cp.maximum(a, 0.0).mean(0, dtype=cp.float64)


A, B = [], []
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    T = cp.asarray(FM[net])
    ea, eb = [], []
    for sh in range(NSH):
        sd = 4242 + 977 * net + 13 * sh          # 🔒 ТОЙ САМИЙ сід обом
        ea.append(float(cp.mean((fwd(Wg, pts("mcw", N, sd)) - T) ** 2)))
        eb.append(float(cp.mean((fwd(Wg, pts("rqw", N, sd)) - T) ** 2)))
    A.append(float(np.mean(ea)))
    B.append(float(np.mean(eb)))
    print(f"  {net:>3d}  МК {A[-1]:.4e}   RQMC {B[-1]:.4e}   "
          f"×{A[-1]/B[-1]:.3f}", flush=True)

A, B = np.array(A), np.array(B)
w = int((B < A).sum())
print(f"\n{'':<26s}{'MSE':>14s}{'×виграш':>10s}")
print("=" * 52)
print(f"{'МК + вибілювання (НАША)':<26s}{A.mean():>14.6e}{1.0:>10.4f}")
print(f"{'RQMC + вибілювання':<26s}{B.mean():>14.6e}{A.mean()/B.mean():>10.4f}")
print(f"\n🔒 ЗНАКОВИЙ ТЕСТ: RQMC краща на {w}/{NNET} мережах")
pv = sum(math.comb(NNET, k) for k in range(w, NNET + 1)) / 2 ** NNET
print(f"   одностороннє p = {pv:.5f}")
print(f"   медіана відношення ×{np.median(A/B):.4f}  ·  "
      f"[{(A/B).min():.3f} … {(A/B).max():.3f}]")
V0, E2 = 1.9227e-06, 2.3385e-05
g = A.mean() / B.mean()
m0 = V0 * E2 / (V0 + E2)
m1 = (V0 / g) * E2 / (V0 / g + E2)
print(f"\nПЕРЕВЕДЕННЯ В БАЛ: V {V0:.4e} → {V0/g:.4e} ⟹ "
      f"MSE {m0:.4e} → {m1:.4e} = ×{m0/m1:.4f}")
