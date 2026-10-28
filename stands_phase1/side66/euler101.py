"""euler101 — ТОТОЖНІСТЬ ЕЙЛЕРА+СТАЙНА: E[out] = E[Δout]. ЧИСЕЛЬНА ПЕРЕВІРКА.

ВИВЕДЕННЯ (не вгадано — виведено з двох тотожностей):
  1. мережа БЕЗ БІАСІВ позитивно однорідна степеня 1 ⟹ Ейлер:  out(z) = z·∇out
  2. `z ~ N(0,I)` ⟹ Стайн:  E[z_i·g] = E[∂_i g]
  ⟹ E[out] = Σ_i E[z_i ∂_i out] = Σ_i E[∂_i∂_i out] = **E[Δ out]**

`out` кусково-ЛІНІЙНА ⟹ `Δout` = міра, зосереджена на зламах:
локально біля `{p_l[j]=0}` маємо `out_a = A(z) + u·ReLU(p_l[j])` з лінійними
`A`, `u` ⟹ `Δ ReLU(f) = ‖∇f‖²·δ(f)` (бо `Δf = 0` локально). Отже

    Δout_a = Σ_{l=0..30} Σ_j  B_l[j,a] · ‖J_l[:,j]‖² · δ(p_l[j])
             +                ‖J_31[:,a]‖² · δ(p_31[a])          ← злам самого виходу

    J_l = J_{l-1} diag(D_{l-1}) W_l ,  J_0 = W_0        (∇_z p_l, стовпцями)
    B_{l-1} = W_l diag(D_l) B_l ,      B_30 = W_31 diag(D_31)   (∂out_a/∂h_l[j])

🔒 ПЛОМБА АНАЛІТИЧНА (зроблена ДО коду): один шар, `out = ReLU(w·z)`, `m=0`:
   `Δ = ‖w‖²δ(w·z)` ⟹ `E = ‖w‖²·φ(0)/‖w‖ = ‖w‖/√(2π) = σ/√(2π)` — РІВНО істина.

ЧОМУ ЦЕ ВАЖЛИВО `[орієнтир: orac101, С101]`: ідеальне ДВОМОМЕНТНЕ замикання дає
лише #5 місце ⟹ клас, що несе моменти, розв'язку НЕ МІСТИТЬ. Ця тотожність
не використовує моментів узагалі: об'єкт — ІНТЕГРАЛИ ПО ПОВЕРХНЯХ ЗЛАМІВ.
Вона ж пояснює, чому маски ReLU (С100) несли сигнал ×17.7 над шумом: маски —
це ТІНЬ правильного об'єкта, а правильний об'єкт — межі масок.

ЯК ПЕРЕВІРЯЄМО. `δ` наближаємо гаусовим ядром ширини `ε·s_l[j]` (s = std p).
При `ε→0` зміщення →0, дисперсія →∞ ⟹ дивимось СХОДЖЕННЯ по `ε`, а не одне
число. Обидві сторони рахуються на ТИХ САМИХ семплах ⟹ парне порівняння.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒 глибина 1: збіг у межах 1% на середньому `ε`                  85%
 (б) глибина 32: збіг у межах 5% хоч на якомусь `ε`                   55%
 (в) буде видно плато по `ε` (ознака правильної формули)              50%
 (г) дисперсія оцінки Δ буде БІЛЬШОЮ за пряму МК                      75%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402

W_ = 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NET = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEPTHS = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2
                           else "1,2,4,32").split(",")]
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 8192
BATCH = 64
EPS = [0.015, 0.020, 0.025, 0.030, 0.040, 0.060]

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не одну карту", flush=True)
    sys.exit(2)

f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, 32, W_, W_)
print(f"euler101 · мережа {NET} · глибини {DEPTHS} · {NS} семплів · "
      f"ε = {EPS}", flush=True)


def one_depth(L):
    """L = кількість шарів; out_a = ReLU(p_{L-1}[a])"""
    Wg = [cp.asarray(np.ascontiguousarray(Wall[NET][i], dtype=np.float32))
          for i in range(L)]
    rng = cp.random.default_rng(4242)
    # ── прохід 1: масштаби s_l[j] (для ширини ядра) і пряма МК ──
    S1 = cp.zeros((L, W_), dtype=cp.float64)
    S2 = cp.zeros((L, W_), dtype=cp.float64)
    OUT = cp.zeros(W_, dtype=cp.float64)
    n = 0
    while n < NS:
        nb = min(1024, NS - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = z
        for i in range(L):
            p = a @ Wg[i]
            S1[i] += p.sum(0, dtype=cp.float64)
            S2[i] += (p.astype(cp.float64) ** 2).sum(0)
            a = cp.maximum(p, 0.0)
        OUT += a.sum(0, dtype=cp.float64)
        n += nb
    m_p = S1 / NS
    s_p = cp.sqrt(cp.maximum(S2 / NS - m_p * m_p, 1e-30))
    mc = OUT / NS                                   # пряма МК: E[out]

    # ── прохід 2: E[Δout] через м'яку дельту ──
    rng = cp.random.default_rng(4242)               # ТІ САМІ семпли — парно
    acc = {e: cp.zeros(W_, dtype=cp.float64) for e in EPS}
    n = 0
    while n < NS:
        nb = min(BATCH, NS - n)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        P, D = [], []
        a = z
        J = cp.tile(Wg[0], (nb, 1, 1))              # ∇_z p_0, стовпцями
        Js = []
        for i in range(L):
            p = a @ Wg[i]
            P.append(p)
            d = (p > 0).astype(cp.float32)
            D.append(d)
            Js.append(J)
            if i < L - 1:
                a = cp.maximum(p, 0.0)
                J = cp.matmul(J * d[:, None, :], Wg[i + 1])
        # зворотні коефіцієнти B_l[j,a] = ∂out_a/∂h_l[j]
        B = [None] * L
        if L >= 2:
            B[L - 2] = Wg[L - 1] * D[L - 1][:, None, :]
            for i in range(L - 3, -1, -1):
                B[i] = cp.matmul(Wg[i + 1] * D[i + 1][:, None, :], B[i + 1])
        gn = [cp.sum(Js[i] * Js[i], axis=1) for i in range(L)]   # ‖J_l[:,j]‖²
        for e in EPS:
            wdt = cp.maximum(e * s_p, 1e-20)                     # (L,256)
            tot = cp.zeros((nb, W_), dtype=cp.float64)
            for i in range(L):
                ker = (cp.exp(-0.5 * (P[i] / wdt[i]) ** 2)
                       / (SQ2PI * wdt[i]))                       # δ_ε(p_l)
                g = (gn[i] * ker).astype(cp.float64)             # (nb,256)
                if i == L - 1:
                    tot += g                                     # злам виходу
                else:
                    tot += cp.squeeze(cp.matmul(g[:, None, :].astype(cp.float32),
                                                B[i]), 1).astype(cp.float64)
            acc[e] += tot.sum(0)
        n += nb
    return mc, {e: acc[e] / NS for e in EPS}


print(f"\n{'L':>3s}{'ε':>7s}{'E[Δout]':>14s}{'E[out] (МК)':>14s}"
      f"{'відношення':>12s}{'коментар':>16s}")
print("=" * 68)
for L in DEPTHS:
    mc, dd = one_depth(L)
    tgt = float(cp.mean(mc))
    if L == 1:
        w0 = cp.asarray(Wall[NET][0].astype(np.float64))
        exact = float(cp.mean(cp.sqrt(cp.sum(w0 * w0, axis=0)))) / SQ2PI
        print(f"{L:>3d}{'':>7s}{'':>14s}{tgt:>14.6f}{'':>12s}"
              f"  🔒точно {exact:.6f}  ×{tgt/exact:.4f}")
    for e in EPS:
        v = float(cp.mean(dd[e]))
        print(f"{L:>3d}{e:>7.2f}{v:>14.6f}{tgt:>14.6f}{v/tgt:>12.4f}",
              flush=True)
    print("-" * 68)
