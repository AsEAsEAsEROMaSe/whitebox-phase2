"""sphere100 — ЧИ Є РОЗКИД НОРМИ ВХОДУ ГОЛОВНИМ ДЖЕРЕЛОМ ДИСПЕРСІЇ НАШОЇ МК?

ТОТОЖНІСТЬ (мережа без біасів ⟹ однорідна ступеня 1):
    out(z) = R · out(û),   R = ‖z‖,  û = z/‖z‖
Для z ~ N(0,I) величини R і û НЕЗАЛЕЖНІ ⟹ E[out(z)] = E[R]·E[out(û)] ТОЧНО,
і E[R] = √2·Γ((n+1)/2)/Γ(n/2) — ЧИСЛО, без вибірки.

⟹ оцінювач `Ê = E[R]·mean_i out(û_i)` НЕЗМІЩЕНИЙ і має ІНШУ дисперсію:
    Var(out(z))   = Var(R)Var(o) + Var(R)m² + E[R]²Var(o)     ← пряма МК
    E[R]²Var(o)                                               ← через сферу
    де o = out(û), m = E[o]
⟹ РЕДУКЦІЯ = 1 + Var(R)/E[R]² · (1 + m²/Var(o))

🚨 ЧОМУ ЦЕ НЕ ТЕ, ЩО ЗАКРИЛА С98: там міряли `F1` як ДЕТЕРМІНОВАНУ заміну МК
(0.2778% проти планки 0.1454%) — і той вирок чинний. Тут ІНШИЙ об'єкт:
однорідність застосована як РЕДУКЦІЯ ДИСПЕРСІЇ вибіркової гілки.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) редукція > ×1.10 хоч на одній мережі                          60%
 (б) редукція > ×1.5 (тобто розкид норми домінує)                  25%
 (в) m²/Var(o) > 30 (вихід на сфері майже сталий)                  55%
 (г) 🔒 пломба незміщеності: |E[R]·mean(o) − mean(out(z))| у межах
     шуму обох оцінок                                             90%

🔒 ПЛОМБИ:
 (1) НЕЗМІЩЕНІСТЬ: обидві оцінки мусять збігатись у межах свого шуму
 (2) АНАЛІТИЧНЕ E[R] проти вибіркового mean(‖z‖) — збіг ~1/√(2N)
 (3) ТОТОЖНІСТЬ out(z) = R·out(û) — машинна точність
 (4) МАСШТАБУВАННЯ: редукція НЕ мусить залежати від N
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

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384


def ER_exact(n):
    """E[‖z‖] для z ~ N(0,I_n) — через lgamma, без вибірки."""
    return math.sqrt(2.0) * math.exp(math.lgamma((n + 1) / 2.0)
                                     - math.lgamma(n / 2.0))


def fwd(W, A):
    for i in range(DEPTH - 1):
        A = np.maximum(A @ W[i], 0.0)
    return A @ W[DEPTH - 1]                    # ПЕРЕДактивації останнього шару


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
x = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(x.type):
    x = x.flatten()
Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, W_, W_)

ERx = ER_exact(W_)
print(f"sphere100 · {NNET} мереж × {2*NS} семплів (антитетично)")
print(f"🔒 ПЛОМБА (2): E[R] аналітичне = {ERx:.9f}  (наївне √256 = 16)")

rows = []
for net in range(NNET):
    W = [np.ascontiguousarray(Wall[net][L], dtype=np.float64)
         for L in range(DEPTH)]
    rng = np.random.default_rng(31337 + net)
    z = rng.standard_normal((NS, W_))
    Z = np.concatenate([z, -z], axis=0)
    N = Z.shape[0]

    R = np.linalg.norm(Z, axis=1)
    U = Z / R[:, None]

    P = fwd(W, Z)                              # out(z)      (N, 256)
    O = fwd(W, U)                              # out(û)      (N, 256)

    # 🔒 ПЛОМБА (3): тотожність out(z) = R·out(û)
    idres = float(np.abs(P - R[:, None] * O).max()
                  / max(np.abs(P).max(), 1e-300))

    # цільова величина — E[ReLU(p)] останнього шару
    T = np.maximum(P, 0.0)                     # пряма МК на кожному нейроні
    S = np.maximum(O, 0.0)                     # на сфері

    # дисперсії ОЦІНОК середнього (по 256 нейронах — беремо суму дисперсій)
    var_direct = float(T.var(0, ddof=1).sum())
    var_sphere = float((ERx ** 2) * S.var(0, ddof=1).sum())
    red = var_direct / max(var_sphere, 1e-300)

    m2 = float((S.mean(0) ** 2).sum())
    vo = float(S.var(0, ddof=1).sum())
    varR_rel = float(R.var(ddof=1) / ERx ** 2)

    # 🔒 ПЛОМБА (1): незміщеність — обидві оцінки того самого числа
    e_dir = T.mean(0)
    e_sph = ERx * S.mean(0)
    gap = float(np.abs(e_dir - e_sph).sum() / np.abs(e_dir).sum())
    se_dir = float(np.sqrt(var_direct / N) / np.abs(e_dir).sum())

    rows.append((net, red, m2 / max(vo, 1e-300), varR_rel, gap, se_dir, idres))
    print(f"  мережа {net}: редукція ×{red:.4f} · m²/Var(o) = "
          f"{m2/max(vo,1e-300):>8.2f} · 🔒тотожність {idres:.2e} · "
          f"🔒розбіжність оцінок {gap:.2e} (шум {se_dir:.2e})", flush=True)

print("\n" + "=" * 78)
r = np.array([x[1] for x in rows])
print(f"РЕДУКЦІЯ ДИСПЕРСІЇ МК: ×{r.mean():.4f} ± {r.std(ddof=1)/math.sqrt(len(r)):.4f}"
      f"   (min {r.min():.4f} · max {r.max():.4f})")
q = np.array([x[2] for x in rows])
print(f"m²/Var(o) = {q.mean():.2f}   ⟹ передбачення формули: "
      f"×{1 + np.mean([x[3] for x in rows])*(1+q.mean()):.4f}")
print(f"Var(R)/E[R]² = {np.mean([x[3] for x in rows]):.6f}  (корпус: 0.0331)")
print(f"\n🔒 ПЛОМБА (1) незміщеність: max розбіжність "
      f"{max(x[4] for x in rows):.2e} при шумі {max(x[5] for x in rows):.2e}")
print(f"🔒 ПЛОМБА (3) тотожність:   max {max(x[6] for x in rows):.2e}")
print(f"\nПЕРЕКЛАД У БАЛ (V = 1.9227e-06, MSE = V·E²/(V+E²), E² = 2.2158e-05):")
V0, E2, base = 1.9227e-06, 2.2158e-05, 1.794922e-06
Vn = V0 / r.mean()
print(f"  V {V0:.4e} → {Vn:.4e}   MSE {base:.4e} → "
      f"{Vn*E2/(Vn+E2):.4e}   ⟹ ×{base/(Vn*E2/(Vn+E2)):.4f}")
