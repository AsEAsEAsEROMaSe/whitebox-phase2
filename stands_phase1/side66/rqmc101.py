"""rqmc101 — RQMC ЗАМІСТЬ ПСЕВДОВИПАДКОВОЇ МК. Б'Є В 91% БАЛУ.

ЗВІДКИ: публічний write-up учасника `radiant-allomancer` (сабміт 317660,
adjusted 4.47e-7) на форумі AIcrowd. Його ядро — RQMC-префікс:
    u_i = frac(i·α + shift),  z_i = Φ^{-1}(u_i),  α = вектор Робертса
і він міряє спад ШВИДШИЙ за 1/N (p>1) у номінальних 256 вимірах.

ЧОМУ В НАС ПОТЕНЦІАЛ БІЛЬШИЙ, НІЖ У НЬОГО: у нього ЗМІЩЕННЯ домінує
(bias² ≥ 59.4%) ⟹ RQMC вичерпане. У нас `V` = **91% балу** ⟹ RQMC б'є рівно
туди. І умова застосовності виконана: ми ЗМІРЯЛИ малу ефективну розмірність
(`n_eff` ЦГТ 86→25, ефективний ранг `Cov(out)` ~25).

🚨 ГОЛОВНИЙ РИЗИК — КОНФЛІКТ ІЗ ВИБІЛЮВАННЯМ. Наше вибілювання дає ×2.06 і
   виправляє ВСІ другі моменти вибірки; RQMC покращує рівномірність. Так само
   виглядала сферизація — і вона дала ×0.74 ПОВЕРХ вибілювання (С100).
   ⟹ міряємо ВСІ ЧОТИРИ комбінації, а не лише RQMC соло.

🔒 ПЛОМБИ:
  (1) чиста МК має відтворити спад рівно N^(−0.5)
  (2) 🔒 МАСШТАБУВАННЯ: три `N` ⟹ показник `p` фітується, а не оголошується
  (3) кілька зсувів Кренлі-Паттерсона ⟹ RQMC незміщена, розкид дає похибку
  (4) ціль — `final_means` (1e9) ⟹ незалежна від обох гілок

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) RQMC соло поб'є МК соло щонайменше в 1.3×                      65%
 (б) RQMC + вибілювання поб'є вибілювання соло                      40%
 (в) показник `p` для RQMC вийде > 0.6 (тобто краще за 1/√N)        55%
 (г) виграш у ПОВНОМУ бленді буде вдвічі меншим, ніж у чистій `V`   70%
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NS = [1536, 3072, 6144]
NSHIFT = 4

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
    """φ_d: єдиний дійсний корінь x^(d+1) = x + 1; α_j = φ^(−(j+1))"""
    x = 1.5
    for _ in range(200):
        f = x ** (d + 1) - x - 1.0
        fp = (d + 1) * x ** d - 1.0
        x -= f / fp
        if x <= 1.0:
            x = 1.0 + 1e-12
    return np.array([x ** (-(j + 1)) for j in range(d)])


AL = cp.asarray(roberts_alpha(W_))
print(f"rqmc101 · {NNET} мереж · N ∈ {NS} · {NSHIFT} зсувів\n"
      f"🔒 Roberts α[0]={float(AL[0]):.6f} α[255]={float(AL[255]):.6e}", flush=True)


def pts(kind, n, seed):
    """повертає (2n, 256) точок: антитетична пара для будь-якого kind"""
    rng = cp.random.default_rng(seed)
    if kind.startswith("mc"):
        z = rng.standard_normal((n, W_), dtype=cp.float64)
    else:
        i = cp.arange(1, n + 1, dtype=cp.float64)[:, None]
        sh = rng.random((1, W_), dtype=cp.float64)
        u = cp.mod(i * AL[None, :] + sh, 1.0)
        u = cp.clip(u, 1e-12, 1.0 - 1e-12)
        z = ndtri(u)
    if kind.endswith("w"):                       # вибілювання
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


KINDS = ["mc", "mcw", "rq", "rqw"]
LAB = {"mc": "МК гола", "mcw": "МК + вибілювання (НАША)",
       "rq": "RQMC гола", "rqw": "RQMC + вибілювання"}
res = {(k, n): [] for k in KINDS for n in NS}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float32))
          for i in range(DEPTH)]
    T = cp.asarray(FM[net])
    for n in NS:
        for k in KINDS:
            e = []
            for sh in range(NSHIFT):
                v = fwd(Wg, pts(k, n, 9000 + 137 * net + 31 * sh + 7 * NS.index(n)))
                e.append(float(cp.mean((v - T) ** 2)))
            res[(k, n)].append(float(np.mean(e)))
    print(f"  мережа {net} готова", flush=True)

print(f"\n{'схема':<28s}" + "".join(f"{'N=' + str(n):>13s}" for n in NS)
      + f"{'показник p':>13s}{'×проти НАШОЇ':>14s}")
print("=" * 96)
base = {n: float(np.mean(res[("mcw", n)])) for n in NS}
for k in KINDS:
    v = [float(np.mean(res[(k, n)])) for n in NS]
    p = -0.5 * (math.log(v[-1]) - math.log(v[0])) / (math.log(NS[-1]) - math.log(NS[0]))
    print(f"{LAB[k]:<28s}" + "".join(f"{x:>13.5e}" for x in v)
          + f"{p:>13.3f}{base[NS[-1]]/v[-1]:>14.4f}")
print("\n🔒 (1) «МК гола» мусить дати p ≈ 0.500 — це перевірка приладу")
print("🔒 (2) p — фіт по трьох N, а не оголошення")
print("🚨 вирок за стовпцем «×проти НАШОЇ» при N=6144 (бюджет сабміту)")
