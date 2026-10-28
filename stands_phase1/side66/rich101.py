"""rich101 — ЕКСТРАПОЛЯЦІЯ РІЧАРДСОНА ПО ШИРИНІ. ПОВНІСТЮ ДЕТЕРМІНОВАНА.

ВИВЕДЕННЯ (з тотожності, не з голови). `[ЗМІРЯНО width101]` після He-перенормування
підматриць похибка замикання падає РІВНО як `1/n`: нахили −0.897 … −1.135 на
семи інтервалах, і n=256 відтворює записану сиру похибку 0.9599% (дало 0.9583%).
Якщо
        m_cl(n) = m_true + c/n
то з ДВОХ замикань різної ширини `c` виключається БЕЗ ІСТИНИ:
        m_true = 2·m_cl(256) − m_cl(128)          (Річардсон, r=2)
        загально: m_true = (m_cl(n₂)·n₂ − m_cl(n₁)·n₁)/(n₂ − n₁)
ЦІНА: одне додаткове замикання ширини n₁ = (n₁/256)³ від повного
      ⟹ при n₁=128 це **0.125 × 1.66e9 = 2.1e8 = 0.08% бюджету**.

🚨 КРИТИЧНЕ ПРИПУЩЕННЯ, ЯКЕ ТУТ І ПЕРЕВІРЯЄТЬСЯ: `m_true(n₁)` для ПІДМЕРЕЖІ
   мусить дорівнювати `m_true(256)` на спільних нейронах. Це НЕ очевидно —
   підмережа є іншою мережею. Якщо ні, Річардсон віднімає не те.

🔒 ПЛОМБИ:
  (1) corr(m_true(128), m_true(256)) — якщо низька, метод мертвий одразу
  (2) ПРЯМИЙ ТЕСТ: чи `m_cl(128) − m_cl(256)` корелює з реальною похибкою e(256)
  (3) 🔒 ЧУЖА МЕРЕЖА для (2) — мусить дати нуль
  (4) кінцевий вирок: похибка ПІСЛЯ Річардсона проти похибки ДО, з ФІТОВАНИМ
      коефіцієнтом (закон С88: не прибивати вагою 1)

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) 🔒(1) corr(m_true(128), m_true(256)) > 0.9                        40%
 (б) 🔒(2) corr(різниця замикань, реальна похибка) > 0.3               35%
 (в) Річардсон із вагою 1 ПОГІРШИТЬ (класична пастка)                  75%
 (г) з фітованим коефіцієнтом дасть виграш > ×1.05                     25%
"""
import glob
import math
import sys

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 524288
NS1 = [96, 128, 160, 192, 224]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
tb = pa.ipc.open_stream(pa.memory_map(sorted(glob.glob(SRC + "/data-*.arrow"))[0],
                                      "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
print(f"rich101 · {NNET} мереж · {NS} семплів · n₁ ∈ {NS1}", flush=True)


def SCL(n):
    return math.sqrt(W_ / n)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def closure(Wg, n):
    m = cp.zeros(n, dtype=cp.float64)
    C = cp.eye(n, dtype=cp.float64)
    for i in range(DEPTH):
        w = Wg[i][:n, :n] * SCL(n)
        cpm = w.T @ C @ w
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - mn * mn
        Co = cpm * cp.outer(Ph, Ph) + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s)
        C = Co + cp.diag(d - cp.diagonal(Co))
        m = mn


def truth(Wg, n, ns, seed):
    rng = cp.random.default_rng(seed)
    acc = cp.zeros(n, dtype=cp.float64)
    done = 0
    while done < ns:
        nb = min(16384, ns - done)
        z = rng.standard_normal((nb, n), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            p = a @ (Wg[i][:n, :n] * SCL(n)).astype(cp.float32)
            a = cp.maximum(p, 0.0) if i < DEPTH - 1 else p
        acc += cp.maximum(a, 0.0).sum(0, dtype=cp.float64)
        done += nb
    return acc / (2 * ns)


CL = {n: [] for n in NS1 + [W_]}
TR = {n: [] for n in NS1 + [W_]}
for net in range(NNET):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][i], dtype=np.float64))
          for i in range(DEPTH)]
    for n in NS1 + [W_]:
        CL[n].append(cp.asnumpy(closure(Wg, n)))
        TR[n].append(cp.asnumpy(truth(Wg, n, NS, 777 + net)))
    print(f"  мережа {net} готова", flush=True)


def rel(e, t):
    return math.sqrt((e * e).mean() / (t * t).mean())


print(f"\n🔒(1) ЧИ ТА САМА ІСТИНА НА СПІЛЬНИХ НЕЙРОНАХ?")
print(f"{'n₁':>5s}{'corr(m_true)':>14s}{'віднос.розбіжність':>20s}")
for n in NS1:
    cs = [np.corrcoef(TR[n][k], TR[W_][k][:n])[0, 1] for k in range(NNET)]
    dv = [rel(TR[n][k] - TR[W_][k][:n], TR[W_][k][:n]) for k in range(NNET)]
    print(f"{n:>5d}{np.mean(cs):>14.4f}{100*np.mean(dv):>19.2f}%")

print(f"\n🔒(2) ЧИ РІЗНИЦЯ ЗАМИКАНЬ БАЧИТЬ РЕАЛЬНУ ПОХИБКУ?")
print(f"{'n₁':>5s}{'corr(Δcl, e)':>14s}{'🔒чужа':>10s}{'вага 1':>10s}"
      f"{'ФІТ β*':>10s}{'×виграш':>10s}")
for n in NS1:
    cs, cf, g1, gb, bs = [], [], [], [], []
    for k in range(NNET):
        e = CL[W_][k] - TR[W_][k]                    # реальна похибка (256)
        dcl = CL[n][k] - CL[W_][k][:n]               # ДЕТЕРМІНОВАНА різниця
        en = e[:n]
        cs.append(np.corrcoef(dcl, en)[0, 1])
        o = (k + 1) % NNET
        cf.append(np.corrcoef(CL[n][o] - CL[W_][o][:n], en)[0, 1])
        # Річардсон з вагою 1: m_true = m_cl(256) − c/256, c = Δcl·n·n₂/(n₂−n)
        w1 = n / (W_ - n)
        g1.append(rel(en, TR[W_][k][:n]) /
                  max(rel(en - w1 * dcl, TR[W_][k][:n]), 1e-30))
        b = float(np.dot(dcl, en) / max(np.dot(dcl, dcl), 1e-30))
        bs.append(b)
        gb.append(rel(en, TR[W_][k][:n]) /
                  max(rel(en - b * dcl, TR[W_][k][:n]), 1e-30))
    print(f"{n:>5d}{np.mean(cs):>14.4f}{np.mean(cf):>10.4f}"
          f"{np.mean(g1):>10.4f}{np.mean(bs):>10.4f}{np.mean(gb):>10.4f}")
print("\nвага 1 = теоретичний Річардсон · ФІТ β* = оракульний коефіцієнт (СТЕЛЯ)")

np.savez("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
         "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/rich101.npz",
         **{f"cl{n}": np.array(CL[n]) for n in NS1 + [W_]},
         **{f"tr{n}": np.array(TR[n]) for n in NS1 + [W_]})

print("\n🔒(5) ЧИ ПЕРЕНОСИТЬСЯ β МІЖ МЕРЕЖАМИ? (без цього — оракул, не метод)")
print(f"{'n1':>5s}{'β медіана':>12s}{'розкид β':>11s}{'LOO ×виграш':>13s}"
      f"{'крос A>B':>11s}{'ЧУЖА β':>10s}")
for n in NS1:
    D = [CL[n][k] - CL[W_][k][:n] for k in range(NNET)]
    E = [(CL[W_][k] - TR[W_][k])[:n] for k in range(NNET)]
    T = [TR[W_][k][:n] for k in range(NNET)]
    bs = np.array([float(np.dot(D[k], E[k]) / max(np.dot(D[k], D[k]), 1e-30))
                   for k in range(NNET)])
    loo = []
    for k in range(NNET):
        bk = np.median(np.delete(bs, k))
        loo.append(rel(E[k], T[k]) / max(rel(E[k] - bk * D[k], T[k]), 1e-30))
    A = list(range(0, NNET, 2)); B = list(range(1, NNET, 2))
    bA = np.median(bs[A])
    gB = np.mean([rel(E[k], T[k]) / max(rel(E[k] - bA * D[k], T[k]), 1e-30)
                  for k in B])
    # ЧУЖА: коефіцієнт свій, але Δcl від СУСІДНЬОЇ мережі
    gf = np.mean([rel(E[k], T[k]) /
                  max(rel(E[k] - np.median(bs) * D[(k + 1) % NNET], T[k]), 1e-30)
                  for k in range(NNET)])
    print(f"{n:>5d}{np.median(bs):>12.5f}{100*bs.std()/abs(np.median(bs)):>10.1f}%"
          f"{np.mean(loo):>13.4f}{gB:>11.4f}{gf:>10.4f}")
