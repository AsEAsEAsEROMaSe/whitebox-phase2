"""scan101 — СІМ'Я ГЛОБАЛЬНИХ РУЧОК У РЕКУРСІЇ. Ціна нуль, семпли не потрібні.

МОТИВ [ЗМІРЯНО orac101, 48 мереж]: `allm` ×3.36 · `off` ×1.01 ⟹ увесь важіль у
`m`, канал `C` порожній. А в `m` 2/3 похибки створює ПЕРЕНОС, не крок (С74).
`κ` (С100) — перша ручка цього типу, дала рекорд. Тут перебирається вся сім'я.

РУЧКИ (усі застосовуються ЛИШЕ при `i > KSIG`):
    κ   cpm *= κ                       множник на передактиваційну коваріацію
    μ   mp  = (m @ w) * μ              ПРЯМА атака на перенос m
    λ   m   = mp*Φ + λ * s*φ           множник на φ-член (міняє форму, не масштаб)
    β   C   = cpm*ΦΦ + β*0.5*cpm²*…    множник на Мелерів член
    τ   t   = (mp/s) * τ               множник на аргумент воріт

🚨 ЧОМУ ЛИШЕ `i > KSIG`: `_closure_parts` сабміту віддає опори s,m,q,P,a,b з
   шарів 1..8 у control variate МК-гілки (91% балу). Зсунувши їх, С100 дістала
   ×0.57. Стенд, що міряє ОДНУ гілку, цього НЕ БАЧИТЬ ⟹ обмеження вшите тут.

🔒 ПЛОМБА БАЗИ: при всіх ручках = 1.0 і κ=0.9985 стенд мусить відтворити
   записане чисте замикання 0.5070% (грейдер, `_W_BLEND=1.0`, 50 мереж).
   orac101 уже дав 0.5072% — розбіжність 0.04%.
🔒 КРОС A/B: усе, що не переноситься з половини на половину, ВІДКИДАЄТЬСЯ.
   Один скаляр на 50 мереж уже показував перефіт (С98: A ×1.036 → B ×0.9969).
🔒 LOO по мережах для фінальної конфігурації.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) `μ` (перенос) виявиться найсильнішою одиночною ручкою після κ    40%
 (б) сумарний виграш сім'ї у замиканні ×1.05…×1.20                    50%
 (в) у БАЛІ це дасть ×1.01…×1.04                                      55%
 (г) щонайменше одна ручка НЕ перенесеться на крос A/B                 70%
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
KSIG = 8
V_MC = 1.9227e-06
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NAMES = ["kap", "mu", "lam", "beta", "tau"]
BASE = {"kap": 0.9980, "mu": 1.0, "lam": 0.9950, "beta": 0.8, "tau": 1.02}

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не одну карту — пінування не спрацювало", flush=True)
    sys.exit(2)

f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
NN = tb.num_rows
Wall = np.asarray(xx).reshape(NN, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(NN, W_)

W = [cp.asarray(np.ascontiguousarray(Wall[:, L], dtype=np.float64))
     for L in range(DEPTH)]                      # DEPTH × (NN,256,256)
T = cp.asarray(FM.astype(np.float64))            # (NN,256)
TT = cp.sum(T * T, axis=1)                       # (NN,)
MT2 = float(cp.mean(TT)) / W_
print(f"scan101 · {NN} мереж · батчева рекурсія на GPU · mean(T²)={MT2:.6f}",
      flush=True)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def run(p):
    """батчева рекурсія по всіх мережах; повертає (Σm², ΣmT) на мережу"""
    m = cp.zeros((NN, W_), dtype=cp.float64)
    C = cp.tile(cp.eye(W_, dtype=cp.float64), (NN, 1, 1))
    for i in range(DEPTH):
        w = W[i]
        cpm = cp.matmul(cp.transpose(w, (0, 2, 1)), cp.matmul(C, w))
        on = i > KSIG
        if on:
            cpm = cpm * p["kap"]
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-20))
        mp = cp.squeeze(cp.matmul(m[:, None, :], w), 1)
        if on:
            mp = mp * p["mu"]
        t = mp / s
        if on:
            t = t * p["tau"]
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + (p["lam"] if on else 1.0) * s * ph
        if i == DEPTH - 1:
            break
        bm = p["beta"] if on else 1.0
        Co = (cpm * (Ph[:, :, None] * Ph[:, None, :])
              + (bm * 0.5) * (cpm * cpm)
              * ((ph / s)[:, :, None] * (ph / s)[:, None, :]))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        idx = cp.arange(W_)
        Co[:, idx, idx] = d
        C = Co
    return cp.sum(m * m, axis=1), cp.sum(m * T, axis=1)


def mse_of(mm, mt, sel):
    """агрегатна MSE при глобальному оптимальному α на підмножині `sel`"""
    a = float(cp.sum(mt[sel]) / cp.maximum(cp.sum(mm[sel]), 1e-30))
    v = (a * a * mm[sel] - 2.0 * a * mt[sel] + TT[sel]) / W_
    return float(cp.mean(v)), a


ALL = cp.arange(NN)
A = ALL[::2]
B = ALL[1::2]


def ev(p, sel=ALL):
    mm, mt = run(p)
    return mse_of(mm, mt, sel)[0]


def score(e2):
    return V_MC * e2 / (V_MC + e2)


base = dict(BASE)
v0 = ev(base)
print(f"\n🔒 ПЛОМБА БАЗИ: MSE {v0:.6e} · похибка {100*math.sqrt(v0/MT2):.4f}%"
      f"   [записане чисте замикання 0.5070%]", flush=True)

GRID = {
    "kap":  [round(0.990 + 0.0005 * i, 5) for i in range(25)],
    "mu":   [round(0.990 + 0.0010 * i, 5) for i in range(21)],
    "lam":  [round(0.900 + 0.0100 * i, 5) for i in range(31)],
    "beta": [round(0.000 + 0.0500 * i, 5) for i in range(29)],
    "tau":  [round(0.960 + 0.0050 * i, 5) for i in range(41)],
}

singles = {}
print("\n── КООРДИНАТНИЙ СПУСК (усі разом, 4 проходи) ──", flush=True)
cur = dict(base)
vcur = v0
for it in range(3):
    for nm in NAMES:
        best = (vcur, cur[nm])
        for g in GRID[nm]:
            p = dict(cur)
            p[nm] = g
            v = ev(p)
            if v < best[0]:
                best = (v, g)
        cur[nm], vcur = best[1], best[0]
    print(f"  прохід {it+1}: " + " ".join(f"{k}={cur[k]:.4f}" for k in NAMES)
          + f"  MSE {vcur:.6e}  похибка {100*math.sqrt(vcur/MT2):.4f}%"
            f"  ×{v0/vcur:.4f}", flush=True)

print(f"\nПІДСУМОК спільного оптимуму:  ×замикання {v0/vcur:.4f}  ·  "
      f"×БАЛ {score(v0)/score(vcur):.4f}", flush=True)

print("\n🔒 КРОС A/B ДЛЯ СПІЛЬНОГО ОПТИМУМУ (підбір на A, вирок на B):", flush=True)
cA, vA = dict(base), ev(base, A)
for it in range(3):
    for nm in NAMES:
        best = (vA, cA[nm])
        for g in GRID[nm]:
            p = dict(cA)
            p[nm] = g
            v = ev(p, A)
            if v < best[0]:
                best = (v, g)
        cA[nm], vA = best[1], best[0]
print("   κ(A) = " + " ".join(f"{k}={cA[k]:.4f}" for k in NAMES), flush=True)
vB0, vB = ev(base, B), ev(cA, B)
print(f"   на B: MSE {vB:.6e}  ×замикання {vB0/vB:.4f}  ·  "
      f"×БАЛ {score(vB0)/score(vB):.4f}", flush=True)
print(f"   (для довідки, власний оптимум B був би кращим — це верхня оцінка "
      f"перефіту)", flush=True)

print("\n🔒 LOO по мережах для спільного оптимуму:", flush=True)
mm0, mt0 = run(base)
mmc, mtc = run(cur)
wins = 0
for i in range(NN):
    sel = cp.asarray([j for j in range(NN) if j != i])
    v0i, _ = mse_of(mm0, mt0, sel)
    vci, _ = mse_of(mmc, mtc, sel)
    if vci < v0i:
        wins += 1
print(f"   фолдів, де спільний оптимум кращий за базу: {wins}/{NN}", flush=True)
