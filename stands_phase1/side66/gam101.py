"""gam101 — ТАБЛИЧНИЙ `γ₄` У РЕКУРСІЇ. СЕМПЛИ НЕ ПОТРІБНІ, ЦІНА ~НУЛЬ.

`[ЗМІРЯНО orac101e, С101]`:
```
edg (істинні γ на НАШИХ m,C), вага 1.0   ×0.971  ⛔
edg, вага ФІТОВАНА 0.5                   ×1.027  ✅ (закон A4: фітувати!)
гейт збіжності |поправка|/|головний|     0.0005  ⟹ ряд НЕ розходиться
γ₃ медіана ≈ 0 · γ₄ систематична 0.18…0.37 і РОСТЕ з глибиною
```
⟹ працює ВИКЛЮЧНО `γ₄`, а він універсальний по мережах ⟹ вшивається ТАБЛИЦЕЮ
констант, і тоді семпли не потрібні ВЗАГАЛІ: рекурсія детермінована.

    m = mp·Φ(t) + s·φ(t) + c·s·(γ₄[i]/24)·(t²−1)·φ(t)      при i > ksig

🚨 УСЕ ЛИШЕ ПРИ `i > ksig` — опори λ-блоку (шари 1..8) годують control variate
   МК-гілки (91% балу). С100/С101 двічі горіли на цьому.
🚨 І ПАМ'ЯТАТИ ГОЛОВНИЙ УРОК С101: стенд ЧИСТОГО ЗАМИКАННЯ БРЕШЕ, бо `m_deep`
   іде в бленд ДВІЧІ (як передбачення І як опора фіту `_ai`). Вирок — ЛИШЕ
   повним оцінювачем, парно, `--format json`.

🔒 ПЛОМБА БАЗИ: при c=0 мусить збігтися з базою scan101 (2.218914e-05).
🔒 КРОС A/B: підбір `c` і `κ` на половині A, вирок на B.
🔒 LOO по мережах.
🔒 КОНТРОЛЬ «ЧУЖА ТАБЛИЦЯ»: γ₄ перевернутий по глибині (table[::-1]) — якщо
   він дає той самий виграш, то це ефект калібровки, а не інформації.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) табличний γ₄ дасть ×1.02…×1.06 у ЗАМИКАННІ                       55%
 (б) крос A→B збереже щонайменше половину                             60%
 (в) 🔒 перевернута таблиця дасть помітно менше                        65%
 (г) оптимальний `c` буде МЕНШИМ за 1.0 (поправка СТИСНЕНА)            75%
"""
import glob
import json
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
V_MC = 1.9227e-06
KSIG = 8
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
G4SRC = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
         "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/orac101e.json")
KGRID = [round(0.990 + 0.001 * i, 4) for i in range(21)]
CGRID = [round(-0.2 + 0.1 * i, 2) for i in range(23)]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)

st = json.load(open(G4SRC))
G4T = np.median(np.array([st[k]["_g4"] for k in st]), axis=0)
print(f"gam101 · таблиця γ₄ з {len(st)} мереж")
print("γ₄(шар) =", " ".join(f"{v:.3f}" for v in G4T[::4]), "…", flush=True)

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
     for L in range(DEPTH)]
T = cp.asarray(FM.astype(np.float64))
TT = cp.sum(T * T, axis=1)
MT2 = float(cp.mean(TT)) / W_
G4 = cp.asarray(G4T)
G4R = cp.asarray(G4T[::-1].copy())          # 🔒 контроль «чужа таблиця»


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def run(kap, c, tab):
    m = cp.zeros((NN, W_), dtype=cp.float64)
    C = cp.tile(cp.eye(W_, dtype=cp.float64), (NN, 1, 1))
    idx = cp.arange(W_)
    for i in range(DEPTH):
        w = W[i]
        cpm = cp.matmul(cp.transpose(w, (0, 2, 1)), cp.matmul(C, w))
        on = i > KSIG
        if on:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-20))
        mp = cp.squeeze(cp.matmul(m[:, None, :], w), 1)
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        if on and c != 0.0:
            m = m + c * s * (tab[i] / 24.0) * (t * t - 1.0) * ph
        if i == DEPTH - 1:
            break
        Co = (cpm * (Ph[:, :, None] * Ph[:, None, :])
              + 0.5 * (cpm * cpm) * ((ph / s)[:, :, None] * (ph / s)[:, None, :]))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        Co[:, idx, idx] = d
        C = Co
    return cp.sum(m * m, axis=1), cp.sum(m * T, axis=1)


def mse(mm, mt, sel):
    a = float(cp.sum(mt[sel]) / cp.maximum(cp.sum(mm[sel]), 1e-30))
    return float(cp.mean((a * a * mm[sel] - 2.0 * a * mt[sel] + TT[sel]) / W_)), a


ALL = cp.arange(NN)
A = ALL[::2]
B = ALL[1::2]


def scan(tab, sel):
    best = None
    for k in KGRID:
        for c in CGRID:
            mm, mt = run(k, c, tab)
            v, a = mse(mm, mt, sel)
            if math.isfinite(v) and (best is None or v < best[0]):
                best = (v, k, c, a, mm, mt)
    return best


def score(e2):
    return V_MC * e2 / (V_MC + e2)


mm0, mt0 = run(0.9985, 0.0, G4)
v0, a0 = mse(mm0, mt0, ALL)
print(f"\n🔒 ПЛОМБА БАЗИ (c=0, κ=0.9985): MSE {v0:.6e} · "
      f"похибка {100*math.sqrt(v0/MT2):.4f}%   [scan101 дав 2.218914e-05]",
      flush=True)

for nm, tab in (("γ₄ ТАБЛИЧНИЙ", G4), ("🔒 ПЕРЕВЕРНУТИЙ (контроль)", G4R)):
    v, k, c, a, mm, mt = scan(tab, ALL)
    vA, kA, cA, _, _, _ = scan(tab, A)
    mmB, mtB = run(kA, cA, tab)
    vB, _ = mse(mmB, mtB, B)
    v0B, _ = mse(mm0, mt0, B)
    wins = 0
    for i in range(NN):
        s2 = cp.asarray([j for j in range(NN) if j != i])
        if mse(mm, mt, s2)[0] < mse(mm0, mt0, s2)[0]:
            wins += 1
    print(f"\n── {nm} ──")
    print(f"   оптимум κ={k}  c={c}  α*={a:.5f}")
    print(f"   MSE {v:.6e} · похибка {100*math.sqrt(v/MT2):.4f}% · "
          f"×замикання {v0/v:.4f} · ×БАЛ {score(v0)/score(v):.4f}")
    print(f"   🔒 крос A→B: κ={kA} c={cA} ⟹ ×замикання {v0B/vB:.4f}")
    print(f"   🔒 LOO: {wins}/{NN} фолдів кращі за базу", flush=True)
