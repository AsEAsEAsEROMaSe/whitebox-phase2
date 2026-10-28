"""gam101b — `γ₄` НЕ ЛИШЕ В `m`, А Й У ДІАГОНАЛЬ `C`. ЦЕ ГЕЙТ УЗГОДЖЕНОСТІ.

ЩО НОВОГО. Усі 101 сесія правили `γ₄` ВИКЛЮЧНО в першому моменті. Наш власний
записаний гейт каже: **поправка мусить чіпати діагональ І позадіагональ РАЗОМ**,
інакше мертво наперед. Ми його порушували — і міряли ×1.02.

ВИВЕДЕННЯ (не вгадано). `p = m + s·x`, щільність `x` з Еджвортом
`φ(x)[1 + (γ₄/24)He₄(x)]`. Використано `∫_a^∞ φ He_n = φ(a)He_{n−1}(a)` і
`x·He_n = He_{n+1} + n·He_{n−1}`:

    E[p·1{p>0}]  поправка = (γ₄/24)·φ·[m(3t−t³) + s(t⁴−2t²−1)]
                 при m = ts  ⟹  s·(γ₄/24)·(t²−1)·φ      🔒 = ЗАПИСАНА формула
    E[p²·1{p>0}] поправка = (γ₄/24)·φ·[m²(3t−t³) + 2ms(t⁴−2t²−1) + s²(t³−t⁵)]
                 при m = ts  ⟹  **−s²·(γ₄/12)·t·φ**     ← НОВЕ

🔒 ПЛОМБА ВИВЕДЕННЯ: та сама процедура для 1-го моменту відтворює записану
   формулу `(γ₄/24)(t²−1)φ` ТОЧНО ⟹ алгебра перевірена незалежним випадком.

МЕХАНІЗМ, ЗАРАДИ ЯКОГО ЦЕ РОБИТЬСЯ `[ЗМІРЯНО orac101e]`: `γ₄` на НАШИХ `(m,C)`
дає ×1.02, а ПОВЕРХ істинних — ×22.78. Поправляючи `C` тією ж `γ₄`, ми частково
ВМИКАЄМО САМІ СЕБЕ: точніша `C` → точніший наступний крок → точніша `C`.

РЕЖИМИ:
    m       γ₄ лише в m            (як gam101, ×1.0267 з половиною від чужої)
    d       γ₄ лише в діагональ C
    md      γ₄ в ОБИДВА, спільний коефіцієнт
    md2     γ₄ в обидва, ОКРЕМІ коефіцієнти (2-D скан)

🔒 КОНТРОЛЬ «ЧУЖА ТАБЛИЦЯ» (перевернута по глибині) для КОЖНОГО режиму — саме
   він завалив `gam101` (свій ×1.0267 проти чужого ×1.0129).
🔒 КРОС A/B · 🔒 LOO · 🔒 пломба бази (мусить дати 2.218914e-05 біт у біт).
🚨 УСЕ ЛИШЕ ПРИ `i > ksig` (опори λ-блоку годують МК-гілку = 91% балу).
🚨 І ВИРОК — ЛИШЕ ПОВНИМ ОЦІНЮВАЧЕМ: стенд чистого замикання вже збрехав на
   ×1.05 у цій самій сесії (`m_deep` іде в бленд ДВІЧІ).

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) `md` дасть більше за `m`                                          60%
 (б) 🔒 але контроль «чужа» знову з'їсть ≥половину                     55%
 (в) окремі коефіцієнти дадуть протилежні знаки                        50%
 (г) чистий виграш (свій − чужий) перевищить ×1.03 у замиканні         30%
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
KGRID = [round(0.992 + 0.002 * i, 4) for i in range(9)]
CGRID = [round(-0.6 + 0.1 * i, 2) for i in range(19)]

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
st = json.load(open(G4SRC))
G4T = np.median(np.array([st[k]["_g4"] for k in st]), axis=0)
print(f"gam101b · таблиця γ₄ з {len(st)} мереж · "
      f"γ₄(0,8,16,31) = {G4T[0]:.3f} {G4T[8]:.3f} {G4T[16]:.3f} {G4T[31]:.3f}",
      flush=True)

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
G4R = cp.asarray(G4T[::-1].copy())


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def run(kap, cm, cd, tab):
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
        if on and cm != 0.0:
            m = m + cm * s * (tab[i] / 24.0) * (t * t - 1.0) * ph
        if i == DEPTH - 1:
            break
        Co = (cpm * (Ph[:, :, None] * Ph[:, None, :])
              + 0.5 * (cpm * cpm) * ((ph / s)[:, :, None] * (ph / s)[:, None, :]))
        e2 = (mp * mp + s * s) * Ph + (mp * s) * ph
        if on and cd != 0.0:
            e2 = e2 - cd * (s * s) * (tab[i] / 12.0) * t * ph
        Co[:, idx, idx] = e2 - m * m
        C = Co
    return cp.sum(m * m, axis=1), cp.sum(m * T, axis=1)


def mse(mm, mt, sel):
    a = float(cp.sum(mt[sel]) / cp.maximum(cp.sum(mm[sel]), 1e-30))
    return float(cp.mean((a * a * mm[sel] - 2.0 * a * mt[sel] + TT[sel]) / W_))


ALL = cp.arange(NN)
A = ALL[::2]
B = ALL[1::2]


def scan(tab, sel, mode):
    best = None
    for k in KGRID:
        if mode == "m":
            pairs = [(c, 0.0) for c in CGRID]
        elif mode == "d":
            pairs = [(0.0, c) for c in CGRID]
        elif mode == "md":
            pairs = [(c, c) for c in CGRID]
        else:
            pairs = [(a1, b1) for a1 in CGRID[::2] for b1 in CGRID[::2]]
        for cm, cd in pairs:
            v = mse(*run(k, cm, cd, tab), sel)
            if math.isfinite(v) and (best is None or v < best[0]):
                best = (v, k, cm, cd)
    return best


def score(e2):
    return V_MC * e2 / (V_MC + e2)


v0 = mse(*run(0.9985, 0.0, 0.0, G4), ALL)
v0B = mse(*run(0.9985, 0.0, 0.0, G4), B)
print(f"🔒 ПЛОМБА БАЗИ: {v0:.6e} · похибка {100*math.sqrt(v0/MT2):.4f}%   "
      f"[gam101/scan101 дали 2.218914e-05]\n", flush=True)

print(f"{'режим':<6s}{'κ*':>7s}{'c_m':>7s}{'c_d':>7s}{'×замик.':>10s}"
      f"{'×БАЛ':>8s}{'🔒чужа':>9s}{'ЧИСТИЙ':>9s}{'крос A→B':>10s}")
print("=" * 74)
for mode in ("m", "d", "md", "md2"):
    v, k, cm, cd = scan(G4, ALL, mode)
    vr, kr, cmr, cdr = scan(G4R, ALL, mode)
    vA, kA, cmA, cdA = scan(G4, A, mode)
    vB = mse(*run(kA, cmA, cdA, G4), B)
    own, foreign = v0 / v, v0 / vr
    print(f"{mode:<6s}{k:>7.4f}{cm:>7.2f}{cd:>7.2f}{own:>10.4f}"
          f"{score(v0)/score(v):>8.4f}{foreign:>9.4f}{own/foreign:>9.4f}"
          f"{v0B/vB:>10.4f}", flush=True)
print("\nЧИСТИЙ = свій / чужий. Саме він вирішує (закон С88), а не величина.")
