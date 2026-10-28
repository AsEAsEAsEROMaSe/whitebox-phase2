"""orac101e — ЧИ ДАЄ 4-МОМЕНТНИЙ ОРАКУЛ ТОП-3? ПРЯМИЙ ВИМІР ВЛАСНИМ ПРИЛАДОМ.

СУПЕРЕЧНІСТЬ У ЗАПИСАХ (С99), яку треба зняти:
  «повний оракул (m,C,γ₃,γ₄) = 1.48e-08 проти лідера 3.63e-09 ⟹ ×4.1 НАД НИМ»
     якщо 1.48e-08 — MSE ⟹ бал 1.47e-09 ⟹ **#4, майже #3**
  «оракульні γ на НАШИХ (m,C) = ×1.002 · разом ×7.75» ⟹ бал 2.4e-08 ⟹ ~#5
Різниця між прочитаннями **×16**, і вона вирішує, чи 4-моментний клас веде до
призу взагалі. Міряємо самі — закон «відкат лише на власний вимір».

РЕЖИМИ (усі — з ВЛАСНИМ оптимальним α і κ):
    base   наша рекурсія
    allm   істинні m,C щокроку          ⟹ стеля ДВОМОМЕНТНОГО класу (0.0913%)
    edg    Еджворт з ІСТИННИМИ γ₃,γ₄ на НАШИХ (m,C)
    edgm   Еджворт з істинними γ₃,γ₄ ПОВЕРХ істинних (m,C)  ⟹ стеля 4-МОМЕНТНОГО

    E[ReLU]/s = ψ(t) − (γ₃/6)·t·φ + (γ₄/24)·(t²−1)·φ ,  ψ = φ + tΦ
    γ рахуються для ПЕРЕДАКТИВАЦІЙ `p_l` (не активацій!) — саме до них
    застосовується формула кроку.

🔒 ПЛОМБА: `base` мусить відтворити 0.507% (грейдерне чисте замикання).
🔒 ПЛОМБА: `allm` мусить дати 0.09% (вже зміряно двома репліками).
🔒 ГЕЙТ ЗБІЖНОСТІ (записаний закон): рахуємо |поправка|/|головний член| у
   робочій точці й друкуємо — якщо >1, ряд розходиться і числу вірити не можна.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) `edg` (γ на наших m,C) дасть ×1.00…×1.05 — тобто майже нуль    70%
 (б) `edgm` покращить `allm` менш ніж удвічі                        60%
 (в) ⟹ прочитання «×7.75» правильне, «#4» — хибне                   65%
 (г) гейт збіжності покаже |поправка|/|головний| > 0.3 на глибині    50%
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, "/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
                   "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
V_MC = 1.9227e-06
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NPAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 524288
CHUNK = 16384
KSIG = 8
KGRID = sorted({round(0.980 + 0.004 * i, 4) for i in range(11)} | {0.9985, 1.0})
MODES = ["base", "allm", "edg", "edgm"]
CGRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, -0.3, -0.5]
OUT = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/orac101e.json")

if cp.cuda.runtime.getDeviceCount() != 1:
    sys.exit(2)
state = {}
if os.path.exists(OUT):
    try:
        state = json.load(open(OUT))
    except Exception:                                         # noqa: BLE001
        state = {}


def save():
    with open(OUT + ".tmp", "w") as fh:
        json.dump(state, fh)
    os.replace(OUT + ".tmp", OUT)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
xx = tb.column("weights").combine_chunks()
while not pa.types.is_primitive(xx.type):
    xx = xx.flatten()
Wall = np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_)
yy = tb.column("final_means").combine_chunks()
while not pa.types.is_primitive(yy.type):
    yy = yy.flatten()
FM = np.asarray(yy).reshape(tb.num_rows, W_)
NNET = min(NNET, tb.num_rows)
N = 2 * NPAIRS
print(f"orac101e · {NNET} мереж × {N} семплів · {len(KGRID)} точок κ", flush=True)


def closure(W, kap, Ct, mt, G3, G4, mode, cg=1.0):
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    gate = 0.0
    for i in range(DEPTH):
        w = W[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        base = mp * Ph + s * ph
        if mode in ("edg", "edgm"):
            corr = cg * (-s * (G3[i] / 6.0) * t * ph
                         + s * (G4[i] / 24.0) * (t * t - 1.0) * ph)
            m = base + corr
            if i == DEPTH - 1:
                gate = float(cp.abs(corr).mean()
                             / cp.maximum(cp.abs(base).mean(), 1e-300))
        else:
            m = base
        if i == DEPTH - 1:
            break
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        Co = Co + cp.diag(d - cp.diagonal(Co))
        if mode in ("allm", "edgm"):
            C = Ct[i]
            m = mt[i]
        else:
            C = Co
    return m, gate


def one_net(net):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
          for L in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    T = cp.asarray(FM[net].astype(np.float64))
    rng = cp.random.default_rng(777 + net)
    A1 = cp.zeros((DEPTH - 1, W_), dtype=cp.float64)       # активації (для C)
    A2 = cp.zeros((DEPTH - 1, W_, W_), dtype=cp.float64)
    P1 = cp.zeros((DEPTH, W_), dtype=cp.float64)           # ПЕРЕДактивації
    P2 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P3 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P4 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            p = a @ Wg[i]
            p64 = p.astype(cp.float64)
            P1[i] += p64.sum(0)
            q = p64 * p64
            P2[i] += q.sum(0)
            P3[i] += (q * p64).sum(0)
            P4[i] += (q * q).sum(0)
            if i < DEPTH - 1:
                a = cp.maximum(p, 0.0)
                A1[i] += a.sum(0, dtype=cp.float64)
                A2[i] += (a.T @ a).astype(cp.float64)
        done += nb
    del z, a, p, p64, q
    cp.get_default_memory_pool().free_all_blocks()
    mt = A1 / N
    Ct = A2 / N - mt[:, :, None] * mt[:, None, :]
    Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
    mu = P1 / N
    v = cp.maximum(P2 / N - mu * mu, 1e-30)
    sd = cp.sqrt(v)
    m3 = P3 / N - 3.0 * mu * (P2 / N) + 2.0 * mu ** 3
    m4 = (P4 / N - 4.0 * mu * (P3 / N) + 6.0 * mu * mu * (P2 / N)
          - 3.0 * mu ** 4)
    G3 = m3 / sd ** 3
    G4 = m4 / sd ** 4 - 3.0
    out = {"_TT": float(T @ T),
           "_g3": [float(cp.median(G3[i])) for i in (0, 8, 16, 31)],
           "_g4": [float(cp.median(G4[i])) for i in range(DEPTH)]}
    for mode in MODES:
        tab = {}
        cs = CGRID if mode in ("edg", "edgm") else [1.0]
        for k in KGRID:
            for cg in cs:
                mm, gate = closure(W64, k, Ct, mt, G3, G4, mode, cg)
                if bool(cp.all(cp.isfinite(mm))):
                    tab[f"{k}|{cg}"] = [float(mm @ mm), float(mm @ T)]
                    out["_gate_" + mode] = gate
        out[mode] = tab
    del A1, A2, Ct, mt
    cp.get_default_memory_pool().free_all_blocks()
    return out


for net in range(NNET):
    if str(net) in state:
        continue
    t0 = time.time()
    state[str(net)] = one_net(net)
    save()
    print(f"  ✅ мережа {net} ({time.time()-t0:.1f} с) · γ₃(мед) "
          f"{state[str(net)]['_g3']} · гейт edg "
          f"{state[str(net)].get('_gate_edg', 0):.4f}", flush=True)

nets = sorted(state, key=int)
TT = {n: state[n]["_TT"] for n in nets}
MT2 = sum(TT.values()) / len(nets) / W_


def best(mode):
    out = []
    for k in state[nets[0]][mode]:
        num = sum(state[n][mode][k][1] for n in nets)
        den = sum(state[n][mode][k][0] for n in nets)
        if den <= 0:
            continue
        a = num / den
        tot = sum((a * a * state[n][mode][k][0] - 2 * a * state[n][mode][k][1]
                   + TT[n]) / W_ for n in nets)
        out.append((tot / len(nets), k, a))
    return min(out)


def score(e2):
    return V_MC * e2 / (V_MC + e2)


print(f"\n{'режим':<7s}{'κ*':>8s}{'MSE замик.':>14s}{'похибка':>10s}"
      f"{'×замик.':>9s}{'×БАЛ':>8s}{'бал':>12s}{'гейт':>8s}")
print("=" * 78)
b0 = best("base")[0]
for mode in MODES:
    v, k, a = best(mode)
    g = np.mean([state[n].get("_gate_" + mode, 0.0) for n in nets])
    bal = 1.859e-07 * score(v) / score(b0)
    print(f"{mode:<7s}{k:>8s}{v:>14.6e}{100*math.sqrt(v/MT2):>9.4f}%"
          f"{b0/v:>9.4f}{score(b0)/score(v):>8.4f}{bal:>12.3e}{g:>8.4f}")
print("\nборд: #1 4.00e-10 · #2 9.00e-10 · #3 1.00e-9 · #4 4.62e-8 · #5 5.81e-8")
print(f"γ₃ медіана по шарах (0,8,16,31): "
      f"{np.mean([state[n]['_g3'] for n in nets], axis=0).round(4)}")
print(f"γ₄ медіана по шарах (0,8,16,31): "
      f"{np.mean([state[n]['_g4'] for n in nets], axis=0).round(4)}")
