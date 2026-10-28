"""depth109 — ПОШАРОВИЙ РОЗКЛАД СТЕЛІ КРОКУ. Єдиний рядок, що міг перевернути ціну.

`step105` вкинув ТОЧНУ похибку кроку на ВСІХ шарах одразу і дістав `×1.19`.
Пошаровий розклад цієї стелі ніколи не мірявся. Питання прямо цінове: чутливість
до ваг росте з глибиною у `×52.9` (`rd109b`), і якщо стеля кроку розподілена так
само, її більшу частину можна взяти в останніх 3–4 шарах за десяту частину ціни.

    d_l = E_гаус[ReLU | ІСТИННІ (m_l, C_l)] − E_МК[ReLU(p_l)]
    вкидання: ma_l ← ma_l − c·d_l  для l у смузі, решта рекурсії без змін

`d_l` — це похибка формули кроку при ТОЧНИХ входах, тобто верхня межа для будь-якої
ідеї «точніша формула кроку» на цьому шарі.

🔒 ПЛОМБИ
 T1 ПОВНА СМУГА [0,31] мусить відтворити порядок `×1.19` зі `step105`
    (наша база інша: тут просте замикання без κ,gam ⟹ точного збігу не чекаємо)
 T2 ПОРОЖНЯ СМУГА мусить дати РІВНО базу
 T3 ЧУЖА `d` (від іншої мережі) мусить мати оптимум при `c → 0`
 T4 ціль — `final_means` датасету (N = 1e9), НЕЗАЛЕЖНА від наших семплів
    ⟹ шум може лише ЗАНИЖУВАТИ виграш, не завищувати

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ
    останні 4 шари дають ≥50% усього виграшу смуги [0,31]        40%
    один лише шар 31 дає ≥30%                                    35%
    ⟹ якщо обидва хибні, ціновий аргумент остаточно закритий
"""
import glob
import json
import math
import os
import sys

sys.path.insert(0, "/home/trim/gpupkg")
os.environ.setdefault("CUPY_GPU_MEMORY_LIMIT", "1610612736")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402

DEPTH, W_ = 32, 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NSAMP = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
CHUNK = 8192
CS = tuple(np.round(np.arange(0.0, 1.31, 0.05), 3))
BANDS = [("порожня", 99, 99), ("[31,31]", 31, 31), ("[30,31]", 30, 31),
         ("[28,31]", 28, 31), ("[24,31]", 24, 31), ("[16,31]", 16, 31),
         ("[8,31]", 8, 31), ("[0,31]", 0, 31),
         ("[0,0]", 0, 0), ("[0,7]", 0, 7), ("[0,15]", 0, 15),
         ("[0,23]", 0, 23)]
OUT = "/home/trim/wblogs/depth109.json"

import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 ВИДНО НЕ ОДНУ КАРТУ — вихід.", flush=True)
    sys.exit(2)
free, tot = cp.cuda.runtime.memGetInfo()
print(f"depth109 · {NNET} мереж · N={NSAMP} · вільно {free/2**20:.0f} МБ",
      flush=True)

Wall, TRUE = [], []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    xx = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xx.type):
        xx = xx.flatten()
    Wall.append(np.asarray(xx).reshape(tb.num_rows, DEPTH, W_, W_))
    yy = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(yy.type):
        yy = yy.flatten()
    TRUE.append(np.asarray(yy).reshape(tb.num_rows, W_))
Wall = np.concatenate(Wall, 0)
TRUE = np.concatenate(TRUE, 0)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def step_mean(m, C):
    s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
    t = m / s
    return s * (phi(t) + t * ndtr(t))


def step_full(m, C):
    s = cp.sqrt(cp.maximum(cp.diag(C), 1e-300))
    t = m / s
    Ph, ph = ndtr(t), phi(t)
    ma = s * (ph + t * Ph)
    q = s * s * ((1.0 + t * t) * Ph + t * ph)
    rho = C / cp.maximum(cp.outer(s, s), 1e-300)
    Ca = cp.outer(Ph, Ph) * C + 0.5 * cp.outer(ph, ph) * rho * rho \
        * cp.outer(s, s)
    cp.fill_diagonal(Ca, q - ma * ma)
    return ma, Ca


def cos_err(pred, T):
    c = float(cp.dot(pred, T)) / math.sqrt(
        float(cp.dot(pred, pred)) * float(cp.dot(T, T)))
    return math.sqrt(max(0.0, 1.0 - c * c))


def run(W, D, lo, hi, c, T):
    m = cp.zeros(W_, dtype=cp.float64)
    C = (W[0].T @ W[0])
    for i in range(DEPTH):
        ma, Ca = step_full(m, C)
        if lo <= i <= hi:
            ma = ma - c * D[i]
        if i == DEPTH - 1:
            return cos_err(ma, T)
        m = W[i + 1].T @ ma
        C = W[i + 1].T @ Ca @ W[i + 1]


rows = []
for net in range(NNET):
    W = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float64))
         for L in range(DEPTH)]
    W32 = [w.astype(cp.float32) for w in W]
    S1 = [cp.zeros(W_, dtype=cp.float64) for _ in range(DEPTH)]
    S2 = [cp.zeros((W_, W_), dtype=cp.float64) for _ in range(DEPTH)]
    SA = [cp.zeros(W_, dtype=cp.float64) for _ in range(DEPTH)]
    rng = cp.random.default_rng(7700 + net)
    done = 0
    while done < NSAMP:
        nb = min(CHUNK, NSAMP - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], 0)          # антитетика: E[z]=0 точно
        for i in range(DEPTH):
            p = a @ W32[i]
            p64 = p.astype(cp.float64)
            S1[i] += p64.sum(0)
            S2[i] += p64.T @ p64
            a = cp.maximum(p, 0.0)
            SA[i] += a.astype(cp.float64).sum(0)
        done += nb
        del z, a, p, p64
    N2 = 2.0 * NSAMP
    Dl = []
    for i in range(DEPTH):
        mi = S1[i] / N2
        Ci = S2[i] / N2 - cp.outer(mi, mi)
        Dl.append(step_mean(mi, Ci) - SA[i] / N2)
    T = cp.asarray(TRUE[net], dtype=cp.float64)
    r = {}
    for nm, lo, hi in BANDS:
        best = min(((run(W, Dl, lo, hi, float(c), T), float(c)) for c in CS))
        r[nm] = best
    # T3 ЧУЖА d
    if net > 0:
        r["чужа [0,31]"] = min(((run(W, prevD, 0, 31, float(c), T), float(c))
                                for c in CS))
    prevD = Dl
    rows.append(r)
    b0 = r["порожня"][0]
    print(f"  ✅ {net}: база {100*b0:.4f}% · [31,31] ×{b0/r['[31,31]'][0]:.4f}"
          f" · [28,31] ×{b0/r['[28,31]'][0]:.4f}"
          f" · [0,31] ×{b0/r['[0,31]'][0]:.4f} (c*={r['[0,31]'][1]:.2f})",
          flush=True)
    del W, W32, S1, S2, SA
    cp.get_default_memory_pool().free_all_blocks()

json.dump({k: [[r[k][0], r[k][1]] for r in rows if k in r]
           for k in rows[0]} | {"foreign": [r.get("чужа [0,31]", [0, 0])
                                            for r in rows[1:]]},
          open(OUT, "w"))
base = np.array([r["порожня"][0] for r in rows])
print(f"\n база (просте замикання, косинусна) {100*base.mean():.4f}%"
      f"   [корпус: 0.507% з κ,gam · 0.530% без]")
print("\n смуга        ×точності    c*      частка виграшу [0,31]")
full = np.array([r["[0,31]"][0] for r in rows])
gfull = (base / full).mean() - 1.0
for nm, _, _ in BANDS:
    if nm == "порожня":
        continue
    v = np.array([r[nm][0] for r in rows])
    g = (base / v).mean() - 1.0
    cs = np.median([r[nm][1] for r in rows])
    print(f" {nm:<12} ×{(base/v).mean():.4f}   {cs:.2f}      "
          f"{100*g/max(gfull,1e-12):6.1f}%")
if len(rows) > 1:
    fo = np.array([r["чужа [0,31]"][0] for r in rows[1:]])
    fc = np.median([r["чужа [0,31]"][1] for r in rows[1:]])
    print(f"\n🔒 T3 ЧУЖА d: ×{(base[1:]/fo).mean():.4f}, медіана c* = {fc:.2f}"
          f"  (мусить бути c*→0)")
print(f"🔒 T2 порожня смуга = база за побудовою")
print(f"🔒 T1 повна смуга ×{(base/full).mean():.4f}  [step105 дав ×1.19]")
