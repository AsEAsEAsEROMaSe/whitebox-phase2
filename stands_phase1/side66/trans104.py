"""trans104 — ПОШУК ПОПРАВКИ ДО *ПЕРЕНОСУ*. ПЕРШИЙ ПРОГІН ПІД НОВОЮ ЦІЛЛЮ.

ЗМІНА ЦІЛЬОВОЇ ФУНКЦІЇ (рішення ЮЗЕРА, С104): міряємо НЕ бал, а КОСИНУСНУ
ПОХИБКУ ЗАМИКАННЯ. Підстава — важіль точності НЕ сталий:

    точність ×1.5 → бал ×1.10   (важіль 0.22)
    точність ×4   → бал ×2.14   (важіль 0.55)
    точність ×8   → бал ×5.79   (важіль 0.85)   ← ~#3-4 місце
    точність ×15  → бал ×18.0   (важіль 1.07)

⟹ поки МК тримає бал, малі покращення точності ЗАДАВЛЕНІ вп'ятеро, і сто
сесій ми ховали ідеї за числом, яке до них несправедливе. Ціна НЕ гейт:
все замикання = 1.66e9 = 6.1% підлоги, запас ×16.

ЩО САМЕ ШУКАЄМО І ЧОМУ САМЕ ЦЕ:
   С74: 2/3 похибки `m` створює ПЕРЕНОС, а не крок; між «ідеальна формула
        кроку» (×1.111) і «точне m» (×3.109) НЕМАЄ НІЧОГО ПРОМІЖНОГО.
   Клас «кращий КРОК» закрито з чотирьох боків (квадратура ×1.001 · лінійна
        корекція ×1.027 · GP ×1.002 · ранг-r ×1.04).
   ⟹ шукаємо поправку до ПЕРЕНОСУ — тобто до того, що накопичується.

🔑 ЧОМУ СТЕЛЯ ×1.027 (С101-г) СЮДИ НЕ ЗАСТОСОВНА: там фіт корегував ВИХІД
   (один раз, по мережі). Тут поправка діє НА КОЖНОМУ КРОЦІ з ГЛОБАЛЬНИМ
   коефіцієнтом і накопичується — рівно той тип, до якого належать `κ` (×1.055)
   і `gam`. Обидва живі; питання — чи є в цьому класі СТРУКТУРНІ елементи,
   а не лише скаляри.

⛔ ЧОГО НЕ ШУКАЄМО: скалярів на перенос. `mu` дав оптимум РІВНО 1.0 (scan101),
   бо перенос псує `m` ФОРМОЮ, а масштаб `_ALPHA` бере даром. Тому всі шість
   сімей нижче міняють ФОРМУ вектора, і метрика КОСИНУСНА — масштаб винесено.

СІМ'Ї (усі щошару при i>KSIG, усі однопараметричні):
   A  mp += c·(Cp@mp)/mean(diag Cp)        взаємодія нейронів через C
   B  m  += c·s·φ(u)·(u²−1)                форма 4-го порядку, фіксована
   C  s  *= 1 + c·(s/⟨s⟩ − 1)              розтяг спектра σ
   D  mp *= 1 + c·(s/⟨s⟩ − 1)              зв'язок mp зі своїм s
   E  Cp += c·outer(mp,mp)·⟨diag Cp⟩/⟨mp²⟩ підживлення C від m
   F  m  *= 1 + c·(u−⟨u⟩)/⟨u⟩              режимна залежність

🔒 ПЛОМБИ: (1) c=0 у КОЖНІЙ сім'ї мусить дати ОДНЕ Й ТЕ САМЕ число (база)
           (2) база мусить лягти в 0.48…0.52% (грейдер 0.5070%, kap102c 0.4914%)
           (3) вирок ЛИШЕ по кросу A→B і B→A, ніколи in-sample
           (4) κ перепідбирається РАЗОМ із c (закон С100: константи не
               переносяться між конфігураціями)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
   - жодна сім'я не дасть крос >×1.05:                        60%
   - хоч одна дасть крос ×1.05…×1.20:                         30%
   - хоч одна дасть крос >×1.20:                              10%
   - оптимум щонайменше двох сімей ляже РІВНО в c=0:          70%
"""
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_ = 32, 256
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 64
PACK = 16
KSIG = 8
GGRID = [1.030, 1.040, 1.048, 1.056, 1.066]
KGRID = [0.9970, 0.9985, 1.0000]
# 🚨 ПЕРШИЙ ПРОГІН МАВ СІТКУ НА ДВА ПОРЯДКИ ЗАШИРОКУ: при |c|=0.02 похибка вже
#    росла у 1.2…7.4×, тобто мінімум ГОСТРИЙ. Крила симетричні ⟹ оптимум біля
#    нуля, але параболічна оцінка з трьох точок дає c* = 0.0004…0.004 і різні
#    масштаби для різних сімей. Тому сітка тепер ПЕР-СІМЕЙНА, з кроком ~c*/4.
SCALE = {"A": 0.0010, "B": 0.0060, "C": 0.0050,
         "D": 0.0015, "E": 0.0090, "F": 0.0005}
STEPS = [-1.0, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.0]
FAMS = ["A", "B", "C", "D", "E", "F"]


def cgrid(fam):
    return [round(SCALE[fam] * s, 8) for s in STEPS]
OUT = "/home/trim/wblogs/trans104.json"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту — пінування зламане")
    sys.exit(2)
state = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    with open(OUT + ".tmp", "w") as fh:
        json.dump(state, fh)
    os.replace(OUT + ".tmp", OUT)


def stream(nmax):
    """Читаємо БАТЧАМИ з mmap — ніякого read_all/combine_chunks на всю
    колонку (саме воно тримало 17 таблиць живими і поклало машину в С103)."""
    got = 0
    for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= nmax:
            break
        rd = pa.ipc.open_stream(pa.memory_map(fp, "rb"))
        for b in rd:
            if got >= nmax:
                break
            xw = b.column("weights")
            while not pa.types.is_primitive(xw.type):
                xw = xw.flatten()
            xt = b.column("final_means")
            while not pa.types.is_primitive(xt.type):
                xt = xt.flatten()
            nb = b.num_rows
            wv = np.asarray(xw).reshape(nb, DEPTH, W_, W_)
            tv = np.asarray(xt).reshape(nb, W_)
            n = min(nb, nmax - got)
            for j in range(n):
                yield np.array(wv[j], dtype=np.float32), \
                    np.array(tv[j], dtype=np.float64)
            got += n
            del xw, xt, wv, tv
        del rd


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def run(Wb, kap, gam, fam, c):
    """Батчева рекурсія: Wb (P,32,256,256) fp32.  C у fp32 (матмули),
    m у fp64 (акумуляція).  Повертає m (P,256) fp64."""
    P = Wb.shape[0]
    m = cp.zeros((P, W_), dtype=cp.float64)
    C = cp.empty((P, W_, W_), dtype=cp.float32)
    C[:] = cp.eye(W_, dtype=cp.float32)
    idx = cp.arange(W_)
    for i in range(DEPTH):
        w = Wb[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w)
        on = i > KSIG
        if on and kap != 1.0:
            cpm *= kap
        mp = cp.einsum('bi,bij->bj', m, w.astype(cp.float64))
        if on and fam == "E" and c != 0.0:
            sc = cp.mean(cpm[:, idx, idx], axis=1) / cp.maximum(
                cp.mean(mp * mp, axis=1), 1e-30)
            cpm += (c * sc)[:, None, None] * (mp[:, :, None]
                                              * mp[:, None, :]).astype(cp.float32)
        s2 = cp.maximum(cpm[:, idx, idx].astype(cp.float64), 1e-30)
        s = cp.sqrt(s2)
        if on and fam == "C" and c != 0.0:
            s = s * (1.0 + c * (s / cp.mean(s, axis=1, keepdims=True) - 1.0))
            s = cp.maximum(s, 1e-30)
            s2 = s * s                       # діагональ мусить лишитись s²
        if on and fam == "D" and c != 0.0:
            mp = mp * (1.0 + c * (s / cp.mean(s, axis=1, keepdims=True) - 1.0))
        if on and fam == "A" and c != 0.0:
            av = cp.mean(cpm[:, idx, idx].astype(cp.float64), axis=1)
            mp = mp + c * cp.einsum('bij,bj->bi', cpm.astype(cp.float64),
                                    mp) / cp.maximum(av, 1e-30)[:, None]
        u = gam * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        if on and c != 0.0:
            if fam == "B":
                mn = mn + c * s * ph * (u * u - 1.0)
            elif fam == "F":
                um = cp.mean(u, axis=1, keepdims=True)
                mn = mn * (1.0 + c * (u - um) / cp.maximum(cp.abs(um), 1e-30))
        if i == DEPTH - 1:
            return mn
        pv = (ph / s).astype(cp.float32)
        Ph32 = Ph.astype(cp.float32)
        C2 = cpm * cpm
        C2 *= 0.5
        C2 *= pv[:, :, None]
        C2 *= pv[:, None, :]
        cpm *= Ph32[:, :, None]
        cpm *= Ph32[:, None, :]
        cpm += C2
        del C2
        cpm[:, idx, idx] = ((mp * mp + s2) * Ph + (mp * s) * ph
                            - mn * mn).astype(cp.float32)
        C = cpm
        m = mn
    return m


print(f"trans104 · {NNET} мереж · пачка {PACK} · сім'ї {FAMS} · "
      f"масштаби c {SCALE} · κ {KGRID} · gam {GGRID}", flush=True)

buf_w, buf_t, base = [], [], 0
t0 = time.time()
for wj, tj in stream(NNET):
    buf_w.append(wj)
    buf_t.append(tj)
    if len(buf_w) < PACK:
        continue
    key = f"p{base}"
    if key not in state:
        Wb = cp.asarray(np.stack(buf_w))
        Tb = cp.asarray(np.stack(buf_t))
        TT = (Tb * Tb).sum(1)
        rec = {"TT": [float(x) for x in TT]}
        for kap in KGRID:
          for gam in GGRID:
            for fam in FAMS:
                for c in cgrid(fam):
                    if c == 0.0 and fam != "A":
                        continue                      # 🔒 пломба (1): одна база
                    mm = run(Wb, kap, gam, fam, c)
                    rec[f"{kap}|{gam}|{fam}|{c}"] = [
                        [float(x) for x in (mm * mm).sum(1)],
                        [float(x) for x in (mm * Tb).sum(1)]]
                    del mm
        state[key] = rec
        save()
        del Wb, Tb
        cp.get_default_memory_pool().free_all_blocks()
        print(f"  ✅ пачка {base}..{base+PACK-1} ({time.time()-t0:.1f} с)",
              flush=True)
    base += PACK
    buf_w, buf_t = [], []

# ------------------------------------------------------------------ аналіз
packs = sorted(state, key=lambda s: int(s[1:]))
NET = []
for p in packs:
    for j in range(len(state[p]["TT"])):
        NET.append((p, j))
NN = len(NET)
HA = list(range(0, NN // 2))
HB = list(range(NN // 2, NN))


def err(sub, kap, gam, fam, c):
    key = f"{kap}|{gam}|{fam}|{c}" if not (c == 0.0 and fam != "A") \
        else f"{kap}|{gam}|A|0.0"
    tot = ttot = 0.0
    for i in sub:
        p, j = NET[i]
        if key not in state[p]:
            return None
        mm = state[p][key][0][j]
        mT = state[p][key][1][j]
        tt = state[p]["TT"][j]
        tot += tt - (mT * mT) / max(mm, 1e-300)
        ttot += tt
    return math.sqrt(max(tot, 0.0) / ttot)


bA = min((err(HA, k, g, "A", 0.0), k, g) for k in KGRID for g in GGRID)
bB = min((err(HB, k, g, "A", 0.0), k, g) for k in KGRID for g in GGRID)
bAll = min((err(range(NN), k, g, "A", 0.0), k, g)
           for k in KGRID for g in GGRID)
print(f"\n🔒 ПЛОМБА (2) база на {NN} мережах: {100*bAll[0]:.4f}% "
      f"(κ*={bAll[1]}, gam*={bAll[2]}) · очікувано 0.48…0.52%")
print(f"   база A-половина {100*bA[0]:.4f}%  ·  B-половина {100*bB[0]:.4f}%")

print(f"\n{'сім.':>5} {'c*(A)':>8} {'κ*':>7} {'gam*':>6} {'in-sample':>10} "
      f"{'×A':>7} {'КРОС A→B':>9} {'×B':>7} {'c*(B)':>8} {'КРОС B→A':>8} {'×A':>7}")
res = []
for fam in FAMS:
    best = b2 = None
    for kap in KGRID:
      for gam in GGRID:
        for c in cgrid(fam):
            e = err(HA, kap, gam, fam, c)
            if e is not None and e == e and (best is None or e < best[0]):
                best = (e, kap, gam, c)     # e==e відсіює nan (сім'ї A,F рвуться)
            e = err(HB, kap, gam, fam, c)
            if e is not None and e == e and (b2 is None or e < b2[0]):
                b2 = (e, kap, gam, c)
    if best is None or b2 is None:
        continue
    xb = err(HB, best[1], best[2], fam, best[3])
    xa = err(HA, b2[1], b2[2], fam, b2[3])
    if xb != xb or xa != xa:
        continue
    res.append((fam, best, b2, xa, xb))
    print(f"{fam:>5} {best[3]:>8.5f} {best[1]:>7.4f} {best[2]:>6.3f} "
          f"{100*best[0]:>9.4f}% {bA[0]/best[0]:>7.4f} {100*xb:>8.4f}% "
          f"{bB[0]/xb:>7.4f} {b2[3]:>8.5f} {100*xa:>7.4f}% {bA[0]/xa:>7.4f}")

print("\n🔒 ПЛОМБА (3): вирок ЛИШЕ за колонками КРОС. in-sample завжди ≥1.")
print("🔒 ПЛОМБА (4): κ перепідбирався разом із c у кожній клітинці.")
liv = [r for r in res if bB[0] / r[4] > 1.02 and bA[0] / r[3] > 1.02]
if liv:
    for r in liv:
        g = math.sqrt((bB[0] / r[4]) * (bA[0] / r[3]))
        lev = {1.0: 1.0, 1.5: 1.095, 2.0: 1.228, 4.0: 2.139, 8.0: 5.787}
        print(f"\n🟢 ЖИВА сім'я {r[0]}: крос ×{g:.4f} у ТОЧНОСТІ "
              f"⟹ у балі ≈×{1 + (g - 1) * 0.22:.4f}")
else:
    print("\n⛔ ЖОДНА сім'я не дала крос >×1.02 в ОБИДВА боки.")
