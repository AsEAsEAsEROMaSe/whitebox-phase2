"""step105 — СТЕЛЯ «ІДЕАЛЬНОГО КРОКУ» В РЕАЛЬНІЙ РЕКУРСІЇ (питання ЮЗЕРА про 1e-8).

ЩО ЗАКРИВАЄ. Питання «чи 1e-8 досяжне без повного тензора γ» має пряму відповідь
БЕЗ прогону: `orac101e` уже підставив ТОЧНІ понейронні γ (оракульний еквівалент
тензора 256³) у рекурсію з нашими (m,C) і дістав ×1.02. Тензор — не вузьке місце.

АЛЕ В ЛАНЦЮГУ ЛИШИЛАСЬ ДІРА, І ВОНА ЄДИНА ЗМІСТОВНА:
   edgr104   похибка КРОКУ = Еджворт на R² = 0.983   (при ІСТИННИХ входах)
   equil104  крок 0.14…0.26% проти накопиченої 0.668…0.821%  ⟹ крок ≈ 1/4
   арифметика: зняти 98% від 1/4 мало б дати ×1.3 у накопиченій
   orac101e  зміряно ×1.02
Розрив ×1.3 проти ×1.02 не пояснений числом. Дві гіпотези:
   (H1) поправка ПРОМАХУЄТЬСЯ ТОЧКОЮ ПРИКЛАДАННЯ — наші (t,s) зміщені, тож
        поправка, побудована для істинної точки, декорелює з реальною похибкою
   (H2) похибка кроку вже частково ПОГЛИНУТА калібровкою κ, gam ⟹ знімати її
        вдруге нічого не дає

ЯК РОЗДІЛЯЄМО. Замість еджвортівської АПРОКСИМАЦІЇ вкидаємо ТОЧНУ похибку кроку
`b_l` (зміряну незалежно) — це стеля всього класу «кращий крок», якої в нас нема:
С74 міряла на 6 мережах і в конфігурації БЕЗ κ, gam.

   base   наша рекурсія (κ, gam)                                 ← ПЛОМБА
   inj    m ← base − c·b_l   на кожному шарі, b_l ТОЧНА          ← СТЕЛЯ КЛАСУ
   injp   те саме, але b_l перенесена в НАШУ точку через ∂/∂(mp,s)   (H1)
   foreign  b_l ЧУЖОЇ мережі                                     ← 🔒 КОНТРОЛЬ
   injm   inj ПОВЕРХ істинних (m,C)                              ← 🔒 ТАВТОЛОГІЯ

`b_l` = (наш крок від ІСТИННИХ (m_l,C_l)) − (істинне m_{l+1}). Саме той об'єкт,
який edgr104 показав еджвортівським; тут він береться ТОЧНО, без апроксимації.

🔒 ПЛОМБИ
 (P1) base на `full` з gam=1.048 мусить дати ≈0.43%  [base104, 100 мереж]
      — без цього прогін НЕВАЛІДНИЙ, решту чисел не читати
 (P2) injm (точний крок ПОВЕРХ істинних входів) мусить дати ≈0.00% — тавтологія,
      це пломба ПРИЛАДУ: якщо вона не проходить, `b_l` рахується не так
 (P3) foreign мусить дати оптимум c → 0
 (P4) розкид по мережах друкуємо: вирок на 16 мережах читати з ним разом

📉 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
   inj дасть ×1.05…×1.6 (клас мертвий остаточно)              70%
   inj ≥ ×3 (клас оживає, варто шукати кращу формулу кроку)   15%
   injp помітно > inj (винна ТОЧКА ПРИКЛАДАННЯ, H1)           35%
   оптимальний c виявиться < 1 («поправка стиснена», 6-й раз) 60%

ЦІНА ПИТАННЯ: inj — це верхня межа для БУДЬ-ЯКОЇ ідеї виду «точніша формула
кроку», включно з усіма γ-схемами, квадратурами і навченими поправками разом.
Якщо inj < ×1.6, то `1e-8` закрито з боку кроку остаточно, і лишається лише
ПЕРЕНОС (який С101 scan101 і С104 trans104 уже завалили з двох боків).
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NPAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
CHUNK = 4096
KSIG = 8
KGRID = [0.9970, 0.9985, 1.0000]
GGRID = [1.032, 1.040, 1.048, 1.056, 1.064]
CGRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
MODES = ["base", "inj", "injp", "foreign", "injm"]
CKPT = "/home/trim/wblogs/step105_state.json"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 бачу не рівно одну карту — пінування по UUID не спрацювало")
    sys.exit(2)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def stream(nmax):
    got = 0
    for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= nmax:
            break
        for bt in pa.ipc.open_stream(pa.memory_map(fp, "rb")):
            if got >= nmax:
                break
            xw = bt.column("weights")
            while not pa.types.is_primitive(xw.type):
                xw = xw.flatten()
            xt = bt.column("final_means")
            while not pa.types.is_primitive(xt.type):
                xt = xt.flatten()
            nb = bt.num_rows
            wv = np.asarray(xw).reshape(nb, DEPTH, W_, W_)
            tv = np.asarray(xt).reshape(nb, W_)
            for j in range(min(nb, nmax - got)):
                yield (np.array(wv[j], dtype=np.float32),
                       np.array(tv[j], dtype=np.float64))
            got += min(nb, nmax - got)
            del xw, xt, wv, tv


def truth(W64, seed):
    """істинні (m_l, C_l) активацій — з яких рахується ТОЧНА похибка кроку."""
    N = 2 * NPAIRS
    rng = cp.random.default_rng(seed)
    A1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    A2 = cp.zeros((DEPTH, W_, W_), dtype=cp.float64)
    O1 = cp.zeros(W_, dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)
        for i in range(DEPTH):
            a64 = a.astype(cp.float64)
            A1[i] += a64.sum(0)
            A2[i] += a64.T @ a64
            p = a @ cp.asarray(W64[i], dtype=cp.float32)
            a = cp.maximum(p, 0.0)
            if i == DEPTH - 1:
                O1 += a.astype(cp.float64).sum(0)
        done += nb
    del z, a, a64, p
    cp.get_default_memory_pool().free_all_blocks()
    mt = A1 / N
    Ct = A2 / N - mt[:, :, None] * mt[:, None, :]
    Ct = 0.5 * (Ct + cp.transpose(Ct, (0, 2, 1)))
    return mt, Ct, O1 / N


def step(w, C, m, kap, gam, i):
    """один крок замикання; повертає (m_new, mp, s, Ph, ph, cpm)."""
    cpm = w.T @ C @ w
    if i > KSIG:
        cpm = cpm * kap
    s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-30))
    mp = m @ w
    u = gam * (mp / s)
    Ph, ph = ndtr(u), phi(u)
    return mp * Ph + s * ph, mp, s, Ph, ph, cpm


def stepb(W64, mt, Ct, T, kap, gam):
    """ТОЧНА похибка кроку при ІСТИННИХ входах: b_l = наш_крок(істина) − істина.

    Разом із нею — похідні ∂m/∂mp і ∂m/∂s у ТІЙ САМІЙ точці, щоб уміти
    перенести b у нашу точку (режим injp, гіпотеза H1)."""
    B = cp.zeros((DEPTH, W_), dtype=cp.float64)
    MP = cp.zeros((DEPTH, W_), dtype=cp.float64)
    SS = cp.zeros((DEPTH, W_), dtype=cp.float64)
    for i in range(DEPTH):
        Cin = cp.eye(W_, dtype=cp.float64) if i == 0 else Ct[i]
        min_ = cp.zeros(W_, dtype=cp.float64) if i == 0 else mt[i]
        mn, mp, s, Ph, ph, _ = step(W64[i], Cin, min_, kap, gam, i)
        tgt = T if i == DEPTH - 1 else mt[i + 1]
        B[i] = mn - tgt
        MP[i], SS[i] = mp, s
    return B, MP, SS


def closure(W64, mt, Ct, kap, gam, mode, c, B, MP, SS):
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        mn, mp, s, Ph, ph, cpm = step(W64[i], C, m, kap, gam, i)
        if mode != "base" and c != 0.0:
            corr = B[i]
            if mode == "injp":
                # перенос b у НАШУ точку: b(mp,s) ≈ b(mp₀,s₀)·(s/s₀), бо b
                # однорідна першого степеня по масштабу — це і є перевірка H1
                corr = corr * (s / cp.maximum(SS[i], 1e-30))
            mn = mn - c * corr
        m = mn
        if i == DEPTH - 1:
            break
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        Co = Co + cp.diag(d - cp.diagonal(Co))
        if mode == "injm":
            # 🚨 вхід НАСТУПНОГО кроку — це (mt[i+1], Ct[i+1]), не Ct[i].
            # На цьому пломба P2 упіймала прилад у димовому прогоні.
            C, m = Ct[i + 1], mt[i + 1]
        else:
            C = Co
    return m


state = {}
if os.path.exists(CKPT):
    try:
        state = json.load(open(CKPT))
    except Exception:
        state = {}

print(f"step105 · {NNET} мереж × {NPAIRS} пар · режими {MODES}", flush=True)
t0 = time.time()
PREV = None
fails = 0
for net, (Wn, Tn) in enumerate(stream(NNET)):
    if str(net) in state:
        PREV = None
        continue
    try:
        W64 = [cp.asarray(Wn[L]).astype(cp.float64) for L in range(DEPTH)]
        T = cp.asarray(Tn)
        mt, Ct, out_t = truth(W64, 1050 + net)
        rec = {"_TT": float(T @ T)}
        KEEP = None
        for kap in KGRID:
            for gam in GGRID:
                B, MP, SS = stepb(W64, mt, Ct, T, kap, gam)
                for mode in MODES:
                    if mode == "foreign" and PREV is None:
                        continue          # перша мережа не має чужої B
                    cs = [1.0] if mode == "base" else CGRID
                    Bu = PREV if mode == "foreign" else B
                    for c in cs:
                        mm = closure(W64, mt, Ct, kap, gam, mode, c, Bu, MP, SS)
                        if not bool(cp.all(cp.isfinite(mm))):
                            continue
                        rec[f"{mode}|{kap}|{gam}|{c}"] = [
                            float(mm @ mm), float(mm @ T)]
                if kap == 0.9985 and gam == 1.048:
                    rec["_bn"] = [float(cp.linalg.norm(B[i])) for i in
                                  (0, 1, 8, 16, 24, 31)]
                    rec["_mn"] = float(cp.linalg.norm(T))
                    KEEP = B.copy()
        state[str(net)] = rec
        if KEEP is not None:
            PREV = KEEP
        tmp = CKPT + ".tmp"
        json.dump(state, open(tmp, "w"))
        os.replace(tmp, CKPT)
        print(f"  ✅ мережа {net} ({time.time()-t0:.1f} с)", flush=True)
        fails = 0
    except Exception as e:                                    # noqa: BLE001
        fails += 1
        print(f"  ⚠️ мережа {net}: {type(e).__name__}: {e}", flush=True)
        if fails >= 3:
            print("🚨 три невдачі поспіль — чистий вихід зі збереженням")
            break
    finally:
        cp.get_default_memory_pool().free_all_blocks()


def agg(key):
    """агрегат по мережах, де ключ присутній (foreign не має першої мережі)."""
    num = den = 0.0
    ind = []
    for r in state.values():
        if key not in r:
            continue
        xx, xt = r[key]
        tt = r["_TT"]
        e = tt - xt * xt / max(xx, 1e-300)
        num += e
        den += tt
        ind.append(math.sqrt(max(e, 0.0) / tt))
    if len(ind) < max(2, len(state) - 2):
        return None, None
    return math.sqrt(num / den), np.array(ind)


nets = len(state)
print(f"\n📊 {nets} мереж · косинусна похибка (масштаб оптимальний по мережі)")
best = {}
for mode in MODES:
    rows = []
    for kap in KGRID:
        for gam in GGRID:
            for c in ([1.0] if mode == "base" else CGRID):
                a, ind = agg(f"{mode}|{kap}|{gam}|{c}")
                if a is not None:
                    rows.append((a, kap, gam, c, ind))
    if not rows:
        continue
    rows.sort()
    best[mode] = rows[0]
    a, kap, gam, c, ind = rows[0]
    print(f"{mode:>9}  {100*a:8.4f}%   κ={kap} gam={gam} c={c:4.2f}"
          f"   медіана {100*np.median(ind):7.4f}%  sd(log) {np.std(np.log(ind + 1e-30)):.3f}")

if "base" in best and "inj" in best:
    b0 = best["base"][0]
    print(f"\n{'режим':>9}{'×точність':>12}{'бал':>14}")
    for mode in MODES:
        if mode not in best:
            continue
        if best[mode][0] < 1e-9:
            print(f"{mode:>9}{'∞':>12}{'0 (тавтологія)':>14}")
            continue
        k = b0 / best[mode][0]
        # бал через зміряну модель V·E²/(V+E²), V = 1.9227e-06, E² від k
        E2 = 2.3385e-05 / (k * k)
        V = 1.9227e-06
        mse = V * E2 / (V + E2)
        sc = 1.855e-07 * (mse / 1.7767e-06)
        print(f"{mode:>9}{k:>12.4f}{sc:>14.3e}")

print("\n🔒 ПЛОМБИ")
if "base" in best:
    print(f" (P1) base = {100*best['base'][0]:.4f}% проти очікуваних ≈0.43% "
          f"[base104, full, gam=1.048]")
if "injm" in best:
    print(f" (P2) injm = {100*best['injm'][0]:.4f}% — мусить бути ≈0 "
          f"(тавтологія: точний крок при точних входах)")
if "foreign" in best:
    print(f" (P3) foreign: оптимальний c = {best['foreign'][3]:.2f} "
          f"— мусить бути 0.00")
print("\n📉 ПЕРЕДБАЧЕННЯ: inj ×1.05…×1.6 (70%) · inj ≥ ×3 (15%) · c<1 (60%)")
if "inj" in best and "base" in best:
    k = best["base"][0] / best["inj"][0]
    if k >= 3.0:
        print(f"🟢 inj = ×{k:.3f} ⟹ КЛАС «кращий крок» ОЖИВАЄ — шукати формулу")
    elif k >= 1.6:
        print(f"🟡 inj = ×{k:.3f} ⟹ між передбаченнями, читати з розкидом")
    else:
        print(f"⛔ inj = ×{k:.3f} ⟹ СТЕЛЯ КЛАСУ «кращий крок» НИЖЧЕ ×1.6 —")
        print("   1e-8 закрито з боку КРОКУ остаточно; лишається лише ПЕРЕНОС,")
        print("   а він завалений з двох боків (scan101 ×0.9955, trans104 нуль)")
