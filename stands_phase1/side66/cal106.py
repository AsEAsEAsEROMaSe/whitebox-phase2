"""cal106 — ЧИ ПЕРЕФІЧЕНІ НАШІ КОНСТАНТИ? 1000 МЕРЕЖ ПРОТИ 50.

ПИТАННЯ (призове, не методичне). Фінальний ранг фази 1 — приватна переоцінка на
СВІЖИХ мережах. Наші `κ` і `gam` підбирались на 50 публічних. Закон ємності
(С84) дає штраф перефіту 11.76%·k/N ⟹ при N=50, k=2 це ~0.5%. Але це ОЦІНКА
з іншої задачі; ПРЯМО на цих двох константах перефіт ніколи не мірявся.

ЧОМУ САМЕ `κ` І `gam`, І НІЧОГО БІЛЬШЕ.
Ціль після оптимального масштабу є КОСИНУСНОЮ (С100):
    min_a ‖a·m − T‖² = ‖T‖²·(1 − cos²(m,T))
⟹ `_ALPHA`, `_A_SHRINK`, `a_i`, `_W_BLEND` НЕ ВХОДЯТЬ у неї за побудовою —
вони чисті масштаби/ваги. Кут міняють РІВНО ДВІ константи: `κ` (множник на `C`
при i>KSIG) і `gam` (множник на аргумент воріт). Тому цей стенд має право
судити саме їх — і не має права судити решту.

ПЛАН. Сітка 5×5 по (κ, gam), 1000 мереж спліту `full`, точна істина 1e9.
   A. оптимум на ВСІХ 1000
   B. оптимум на ПЕРШИХ 50 (імітує те, що ми робили насправді)
   C. ціна перенесення B на решту 950 = ПРЯМИЙ вимір перефіту
   D. розкид оптимуму по 20 непересічних блоків по 50 ⟹ чи стабільний він

🔒 ПЛОМБИ
   П1 наша точка (0.9985, 1.048) при 1000 мережах мусить дати ≈0.45–0.52%
      (base104: full/100 мереж дає 0.4384%, mini/100 дає ≈0.507%)
   П2 сітка мусить бути ГЛАДКОЮ (парабола) — інакше крок завеликий (закон С104)
   П3 κ=1.0 мусить бути ГІРШЕ за κ=0.9985 (три незалежні виміри це казали)

📣 ПЕРЕДБАЧЕННЯ, ЗАПИСАНЕ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
   (1) оптимум 1000 мереж лежить у межах ОДНОГО кроку сітки від (0.9985, 1.048)
       — бо обидві константи вже двічі незалежно перевідкриті іншими стендами
       на інших наборах (`step105` скан, 9×9 GP-скан).           шанс 65%
   (2) ціна перенесення 50→950 буде МЕНШОЮ за 0.5% у косинусній похибці
                                                                 шанс 60%
   (3) якщо щось попливе — це буде `gam`, а не `κ`: `gam` підбирався РАЗОМ
       із `_W_BLEND` у повному оцінювачі, тобто в іншій цільовій функції
                                                                 шанс 55%
   ⚠️ Вирок (2) читати ЛИШЕ як «чи стабільні константи», НЕ як приріст балу:
   косинусна похибка — не бал, важіль точності 0.22 у нашій робочій точці.
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
KSIG = 8
ROOT = ("/home/trim/.cache/huggingface/hub/"
        "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
        "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
PACK = 20
OUT = "/home/trim/wblogs/cal106_state.json"

KAPS = [0.9975, 0.9980, 0.9985, 0.9990, 0.9995]
GAMS = [1.0080, 1.0160, 1.0240, 1.0320, 1.0480]
OURS = (0.9985, 1.0480)

if cp.cuda.runtime.getDeviceCount() != 1:
    print("ВІДМОВА: видно не рівно одну карту — пінування по UUID обов'язкове")
    sys.exit(2)


def phi(x):
    return cp.exp(-0.5 * x * x) / SQ2PI


def stream(split, nmax):
    got = 0
    for fp in sorted(glob.glob(ROOT + split + "/data-*.arrow")):
        if got >= nmax:
            break
        for b in pa.ipc.open_stream(pa.memory_map(fp, "rb")):
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
            for j in range(min(nb, nmax - got)):
                yield np.array(wv[j], dtype=np.float32), \
                    np.array(tv[j], dtype=np.float64)
            got += min(nb, nmax - got)
            del xw, xt, wv, tv


def recur(Wb, kap, gam):
    """Рекурсія сабміту, скорочена до того, що впливає на КУТ."""
    P = Wb.shape[0]
    m = cp.zeros((P, W_), dtype=cp.float64)
    C = cp.empty((P, W_, W_), dtype=cp.float32)
    C[:] = cp.eye(W_, dtype=cp.float32)
    dg = cp.arange(W_)
    for i in range(DEPTH):
        w = Wb[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w).astype(cp.float64)
        if i > KSIG:
            cpm = cpm * kap
        s2 = cp.maximum(cpm[:, dg, dg], 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w.astype(cp.float64))
        u = gam * (mp / s)
        Ph, ph = ndtr(u), phi(u)
        mn = mp * Ph + s * ph
        if i == DEPTH - 1:
            return mn
        pv = (ph / s).astype(cp.float32)
        P32 = Ph.astype(cp.float32)
        c32 = cpm.astype(cp.float32)
        C2 = c32 * c32
        C2 *= 0.5
        C2 *= pv[:, :, None]
        C2 *= pv[:, None, :]
        c32 *= P32[:, :, None]
        c32 *= P32[:, None, :]
        c32 += C2
        del C2
        c32[:, dg, dg] = ((mp * mp + s2) * Ph + (mp * s) * ph
                          - mn * mn).astype(cp.float32)
        C, m = c32, mn
    return m


# ── стійкість: підхоплення зробленого ────────────────────────────────────────
done = 0
NUM = {}
DEN = []
SC = {}
if os.path.exists(OUT):
    st = json.load(open(OUT))
    done, NUM, DEN = st["done"], {k: v for k, v in st["num"].items()}, st["den"]
    SC = {k: v for k, v in st.get("sc", {}).items()}
    print(f"підхоплено {done} мереж із чекпоінта", flush=True)

print(f"cal106 · {NNET} мереж `full` · сітка {len(KAPS)}×{len(GAMS)} · "
      f"пачка {PACK}", flush=True)
t0 = time.time()
bw, bt, seen, fails = [], [], 0, 0

for wn, tn in stream("full", NNET):
    seen += 1
    if seen <= done:
        continue
    bw.append(wn)
    bt.append(tn)
    if len(bw) < PACK and seen < NNET:
        continue
    try:
        Wb = cp.asarray(np.stack(bw))
        Tb = cp.asarray(np.stack(bt))
        tt = [float(Tb[a] @ Tb[a]) for a in range(len(bw))]
        DEN.extend(tt)
        for kap in KAPS:
            for gam in GAMS:
                key = f"{kap:.4f}_{gam:.4f}"
                mm = recur(Wb, kap, gam)
                acc = NUM.setdefault(key, [])
                sc = SC.setdefault(key, [0.0, 0.0])
                for a in range(len(bw)):
                    t, x = Tb[a], mm[a]
                    xt, xx = float(x @ t), float(x @ x)
                    acc.append(tt[a] - xt ** 2 / max(xx, 1e-300))
                    sc[0] += xt
                    sc[1] += xx
                del mm
        del Wb, Tb
        fails = 0
    except Exception as exc:                                   # noqa: BLE001
        fails += 1
        print(f"⚠️ пачка впала ({exc}); {fails} поспіль", flush=True)
        if fails >= 3:
            print("три невдачі поспіль — чистий вихід зі збереженням")
            break
    done = seen
    bw, bt = [], []
    cp.get_default_memory_pool().free_all_blocks()
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"done": done, "num": NUM, "den": DEN, "sc": SC}, fh)
    os.replace(tmp, OUT)
    if done % 100 == 0:
        print(f"  {done}/{NNET} ({time.time()-t0:.0f} с)", flush=True)

# ── аналіз ───────────────────────────────────────────────────────────────────
N = len(DEN)
den = np.array(DEN)


def agg(key, idx):
    a = np.array(NUM[key])[idx]
    return math.sqrt(a.sum() / den[idx].sum())


ALL = np.arange(N)
FIRST50 = np.arange(min(50, N))
REST = np.arange(min(50, N), N)

print(f"\n════ ЗМІРЯНО НА {N} МЕРЕЖАХ ════\n")
print("КОСИНУСНА ПОХИБКА, %% — рядки κ, стовпці gam")
hdr = "   κ\\gam" + "".join(f"{g:>10.4f}" for g in GAMS)
print(hdr)
grid = np.zeros((len(KAPS), len(GAMS)))
for i, kap in enumerate(KAPS):
    row = f"{kap:>9.4f}"
    for j, gam in enumerate(GAMS):
        v = agg(f"{kap:.4f}_{gam:.4f}", ALL)
        grid[i, j] = v
        row += f"{100*v:>10.4f}"
    print(row)

bi, bj = np.unravel_index(np.argmin(grid), grid.shape)
ours = agg(f"{OURS[0]:.4f}_{OURS[1]:.4f}", ALL)
print(f"\nA. ОПТИМУМ на {N} мережах: κ={KAPS[bi]:.4f} gam={GAMS[bj]:.4f} "
      f"⟹ {100*grid[bi,bj]:.4f}%")
print(f"   НАША ТОЧКА  κ={OURS[0]:.4f} gam={OURS[1]:.4f} "
      f"⟹ {100*ours:.4f}%   відставання ×{ours/grid[bi,bj]:.5f}")

g50 = np.array([[agg(f"{k:.4f}_{g:.4f}", FIRST50) for g in GAMS] for k in KAPS])
b50i, b50j = np.unravel_index(np.argmin(g50), g50.shape)
print(f"\nB. ОПТИМУМ на ПЕРШИХ 50: κ={KAPS[b50i]:.4f} gam={GAMS[b50j]:.4f}")

if len(REST):
    on_rest_50 = agg(f"{KAPS[b50i]:.4f}_{GAMS[b50j]:.4f}", REST)
    gr = np.array([[agg(f"{k:.4f}_{g:.4f}", REST) for g in GAMS] for k in KAPS])
    ri, rj = np.unravel_index(np.argmin(gr), gr.shape)
    print(f"C. ЦІНА ПЕРЕФІТУ на решті {len(REST)}: підібране на 50 дає "
          f"{100*on_rest_50:.4f}% проти найкращого {100*gr[ri,rj]:.4f}% "
          f"⟹ ×{on_rest_50/gr[ri,rj]:.5f}")
    print(f"   наша точка на решті: {100*agg(f'{OURS[0]:.4f}_{OURS[1]:.4f}', REST):.4f}%")

print("\nD. СТАБІЛЬНІСТЬ ОПТИМУМУ ПО НЕПЕРЕСІЧНИХ БЛОКАХ ПО 50")
cnt = {}
for b in range(N // 50):
    idx = np.arange(b * 50, (b + 1) * 50)
    gb = np.array([[agg(f"{k:.4f}_{g:.4f}", idx) for g in GAMS] for k in KAPS])
    i2, j2 = np.unravel_index(np.argmin(gb), gb.shape)
    cnt[(KAPS[i2], GAMS[j2])] = cnt.get((KAPS[i2], GAMS[j2]), 0) + 1
for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
    print(f"   κ={k[0]:.4f} gam={k[1]:.4f} обрано {v} з {N//50} блоків")

print("\n🔒 П1 наша точка в межах 0.45–0.52%: "
      f"{'ТАК' if 0.0045 <= ours <= 0.0052 else 'НІ'} ({100*ours:.4f}%)")
sm = all(grid[i, j] <= max(grid[max(i-1,0), j], grid[min(i+1,len(KAPS)-1), j])
         + 1e-9 or True for i in range(len(KAPS)) for j in range(len(GAMS)))
print(f"🔒 П2 гладкість — дивитись таблицю очима (парабола в обидва боки)")
kedge = agg(f"{KAPS[-1]:.4f}_{OURS[1]:.4f}", ALL)
print(f"🔒 П3 κ вище за 0.9985 гірше: "
      f"{'ТАК' if kedge > ours else 'НІ'} "
      f"(κ={KAPS[-1]:.4f} дає {100*kedge:.4f}% проти {100*ours:.4f}%)")
print("\nЕ. ОПТИМАЛЬНИЙ ГЛОБАЛЬНИЙ МАСШТАБ (пулений МНК) ДЛЯ КОЖНОЇ ТОЧКИ")
print("   α = Σ(x·T)/Σ(x·x) − 1 ; наш сабміт несе α = −0.00679")
print("   κ\\gam" + "".join(f"{g:>10.4f}" for g in GAMS))
for kap in KAPS:
    row = f"{kap:>9.4f}"
    for gam in GAMS:
        st_ = SC[f"{kap:.4f}_{gam:.4f}"]
        row += f"{st_[0]/st_[1]-1.0:>10.5f}"
    print(row)
print(f"\nчас {time.time()-t0:.0f} с")
