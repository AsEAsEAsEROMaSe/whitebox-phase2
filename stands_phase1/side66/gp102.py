"""gp102 — ПОШУК У ПРОСТОРІ СТРУКТУР, А НЕ КОНСТАНТ

ЩО БУЛО НЕ ТАК ДОСІ (визнано): tour102/evo102 сканували сітку параметрів У
ФОРМІ, ЯКУ Я САМ ЗАДАВ. Простір не розширювався — розширювалась розмітка.
AlphaZero сильний тим, що ХІД — це елемент простору, а не коефіцієнт у
людській формулі. Тут хід = ВУЗОЛ ДЕРЕВА ВИРАЗУ.

ЩО ШУКАЄМО. Крок замикання при i>KSIG:
    u  = U(t)          аргумент воріт   (база: u = t · sub127f: u = 1.048·t)
    m' = mp·Φ(u) + A(t)·s·φ(u)          (база: A = 1)
    C  = κ·WᵀCW далі як було
`U` і `A` — ДЕРЕВА ВИРАЗІВ, не многочлени.

🔑 РОЗМІРНИЙ АНАЛІЗ ЯК ЧАСТИНА ГРАМАТИКИ: `u` і `A` безрозмірні ⟹ листками
можуть бути ЛИШЕ безрозмірні поля (`t`, глибина, константи), і `mp`/`s` у
граматику не входять. Це не обмеження зі стелі — це те, що робить простір
осмисленим, а не сміттям.

🔒 ВБУДОВАНА ПЛОМБА: дерево (U=t, A=1) МУСИТЬ відтворити базу біт у біт,
дерево (U=1.048·t, A=1) — прогін sub127f. Обидва в стартовій популяції.

ВИРОК — ЛИШЕ TEST-половина, якої пошук не бачив. Друкуємо ВСЮ популяцію-еліту
з TRAIN і TEST поруч: розрив між ними і є мірою перефіту структури.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
  - шанс, що GP знайде форму, яка б'є `gam` на TEST більш ніж на ×1.01: ~35%
  - шанс, що знайдене переживе повний оцінювач (грейдер): ~15%
  - найімовірніший результат: форма, близька до лінійної `u≈1.05t`, бо
    С101-г зміряла стелю лінійної корекції ×1.027, а нелінійність по `t`
    у робочій смузі |t|~3.5 має мало місця для маневру.
"""
import glob
import math
import random
import sys
import time

sys.path.insert(0, "/home/trim/gpupkg")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import cupy as cp                                             # noqa: E402
from cupyx.scipy.special import ndtr                          # noqa: E402

DEPTH, W_, KSIG = 32, 256, 8
SQ2PI = math.sqrt(2.0 * math.pi)
SRC = ("/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/"
       "snapshots/7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 500
POP, GENS, ELITE = 48, 22, 6
KAP = 0.9985
random.seed(20260811)

# ---------------------------------------------------------------- граматика
OP1 = ["neg", "tanh", "gauss", "ndtr", "sq", "rec"]
OP2 = ["add", "sub", "mul", "dvs"]
LEAF = ["t", "one", "dep"]
CONST = [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
MAXN = 13


def size(n):
    if n[0] in ("L", "K"):
        return 1
    if n[0] == "U":
        return 1 + size(n[2])
    return 1 + size(n[2]) + size(n[3])


def rnd(d=0):
    if d >= 2 or random.random() < 0.35:
        return ("K", random.choice(CONST)) if random.random() < 0.4 \
            else ("L", random.choice(LEAF))
    if random.random() < 0.35:
        return ("U", random.choice(OP1), rnd(d + 1))
    return ("B", random.choice(OP2), rnd(d + 1), rnd(d + 1))


def nodes(n, path=()):
    out = [(path, n)]
    if n[0] == "U":
        out += nodes(n[2], path + (2,))
    elif n[0] == "B":
        out += nodes(n[2], path + (2,)) + nodes(n[3], path + (3,))
    return out


def put(n, path, new):
    if not path:
        return new
    i = path[0]
    lst = list(n)
    lst[i] = put(n[i], path[1:], new)
    return tuple(lst)


def mutate(n):
    r = random.random()
    ns = nodes(n)
    path, sub = random.choice(ns)
    if r < 0.30:                                   # заміна піддерева
        return put(n, path, rnd(1))
    if r < 0.55 and sub[0] == "K":                 # зсув константи
        v = sub[1] * math.exp(random.gauss(0, 0.35)) + random.gauss(0, 0.03)
        return put(n, path, ("K", float(np.clip(v, -4, 4))))
    if r < 0.75 and sub[0] == "B":                 # заміна бінарної операції
        return put(n, path, ("B", random.choice(OP2), sub[2], sub[3]))
    if r < 0.88 and sub[0] == "U":                 # заміна унарної операції
        return put(n, path, ("U", random.choice(OP1), sub[2]))
    if r < 0.95:                                   # обгортка
        return put(n, path, ("B", random.choice(OP2), sub, rnd(2)))
    return put(n, path, ("U", random.choice(OP1), sub))


def cross(a, b):
    pa_, _ = random.choice(nodes(a))
    _, sb = random.choice(nodes(b))
    return put(a, pa_, sb)


def show(n):
    if n[0] == "K":
        return f"{n[1]:.4g}"
    if n[0] == "L":
        return n[1]
    if n[0] == "U":
        return f"{n[1]}({show(n[2])})"
    o = {"add": "+", "sub": "-", "mul": "*", "dvs": "/~"}[n[1]]
    return f"({show(n[2])}{o}{show(n[3])})"


def ev(n, t, dep):
    k = n[0]
    if k == "K":
        return n[1]
    if k == "L":
        return t if n[1] == "t" else (dep if n[1] == "dep" else 1.0)
    if k == "U":
        a = ev(n[2], t, dep)
        o = n[1]
        if o == "neg":
            return -a
        if o == "tanh":
            return cp.tanh(a)
        if o == "gauss":
            return cp.exp(-0.5 * a * a)
        if o == "ndtr":
            return ndtr(a) if not isinstance(a, float) else 0.5
        if o == "sq":
            return a * a
        return a / (1.0 + a * a)                    # rec: обмежена 1/x
    a = ev(n[2], t, dep)
    b = ev(n[3], t, dep)
    o = n[1]
    if o == "add":
        return a + b
    if o == "sub":
        return a - b
    if o == "mul":
        return a * b
    return a / cp.sqrt(1.0 + b * b)                 # dvs: безпечне ділення


# ------------------------------------------------------------------- дані
# 🚨 ЗАВАНТАЖЕННЯ В ОДНУ КОПІЮ (С103, після падіння машини).
# СТАРИЙ ШАБЛОН БУВ ПАСТКОЮ: `Ws.append(Wa[r])` кладе VIEW на таблицю файлу,
# тому `del Wa` її НЕ звільняв — усі 17 таблиць по ~495 МБ лишались живими
# ⟹ ~8 ГБ RAM на процес. Три такі процеси вичерпали 31 ГБ і система лягла.
Wall = np.empty((NNET, DEPTH, W_, W_), dtype=np.float32)
Tall = np.empty((NNET, W_), dtype=np.float64)
k = 0
for fp in sorted(glob.glob(SRC + "/data-*.arrow")):
    if k >= NNET:
        break
    tb = pa.ipc.open_stream(pa.memory_map(fp, "rb")).read_all()
    xw = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(xw.type):
        xw = xw.flatten()
    xt = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(xt.type):
        xt = xt.flatten()
    n = min(tb.num_rows, NNET - k)
    Wall[k:k + n] = np.asarray(xw).reshape(tb.num_rows, DEPTH, W_, W_)[:n]
    Tall[k:k + n] = np.asarray(xt).reshape(tb.num_rows, W_)[:n]
    k += n
    del tb, xw, xt
NNET = k
Wg = cp.asarray(Wall[:NNET])
Tg = cp.asarray(Tall[:NNET])
del Wall, Tall
IDX = cp.arange(W_)
IA = cp.arange(0, NNET // 2)
IB = cp.arange(NNET // 2, NNET)
print(f"gp102 · {NNET} мереж · TRAIN {len(IA)} / TEST {len(IB)} · "
      f"поп {POP} × {GENS} поколінь", flush=True)


def _san(x, lim, fill):
    """дерево може віддати і float (константа), і масив — обидва санітизуємо"""
    if isinstance(x, float):
        return fill if not math.isfinite(x) else float(np.clip(x, -lim, lim))
    return cp.nan_to_num(cp.clip(x, -lim, lim), nan=fill,
                         posinf=lim, neginf=-lim)


def _step(m, C, w, u_fn, kap=1.0):
    """один крок; u_fn(t) віддає (u, A). `cpm` після matmul — свіжий масив,
    тож його можна псувати in-place: це знімає 3 тимчасові по B·256²·4 байти.
    ⚠️ `kap` МУСИТЬ множити саме `cpm`, а не ваги: `mp = m@w` теж бере `w`,
    тож √κ у вагах просочився б у `m` — це вже інша модель."""
    cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w)
    if kap != 1.0:
        cpm *= kap
    s2 = cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-30)
    s = cp.sqrt(s2)
    mp = cp.einsum('bi,bij->bj', m, w)
    u, A = u_fn(mp / s)
    Ph = ndtr(u)
    ph = cp.exp(-0.5 * u * u) / SQ2PI
    m_new = mp * Ph + A * (s * ph)
    pv = ph / s
    C2 = cpm * cpm
    C2 *= 0.5
    C2 *= pv[:, :, None]
    C2 *= pv[:, None, :]
    cpm *= Ph[:, :, None]                       # ← in-place, cpm далі не потрібен
    cpm *= Ph[:, None, :]
    cpm += C2
    del C2
    cpm[:, IDX, IDX] = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
    return m_new, cpm


# 🔑 ПРЕФІКС КЕШУЄТЬСЯ: шари 0..KSIG не залежать від дерев (там `u=t, A=1`),
#    а перераховувались у кожному з ~950 прогонів ⟹ 9/32 = 28% роботи даремно.
_B = Wg.shape[0]
_m = cp.zeros((_B, W_), dtype=cp.float32)
_C = cp.broadcast_to(cp.eye(W_, dtype=cp.float32), (_B, W_, W_)).copy()
for _i in range(KSIG + 1):
    _m, _C = _step(_m, _C, Wg[:, _i], lambda t: (t, 1.0))
M0, C0 = _m, _C
del _m, _C


def run(tU, tA):
    m, C = M0, C0.copy()
    for i in range(KSIG + 1, DEPTH):
        dp = float(i) / 31.0

        def u_fn(t, _u=tU, _a=tA, _d=dp):
            return (_san(ev(_u, t, _d), 40.0, 0.0),
                    _san(ev(_a, t, _d), 20.0, 1.0))

        m, C = _step(m, C, Wg[:, i], u_fn, kap=KAP)
    M = cp.nan_to_num(m.astype(cp.float64))
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


T_ID = ("L", "t")
T_ONE = ("K", 1.0)
T_GAM = ("B", "mul", ("K", 1.048), ("L", "t"))

t0 = time.time()
e_base = run(T_ID, T_ONE)
e_gam = run(T_GAM, T_ONE)
dt = time.time() - t0
print(f"\n🔒 ПЛОМБИ ({dt/2:.2f} с/прогін)")
print(f"{'база u=t, A=1':>22} TRAIN {100*float(e_base[IA].mean()):.4f}%  "
      f"TEST {100*float(e_base[IB].mean()):.4f}%")
print(f"{'sub127f u=1.048t':>22} TRAIN {100*float(e_gam[IA].mean()):.4f}%  "
      f"TEST {100*float(e_gam[IB].mean()):.4f}%", flush=True)
GAM_TR = float(e_gam[IA].mean())
GAM_TE = float(e_gam[IB].mean())
BASE_TE = float(e_base[IB].mean())


STAB = 1e-6      # штраф на вузол. КАЛІБРОВКА: виграш gam = 1.17e-4 у частках
#                 ⟹ 2e-5/вузол (перша спроба) коштував БІЛЬШЕ за весь виграш
#                 і замикав пошук на коротких деревах. 1e-6 × 13 вузлів = 1.3e-5,
#                 тобто 11% виграшу gam — достатньо проти роздування, замало щоб
#                 заборонити структуру.


def fit(ind):
    """повертає (штрафоване для селекції, чисте для друку)"""
    try:
        e = run(ind[0], ind[1])
        v = float(e[IA].mean())
        if not math.isfinite(v) or v <= 0:
            return (9.9, 9.9)
        return (v + STAB * (size(ind[0]) + size(ind[1])), v)
    except Exception:
        return (9.9, 9.9)


pop = [(T_ID, T_ONE), (T_GAM, T_ONE),
       (("B", "mul", ("K", 1.02), ("L", "t")), T_ONE),
       (("B", "mul", ("K", 1.08), ("L", "t")), T_ONE)]
while len(pop) < POP:
    pop.append((rnd(), T_ONE) if random.random() < 0.5 else (rnd(), rnd()))

scored = [fit(p) + (p,) for p in pop]
print(f"\n--- ЕВОЛЮЦІЯ (штраф {STAB:.0e}/вузол; TRAIN друкується ЧИСТИЙ) ---",
      flush=True)
for g in range(GENS):
    scored.sort(key=lambda x: x[0])
    keep = scored[:ELITE]
    kids = []
    while len(kids) < POP - ELITE:
        a = min(random.sample(scored, 3), key=lambda x: x[0])[2]
        if random.random() < 0.30:
            b = min(random.sample(scored, 3), key=lambda x: x[0])[2]
            k = (cross(a[0], b[0]), a[1]) if random.random() < 0.5 \
                else (a[0], cross(a[1], b[1]))
        else:
            k = (mutate(a[0]), a[1]) if random.random() < 0.6 \
                else (a[0], mutate(a[1]))
        if size(k[0]) <= MAXN and size(k[1]) <= MAXN:
            kids.append(k)
    scored = keep + [fit(k) + (k,) for k in kids]
    b = min(scored, key=lambda x: x[0])
    print(f"  пок {g+1:2d}: TRAIN {100*b[1]:.4f}%  vs gam ×{GAM_TR/b[1]:.4f}  "
          f"U={show(b[2][0])}  A={show(b[2][1])}", flush=True)

scored.sort(key=lambda x: x[0])
print(f"\n--- ЕЛІТА: TRAIN vs TEST (розрив = перефіт структури) ---")
V, E2 = 1.9227e-06, 2.3385e-05
seen = set()
for _pen, _raw, ind in scored[:10]:
    key = (show(ind[0]), show(ind[1]))
    if key in seen:
        continue
    seen.add(key)
    e = run(ind[0], ind[1])
    tr, te = float(e[IA].mean()), float(e[IB].mean())
    x = GAM_TE / te
    e2 = E2 / x ** 2
    bal = (V * E2 / (V + E2)) / (V * e2 / (V + e2))
    sig = int((e[IB] < e_gam[IB]).sum())
    print(f"TRAIN {100*tr:.4f}%  TEST {100*te:.4f}%  vs gam ×{x:.4f}  "
          f"знак {sig}/{len(IB)}  бал ×{bal:.4f}")
    print(f"     U={show(ind[0])}   A={show(ind[1])}", flush=True)
print(f"\nдовідка: база TEST {100*BASE_TE:.4f}% · gam TEST {100*GAM_TE:.4f}%")
