"""gp102b — ГРАМАТИКА ЗІ СКАЛЯРАМИ РЕЖИМУ. Поправка стає функцією СТАНУ

ЧОМУ САМЕ ЦЕ, А НЕ «ЩЕ ОДНА ФОРМА ПО t»:
`edg` (γ₄ на НАШИХ m,C) = ×1.02 проти `edgm` (те саме на ІСТИННИХ) = ×22.8
⟹ покращення ФОРМИ розподілу при неточних моментах дає ×1.0x за побудовою.
Живе те, що чіпає НАКОПИЧЕННЯ по рекурсії — саме тому спрацювали `κ` і `gam`,
і саме тому вони глобальні, а не понейронні.

⟹ поправка мусить залежати не від `t` окремого нейрона, а від РЕЖИМУ мережі
на цьому шарі. Три скаляри, які замикання вже має задарма:

    fr = ‖m‖²/(‖m‖²+tr C)         «замороженість»   [С102: 0→0.943 по глибині]
    er = ((tr C)²/‖C‖_F²)/256     ефективний ранг    [С102: 0.65→0.011]
    mt = mean|t| / 4              насиченість воріт  [медіана |t| ≈ 3.5]

ЦІНА: tr C — сума діагоналі; ‖C‖_F² — 65536 множень × 23 шари = 1.5e6 флопів
= 5.5e-6 бюджету. НУЛЬ. Тобто гейт ціни пройдено ДО реалізації.

🔑 ЦЕ ВІДПОВІДАЄ НА ГЕЙТ «чи міняє ідея представлення стану»: поправка
перестає бути константою і стає функцією режиму, який рекурсія проходить.

🔒 ПЛОМБИ ті самі: (u=t,A=1) = база, (u=1.048t,A=1) = sub127f, обидва в
стартовій популяції. Вирок — TEST-половина.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  - шанс, що режимні листки дадуть понад ×1.02 на TEST проти gam: ~30%
  - найімовірніша знахідка: щось монотонне по `fr` або `dep` — бо обидва
    майже те саме (frozen росте з глибиною), і тоді це буде κ(L), який
    С100 вже зміряв як майже порожній (крос ×1.0109/×1.0001)
  - справжня новина була б, якби виграла `er` або `mt`: вони НЕ монотонні
    функції глибини і несуть щось про КОНКРЕТНУ мережу
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
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 400
POP, GENS, ELITE = 48, 22, 6
KAP = 0.9985
STAB = 1e-6
MAXN = 13
random.seed(20260811)

OP1 = ["neg", "tanh", "gauss", "ndtr", "sq", "rec"]
OP2 = ["add", "sub", "mul", "dvs"]
LEAF = ["t", "one", "dep", "fr", "mt"]
# ⛔ `er` (ефективний ранг) ПРИБРАНО ще до пошуку, з двох незалежних причин:
#    (1) ЦІНА: (cpm*cpm).sum((1,2)) — 105 МБ тимчасових щошару, прогін 0.49→3.09 с
#    (2) ІНФОРМАЦІЯ: перший же прогін показав 0.0968→0.0174 монотонно по глибині
#        ⟹ це перепараметризація `dep`, а С102 вже зміряла, що eff_rank(C) —
#        «ні ціль, ні проксі». Платити ×6 часу за дублікат наявного листка немає за що.
CONST = [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]


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
    path, sub = random.choice(nodes(n))
    if r < 0.30:
        return put(n, path, rnd(1))
    if r < 0.55 and sub[0] == "K":
        v = sub[1] * math.exp(random.gauss(0, 0.35)) + random.gauss(0, 0.03)
        return put(n, path, ("K", float(np.clip(v, -4, 4))))
    if r < 0.75 and sub[0] == "B":
        return put(n, path, ("B", random.choice(OP2), sub[2], sub[3]))
    if r < 0.88 and sub[0] == "U":
        return put(n, path, ("U", random.choice(OP1), sub[2]))
    if r < 0.95:
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


def ev(n, env):
    k = n[0]
    if k == "K":
        return n[1]
    if k == "L":
        return env[n[1]]
    if k == "U":
        a = ev(n[2], env)
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
        return a / (1.0 + a * a)
    a = ev(n[2], env)
    b = ev(n[3], env)
    o = n[1]
    if o == "add":
        return a + b
    if o == "sub":
        return a - b
    if o == "mul":
        return a * b
    return a / cp.sqrt(1.0 + b * b)


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
print(f"gp102b · {NNET} мереж · TRAIN {len(IA)} / TEST {len(IB)} · "
      f"листки {LEAF}", flush=True)


def _san(x, lim, fill):
    if isinstance(x, float):
        return fill if not math.isfinite(x) else float(np.clip(x, -lim, lim))
    return cp.nan_to_num(cp.clip(x, -lim, lim), nan=fill,
                         posinf=lim, neginf=-lim)


TRACE = {}


def run(tU, tA, trace=False):
    B = Wg.shape[0]
    m = cp.zeros((B, W_), dtype=cp.float32)
    C = cp.broadcast_to(cp.eye(W_, dtype=cp.float32), (B, W_, W_)).copy()
    for i in range(DEPTH):
        w = Wg[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w)
        deep = i > KSIG
        if deep:
            cpm = cpm * KAP
        s2 = cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w)
        t = mp / s
        if deep:
            trC = s2.sum(1)
            nm2 = (mp * mp).sum(1)
            env = {
                "t": t, "one": 1.0, "dep": float(i) / 31.0,
                "fr": (nm2 / cp.maximum(nm2 + trC, 1e-30))[:, None],
                "mt": (cp.abs(t).mean(1) / 4.0)[:, None],
            }
            if trace and i in (9, 16, 24, 31):
                # 🔑 РОЗКИД МІЖ МЕРЕЖАМИ — головне число: якщо він малий, листок
                #    є функцією ГЛИБИНИ і дублює `dep`, тобто нової інформації нема
                TRACE[i] = tuple(
                    (float(env[k].mean()), float(env[k].std()))
                    for k in ("fr", "mt"))
            u = _san(ev(tU, env), 40.0, 0.0)
            A = _san(ev(tA, env), 20.0, 1.0)
        else:
            u, A = t, 1.0
        Ph = ndtr(u)
        ph = cp.exp(-0.5 * u * u) / SQ2PI
        m_new = mp * Ph + A * (s * ph)
        pv = ph / s
        C = (cpm * (Ph[:, :, None] * Ph[:, None, :])
             + (0.5 * (cpm * cpm)) * (pv[:, :, None] * pv[:, None, :]))
        C[:, IDX, IDX] = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        m = m_new
    M = cp.nan_to_num(m.astype(cp.float64))
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


T_ID = ("L", "t")
T_ONE = ("K", 1.0)
T_GAM = ("B", "mul", ("K", 1.048), ("L", "t"))

t0 = time.time()
e_base = run(T_ID, T_ONE, trace=True)
e_gam = run(T_GAM, T_ONE)
dt = time.time() - t0
print(f"\n🔒 ПЛОМБИ ({dt/2:.2f} с/прогін)")
print(f"{'база u=t, A=1':>22} TRAIN {100*float(e_base[IA].mean()):.4f}%  "
      f"TEST {100*float(e_base[IB].mean()):.4f}%")
print(f"{'sub127f u=1.048t':>22} TRAIN {100*float(e_gam[IA].mean()):.4f}%  "
      f"TEST {100*float(e_gam[IB].mean()):.4f}%")
print("режимні листки: середнє ± розкид МІЖ МЕРЕЖАМИ (розкид = чи є інформація)")
for i in sorted(TRACE):
    (frm, frs), (mtm, mts) = TRACE[i]
    print(f"   шар {i:2d}:  fr={frm:.4f}±{frs:.4f} (cv {frs/max(frm,1e-9):.3%})"
          f"   mt={mtm:.4f}±{mts:.4f} (cv {mts/max(mtm,1e-9):.3%})")
GAM_TR = float(e_gam[IA].mean())
GAM_TE = float(e_gam[IB].mean())
BASE_TE = float(e_base[IB].mean())
print(flush=True)


def fit(ind):
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
       (("B", "mul", ("B", "add", ("K", 1.0), ("L", "fr")), ("L", "t")), T_ONE),
       (("B", "mul", ("B", "add", ("K", 1.0), ("L", "dep")), ("L", "t")), T_ONE)]
while len(pop) < POP:
    pop.append((rnd(), T_ONE) if random.random() < 0.5 else (rnd(), rnd()))

scored = [fit(p) + (p,) for p in pop]
print(f"--- ЕВОЛЮЦІЯ (штраф {STAB:.0e}/вузол) ---", flush=True)
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
print(f"\n--- ЕЛІТА: TRAIN vs TEST ---")
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
