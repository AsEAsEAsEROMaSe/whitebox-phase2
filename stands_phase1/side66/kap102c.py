"""kap102c — ЧИ Є `κ` І `gam` ОДНАКОВИМИ ДЛЯ ВСІХ МЕРЕЖ?

ПИТАННЯ, ЯКОГО МИ НЕ СТАВИЛИ ЗА 102 СЕСІЇ. `κ` і `gam` підбирались як ГЛОБАЛЬНІ
константи — по всіх мережах відразу. Але косинусна ціль (С100) показала, що
масштаб вільний ПО МЕРЕЖІ; ніхто не питав того самого про ручки в рекурсії.

ЩО МІРЯЄМО (сітка 9×9, для КОЖНОЇ мережі окремо):
  1. глобальний оптимум (κ*, gam*) — те, що ми маємо зараз
  2. ОРАКУЛЬНИЙ по-мережний min — стеля будь-якої схеми калібровки по мережі
  3. розкид (κ_i, gam_i) між мережами — чи взагалі є що калібрувати
  4. чи корелює κ_i з величинами, які замикання знає ЗАДАРМА:
       fr31 = ‖m‖²/(‖m‖²+tr C) на шарі 31 · mt31 = mean|t| на шарі 31
       nrm  = ‖m₃₁‖ · trc = tr C₃₁
     🔒 і КОНТРОЛЬ «ЧУЖА»: ті самі ознаки, переставлені між мережами

ЧОМУ ЦЕ НЕ ПОВТОР `_W_BLEND`-ПО-МЕРЕЖІ (С101-б, стеля ×1.0293, реалізовно 0):
там ціль була ВІДНОШЕННЯМ двох величин, і проксі скорочувалась у ньому. Тут
ціль — параметр САМОЇ рекурсії, а не ваги злиття двох гілок.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  - оракульна стеля по мережі: ×1.05…×1.15 (бо в С101-б аналогічна стеля була
    ×1.029, а тут дві ручки замість однієї ваги)
  - шанс, що знайдеться безкоштовна ознака з |corr| > 0.5 понад контроль: ~25%
  - якщо стеля вийде < ×1.03 — питання закрите числом, і це теж відповідь
"""
import glob
import math
import sys

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
KAPS = [0.9955, 0.9965, 0.9975, 0.9985, 0.9995, 1.0005, 1.0015, 1.0030, 1.0050]
GAMS = [0.990, 1.005, 1.020, 1.035, 1.048, 1.060, 1.075, 1.090, 1.110]

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
FEAT = {}
print(f"kap102c · {NNET} мереж · сітка {len(KAPS)}×{len(GAMS)}", flush=True)


def run(kap, gam, feats=False):
    B = Wg.shape[0]
    m = cp.zeros((B, W_), dtype=cp.float32)
    C = cp.broadcast_to(cp.eye(W_, dtype=cp.float32), (B, W_, W_)).copy()
    for i in range(DEPTH):
        w = Wg[:, i]
        cpm = cp.matmul(cp.matmul(w.transpose(0, 2, 1), C), w)
        deep = i > KSIG
        if deep:
            cpm = cpm * kap
        s2 = cp.maximum(cp.diagonal(cpm, axis1=1, axis2=2), 1e-30)
        s = cp.sqrt(s2)
        mp = cp.einsum('bi,bij->bj', m, w)
        t = mp / s
        u = t * gam if deep else t
        Ph = ndtr(u)
        ph = cp.exp(-0.5 * u * u) / SQ2PI
        m_new = mp * Ph + s * ph
        pv = ph / s
        C = (cpm * (Ph[:, :, None] * Ph[:, None, :])
             + (0.5 * (cpm * cpm)) * (pv[:, :, None] * pv[:, None, :]))
        C[:, IDX, IDX] = (mp * mp + s2) * Ph + (mp * s) * ph - m_new * m_new
        m = m_new
        if feats and i == DEPTH - 1:
            trC = s2.sum(1)
            nm2 = (mp * mp).sum(1)
            FEAT["fr31"] = cp.asnumpy(nm2 / cp.maximum(nm2 + trC, 1e-30))
            FEAT["mt31"] = cp.asnumpy(cp.abs(t).mean(1))
            FEAT["nrm"] = cp.asnumpy(cp.sqrt((m_new * m_new).sum(1)))
            FEAT["trc"] = cp.asnumpy(cp.sqrt(trC))
    M = cp.nan_to_num(m.astype(cp.float64))
    num = (M * Tg).sum(1) ** 2
    den = (M * M).sum(1) * (Tg * Tg).sum(1)
    return cp.sqrt(cp.maximum(1.0 - num / cp.maximum(den, 1e-300), 0.0))


E = np.zeros((len(KAPS), len(GAMS), NNET))
for a, k in enumerate(KAPS):
    for b, g in enumerate(GAMS):
        E[a, b] = cp.asnumpy(run(k, g, feats=(a == 3 and b == 4)))
    print(f"  κ={k:.4f} готово", flush=True)

base = float(run(1.0, 1.0).mean())
gl = E.mean(2)
ia, ib = np.unravel_index(np.argmin(gl), gl.shape)
glob_e = gl[ia, ib]
flat = E.reshape(-1, NNET)
best_i = flat.argmin(0)
orac_e = flat.min(0).mean()
ka = np.array(KAPS)[best_i // len(GAMS)]
ga = np.array(GAMS)[best_i % len(GAMS)]

print(f"\n{'база (κ=1, gam=1)':>34} {100*base:.4f}%")
print(f"{'наш sub127f (0.9985, 1.048)':>34} {100*float(E[3, 4].mean()):.4f}%")
print(f"{'ГЛОБАЛЬНИЙ оптимум сітки':>34} {100*glob_e:.4f}%"
      f"   (κ={KAPS[ia]:.4f}, gam={GAMS[ib]:.3f})")
print(f"{'🔒 ОРАКУЛЬНИЙ по мережі (СТЕЛЯ)':>34} {100*orac_e:.4f}%"
      f"   ×{glob_e/orac_e:.4f} понад глобальний")
V, E2 = 1.9227e-06, 2.3385e-05
x = glob_e / orac_e
e2 = E2 / x ** 2
print(f"{'':>34} у балі ×{(V*E2/(V+E2))/(V*e2/(V+e2)):.4f}")

print(f"\nрозкид оптимумів між мережами:")
print(f"  κ_i : медіана {np.median(ka):.4f}  sd {ka.std():.4f}  "
      f"частки на краях сітки {np.mean((ka <= KAPS[0]) | (ka >= KAPS[-1])):.1%}")
print(f"  gam_i: медіана {np.median(ga):.4f}  sd {ga.std():.4f}  "
      f"частки на краях {np.mean((ga <= GAMS[0]) | (ga >= GAMS[-1])):.1%}")

print(f"\nкореляції оптимумів із БЕЗКОШТОВНИМИ ознаками (+ 🔒 контроль «чужа»):")
rng = np.random.default_rng(4242)
perm = rng.permutation(NNET)
for nm, v in FEAT.items():
    v = np.asarray(v, dtype=np.float64)
    for tag, y in (("κ_i", ka), ("gam_i", ga)):
        r = np.corrcoef(v, y)[0, 1]
        rc = np.corrcoef(v[perm], y)[0, 1]
        print(f"  corr({nm:5s}, {tag:5s}) = {r:+.4f}   🔒чужа {rc:+.4f}")
