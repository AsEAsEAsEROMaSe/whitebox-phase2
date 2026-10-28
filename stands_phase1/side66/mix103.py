"""mix103 — КРИВА «ТОЧНІСТЬ (m,C) → ЧИ ВМИКАЄТЬСЯ γ₄». ДІРКА МІЖ ДВОМА КІНЦЯМИ

ЩО ВІДОМО І ЧОГО БРАКУЄ (перевірено грепом по 495 файлах памʼяті):
    edg   γ₄ на НАШИХ (m,C)      ×1.02     ← вимкнений       [orac101e, С101]
    edgm  γ₄ на ІСТИННИХ (m,C)   ×22.78    ← увімкнений      [orac101e, С101]
    між ними — ЖОДНОЇ виміряної точки, а саме там лежить відповідь на питання
    ЮЗЕРА «чи досяжний 1e-8».

Сито С99 міряло «покращити (m,C) у ×k → бал» БЕЗ γ₄ (×2→1.19 … ∞→2.52).
`edgm` міряв γ₄ ЛИШЕ на ідеалі. Комбінації не міряв ніхто.

ЩО РОБИМО: підставляємо СУМІШ на кожному кроці рекурсії
    m ← (1−μ)·m_наше + μ·m_істинне       C ← (1−μ)·C_наше + μ·C_істинне
для μ = 0 … 1, і для КОЖНОГО μ шукаємо СВІЙ оптимальний κ і СВІЙ коефіцієнт γ.
🔑 Своя калібровка для кожної точки обов'язкова — записаний закон С100:
   «калібрувальні константи НЕ ПЕРЕНОСЯТЬСЯ між конфігураціями».

🔒 ДВІ ВБУДОВАНІ ПЛОМБИ: μ=0 мусить відтворити `base`/`edg` (0.507% / ×1.02),
   μ=1 — `allm`/`edgm` (0.0913% / 8.16e-09). Якщо кінці не сходяться з
   записаним — крива нічого не варта і числу вірити не можна.

ЩО ЦЕ ВИРІШУЄ:
   крива ПЛАВНА  ⟹ «покращити моменти вдвічі» вже щось дає ⟹ напрямок живий
   крива РІЗКА   ⟹ γ₄ вмикається лише біля ідеалу ⟹ напрямок мертвий, і ми
                    перестаємо витрачати на нього час до 19.09

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ (щоб не тлумачити заднім числом):
   - крива буде РІЗКОЮ (виграш γ₄ понад ×2 лише при μ > 0.8):   60%
   - при μ=0.5 сумарний виграш (моменти+γ₄) буде ×2.0…×3.5:     55%
   - точка, де бал перетинає 1e-8, лежить при μ > 0.9:          65%
   ⟹ найімовірніший підсумок: «треба майже ідеальні моменти», і це поганий
      результат, але він вартий 20 хв, бо закриває або відкриває напрямок.
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
V_MC = 1.9227e-06
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NPAIRS = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
CHUNK = 8192
KSIG = 8
MIX = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]
KGRID = [0.990, 0.9955, 0.9985, 1.0, 1.005]
CGRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
OUT = "/home/trim/wblogs/mix103.json"

if cp.cuda.runtime.getDeviceCount() != 1:
    print("🚨 видно не рівно одну карту — пінування зламане")
    sys.exit(2)
state = json.load(open(OUT)) if os.path.exists(OUT) else {}


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
print(f"mix103 · {NNET} мереж × {N} семплів · μ {MIX}", flush=True)


def closure(W, kap, Ct, mt, G3, G4, mu, cg):
    """mu=0 — чисто наше; mu=1 — чистий оракул; між ними — лінійна суміш."""
    m = cp.zeros(W_, dtype=cp.float64)
    C = cp.eye(W_, dtype=cp.float64)
    for i in range(DEPTH):
        w = W[i]
        cpm = w.T @ C @ w
        if i > KSIG:
            cpm = cpm * kap
        s = cp.sqrt(cp.maximum(cp.diagonal(cpm), 1e-20))
        mp = m @ w
        t = mp / s
        ph, Ph = phi(t), ndtr(t)
        m = mp * Ph + s * ph
        if cg != 0.0:
            m = m + cg * (-s * (G3[i] / 6.0) * t * ph
                          + s * (G4[i] / 24.0) * (t * t - 1.0) * ph)
        if i == DEPTH - 1:
            break
        Co = (cpm * cp.outer(Ph, Ph)
              + (0.5 * (cpm * cpm)) * cp.outer(ph / s, ph / s))
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        Co = Co + cp.diag(d - cp.diagonal(Co))
        # 🔑 ПОХИБКА МОМЕНТІВ ДО підтягування — це і є «наскільки точний метод»
        #    на цьому шарі. Без неї вісь μ не перекладається в точність, бо
        #    підтягування щокроку ГЛУШИТЬ НАКОПИЧЕННЯ, а не лише зменшує похибку.
        if i == DEPTH - 2:
            em = float(cp.linalg.norm(m - mt[i])
                       / cp.maximum(cp.linalg.norm(mt[i]), 1e-300))
            ec = float(cp.linalg.norm(Co - Ct[i])
                       / cp.maximum(cp.linalg.norm(Ct[i]), 1e-300))
        if mu > 0.0:
            C = (1.0 - mu) * Co + mu * Ct[i]
            m = (1.0 - mu) * m + mu * mt[i]
        else:
            C = Co
    return m, em, ec


def one_net(net):
    Wg = [cp.asarray(np.ascontiguousarray(Wall[net][L], dtype=np.float32))
          for L in range(DEPTH)]
    W64 = [w.astype(cp.float64) for w in Wg]
    T = cp.asarray(FM[net].astype(np.float64))
    rng = cp.random.default_rng(777 + net)
    A1 = cp.zeros((DEPTH - 1, W_), dtype=cp.float64)
    A2 = cp.zeros((DEPTH - 1, W_, W_), dtype=cp.float64)
    P1 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P2 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P3 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    P4 = cp.zeros((DEPTH, W_), dtype=cp.float64)
    done = 0
    while done < NPAIRS:
        nb = min(CHUNK, NPAIRS - done)
        z = rng.standard_normal((nb, W_), dtype=cp.float32)
        a = cp.concatenate([z, -z], axis=0)          # антитетичні пари
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
    mu_ = P1 / N
    v = cp.maximum(P2 / N - mu_ * mu_, 1e-30)
    sd = cp.sqrt(v)
    m3 = P3 / N - 3.0 * mu_ * (P2 / N) + 2.0 * mu_ ** 3
    m4 = (P4 / N - 4.0 * mu_ * (P3 / N) + 6.0 * mu_ * mu_ * (P2 / N)
          - 3.0 * mu_ ** 4)
    G3 = m3 / sd ** 3
    G4 = m4 / sd ** 4 - 3.0
    out = {"_TT": float(T @ T), "_g4": [float(cp.median(G4[i]))
                                        for i in (0, 8, 16, 31)]}
    for mx in MIX:
        tab = {}
        for k in KGRID:
            for cg in CGRID:
                mm, em, ec = closure(W64, k, Ct, mt, G3, G4, mx, cg)
                if bool(cp.all(cp.isfinite(mm))):
                    tab[f"{k}|{cg}"] = [float(mm @ mm), float(mm @ T), em, ec]
        out[f"mu{mx}"] = tab
    del A1, A2, Ct, mt, W64, Wg
    cp.get_default_memory_pool().free_all_blocks()
    return out


for net in range(NNET):
    if str(net) in state:
        continue
    t0 = time.time()
    state[str(net)] = one_net(net)
    save()
    print(f"  ✅ мережа {net} ({time.time()-t0:.1f} с) · "
          f"γ₄(мед) шари 0/8/16/31 {[round(x,3) for x in state[str(net)]['_g4']]}",
          flush=True)

# ------------------------------------------------------------------ аналіз
nets = sorted(state, key=int)
MT2 = sum(state[n]["_TT"] for n in nets) / len(nets) / W_


def cos_err(mode, k, cg):
    """косинусна похибка = після ОПТИМАЛЬНОГО масштабу (С100: ціль косинусна)"""
    tot = 0.0
    for n in nets:
        key = f"{k}|{cg}"
        if key not in state[n][mode]:
            return None
        mm, mT = state[n][mode][key][:2]
        tot += state[n]["_TT"] - (mT * mT) / max(mm, 1e-300)
    return math.sqrt(max(tot / len(nets) / W_, 0.0) / MT2)


print(f"\n{'μ':>6} {'κ*':>7} {'c*γ':>5} {'похибка':>9} {'×база':>7} "
      f"{'γ₄ дає':>7} {'|Δm|/m':>8} {'|ΔC|/C':>8} {'БАЛ':>10}")
print("  (|Δm|,|ΔC| — похибка НАШИХ моментів на шарі 30 ДО підтягування)")
rows = []
for mx in MIX:
    mode = f"mu{mx}"
    bn = bg = None
    for k in KGRID:
        for cg in CGRID:
            e = cos_err(mode, k, cg)
            if e is None:
                continue
            if bn is None or e < bn[0]:
                bn = (e, k, cg)
            if cg == 0.0 and (bg is None or e < bg[0]):
                bg = (e, k)
    if bn is None:
        continue
    e, k, cg = bn
    E2 = MT2 * e * e * W_ / W_
    E2 = 0.9097 * e * e
    mse = V_MC * E2 / (V_MC + E2)
    rows.append((mx, e, bg[0], mse))
    em = sum(state[n][mode][f"{k}|{cg}"][2] for n in nets) / len(nets)
    ec = sum(state[n][mode][f"{k}|{cg}"][3] for n in nets) / len(nets)
    print(f"{mx:>6.2f} {k:>7.4f} {cg:>5.2f} {100*e:>8.4f}% "
          f"{rows[0][1]/e:>7.3f} {bg[0]/e:>7.3f} {100*em:>7.3f}% {100*ec:>7.2f}% "
          f"{0.1004*mse:>10.3e}")

print("\n🔒 ПЛОМБИ: μ=0 має дати ≈0.507% (base) · μ=1 ≈0.0913% (allm)")
print("   і γ₄ при μ=0 має дати ×1.02, при μ=1 ×22.8 у БАЛІ")
if rows:
    b0 = 0.1004 * rows[0][3]
    print(f"\nбал при μ=0: {b0:.3e}   ціль 1.0e-08")
    for mx, e, e0, mse in rows:
        print(f"  μ={mx:.2f}: бал {0.1004*mse:.3e}  "
              f"×{b0/(0.1004*mse):.2f} понад μ=0"
              f"{'   ✅ ЦІЛЬ ДОСЯГНУТА' if 0.1004*mse <= 1e-8 else ''}")
