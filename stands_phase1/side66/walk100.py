"""walk100 — ПРОЙТИСЬ ПО ШАРАХ І ПОДИВИТИСЬ НА 256 ЧИСЕЛ. ІДЕЯ ЮЗЕРА (С100).

За 99 сесій ми жодного разу не дивились на ТРАЄКТОРІЮ похибки нейрон за
нейроном, шар за шаром. Міряли агрегати. Юзер: «женеш семпл і дивишся, там
всього лиш 256 чисел на кожному слої».

СПОСІБ: одна мережа, МК з великим N поза пісочницею (нема ні бюджету, ні
заборон), паралельно — наше замикання. На КОЖНОМУ шарі кладемо поруч:
   істина:    m, s, γ₃, γ₄, n_eff, частка живих, KS до гауса
   замикання: m̂, ŝ
   похибка:   b = m̂ − m  (256 чисел), і її розклад за С70
      b_ℓ = Φ(t_ℓ)·(b_{ℓ−1} @ W_ℓ)  +  φ(t_ℓ)·δs_ℓ  +  e_лок,ℓ
            └─ ПЕРЕНОС ─┘              └─ через σ ─┘    └─ народжена тут ─┘

ПИТАННЯ, ЗАРАДИ ЯКОГО ЦЕ РОБИТЬСЯ: чи є на шарі понейронна величина, яка
передбачає e_лок і обчислювана з W детерміновано?

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) |corr(e_лок, |t|)| > 0.2 хоч на одному глибокому шарі         70%
 (б) після зняття масштабу залишок ~ шум (усі |corr| < 0.15)       60%
 (в) γ₃,γ₄ проєкцій на плато вже до шару 8                         80%
 (г) знайдеться понейронна детермінована ознака (|corr| > 0.3)     20%
 (д) перенос дасть 0.90..0.96 частки, як у С70 (recon.py)          75%

🔒 ПЛОМБИ:
 (1) шар 0: вхід строго гаусів ⟹ похибка замикання мусить бути ~0 (машинна)
 (2) розклад С70 мусить замикатись: ‖b − (перенос+δs+e_лок)‖/‖b‖ < 1e-4
 (3) частка t>0 ≈ 0.491 на кожному шарі (записано в корпусі)
 (4) шум істини: половини вибірки мусять дати те саме з точністю ~1/√N
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
W_ = 256
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")

NET = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 32768          # пар (антитетично)
SHOW = [0, 1, 2, 4, 8, 16, 24, 30, 31]


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def load_net(idx):
    f = sorted(glob.glob(SRC + "/data-*.arrow"))[0]
    tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    x = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, W_, W_)
    y = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(y.type):
        y = y.flatten()
    FM = np.asarray(y).reshape(tb.num_rows, W_)
    return ([np.ascontiguousarray(Wall[idx][L], dtype=np.float64)
             for L in range(DEPTH)], FM[idx].astype(np.float64))


def corr(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    d = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / d if d > 1e-300 else 0.0


W, truth = load_net(NET)
rng = np.random.default_rng(777 + NET)
z = rng.standard_normal((NS, W_))
A = np.concatenate([z, -z], axis=0)                            # антитетично
N = A.shape[0]
print(f"walk100 · мережа {NET} · {N} семплів (антитетично) · "
      f"{DEPTH} шарів × {W_} нейронів", flush=True)

# ── стан замикання (як у sub99, без обрізання, fp64) ────────────────────────
m_h = np.zeros(W_)
C_h = np.eye(W_)
b_prev = np.zeros(W_)                                          # похибка m попер. шару

rows = []
per_layer = {}

for L in range(DEPTH):
    w = W[L]

    # ─────────── ІСТИНА (МК) ───────────
    P = A @ w                                                  # передактивації
    m_t = P.mean(0)
    Pc = P - m_t
    s_t = np.sqrt((Pc * Pc).mean(0))
    g3 = (Pc ** 3).mean(0) / np.maximum(s_t ** 3, 1e-300)
    g4 = (Pc ** 4).mean(0) / np.maximum(s_t ** 4, 1e-300) - 3.0
    frac_pos = float((m_t > 0).mean())
    alive = float((P.max(0) >= 0).mean())

    # n_eff: ефективна ширина ЦГТ — участь внесків w_ja²·Var(a_j) у Var(p_a)
    va = np.ones(W_) if L == 0 else var_prev_
    contrib = (w * w) * va[:, None]                            # (in, out)
    tot = contrib.sum(0)
    n_eff = (tot ** 2) / np.maximum((contrib ** 2).sum(0), 1e-300)

    # ─────────── ЗАМИКАННЯ ───────────
    cp = w.T @ C_h @ w
    s_h = np.sqrt(np.maximum(np.diagonal(cp), 1e-20))
    mp = m_h @ w
    t = mp / s_h
    ph, Ph = phi(t), ndtr(t)
    m_new = mp * Ph + s_h * ph

    # ─────────── ПОХИБКА І ЇЇ РОЗКЛАД (С70) ───────────
    # істинне середнє АКТИВАЦІЙ цього шару
    Areal = np.maximum(P, 0.0)
    ma_t = Areal.mean(0)
    b = m_new - ma_t                                           # 256 чисел

    # три доданки
    transp = Ph * (b_prev @ w)
    d_s = s_h - s_t
    src_s = ph * d_s
    e_loc = b - transp - src_s

    nb = np.linalg.norm(b)
    fr = (np.linalg.norm(transp) / nb if nb > 1e-300 else 0.0,
          np.linalg.norm(src_s) / nb if nb > 1e-300 else 0.0,
          np.linalg.norm(e_loc) / nb if nb > 1e-300 else 0.0)

    # відносні похибки каналів
    rel_m = float(np.linalg.norm(b) / max(np.linalg.norm(ma_t), 1e-300))
    rel_s = float(np.linalg.norm(d_s) / max(np.linalg.norm(s_t), 1e-300))

    # ── ЧИ ПЕРЕДБАЧУВАНА e_лок ПОНЕЙРОННО? ──
    tt = np.abs(t)
    # знімаємо спільний масштаб (те, що робить _ALPHA) з e_лок відносно m
    scale = float((e_loc @ m_new) / max(float(m_new @ m_new), 1e-300))
    e_res = e_loc - scale * m_new
    feats = {"|t|": tt, "s": s_h, "n_eff": n_eff, "g3": g3, "g4": g4,
             "phi(t)": ph, "1/n_eff": 1.0 / np.maximum(n_eff, 1e-9)}
    cr = {k: corr(v, e_loc) for k, v in feats.items()}
    cr_res = {k: corr(v, e_res) for k, v in feats.items()}

    rows.append((L, float(np.median(tt)), frac_pos, alive,
                 float(np.median(g3)), float(np.median(g4)),
                 float(np.median(n_eff)), rel_m, rel_s, fr,
                 abs(scale), cr, cr_res))
    per_layer[L] = dict(t=t, s_h=s_h, s_t=s_t, m_h=m_new, m_t=ma_t, b=b,
                        e_loc=e_loc, e_res=e_res, g3=g3, g4=g4, n_eff=n_eff,
                        transp=transp, src_s=src_s)

    if L == DEPTH - 1:
        break

    # оновлення істини і замикання
    var_prev_ = Areal.var(0)
    A = Areal
    Ca = w.T @ C_h @ w
    Ca = Ca * np.outer(Ph, Ph) + (0.5 * (Ca * Ca)) * np.outer(ph / s_h,
                                                             ph / s_h)
    d = (mp * mp + s_h * s_h) * Ph + (mp * s_h) * ph - m_new * m_new
    C_h = Ca + np.diag(d - np.diagonal(Ca))
    m_h = m_new
    b_prev = b

# ═══════════════════════ ТАБЛИЦЯ ═══════════════════════
print("\n" + "=" * 118)
print(f"{'шар':>3s}{'мед|t|':>8s}{'m>0':>7s}{'живих':>7s}{'мед γ₃':>9s}"
      f"{'мед γ₄':>9s}{'n_eff':>8s}{'‖b‖/‖m‖':>10s}{'δs/s':>9s}"
      f"{'перенос':>9s}{'δs':>7s}{'e_лок':>8s}{'|масшт|':>9s}")
print("=" * 118)
for (L, mt, fp, al, g3m, g4m, ne, rm, rs, fr, sc, cr, crr) in rows:
    print(f"{L:>3d}{mt:>8.3f}{fp:>7.3f}{al:>7.3f}{g3m:>9.4f}{g4m:>9.4f}"
          f"{ne:>8.1f}{100*rm:>9.4f}%{100*rs:>8.4f}%"
          f"{fr[0]:>9.3f}{fr[1]:>7.3f}{fr[2]:>8.3f}{sc:>9.2e}")

# ═══════════════════════ ПЛОМБИ ═══════════════════════
print("\n🔒 ПЛОМБИ")
b0 = per_layer[0]["b"]
print(f"  (1) шар 0 (вхід строго гаусів): ‖b‖/‖m‖ = "
      f"{100*np.linalg.norm(b0)/np.linalg.norm(per_layer[0]['m_t']):.5f}%"
      f"   мусить бути ~шум МК ({100/math.sqrt(N):.4f}%)")
Lc = 16
pl = per_layer[Lc]
rec = pl["transp"] + pl["src_s"] + pl["e_loc"]
print(f"  (2) розклад С70 замикається (шар {Lc}): "
      f"{np.linalg.norm(pl['b']-rec)/max(np.linalg.norm(pl['b']),1e-300):.2e}")
print(f"  (3) частка m>0 по шарах 8..31: "
      f"{np.mean([r[2] for r in rows[8:]]):.4f}  (корпус: 0.491)")

# ═══════════════════════ КОРЕЛЯЦІЇ ПОНЕЙРОННО ═══════════════════════
print("\n📊 corr(ознака, e_лок) — СИРА  |  після зняття масштабу")
keys = ["|t|", "s", "n_eff", "g3", "g4", "phi(t)", "1/n_eff"]
print(f"{'шар':>3s}" + "".join(f"{k:>10s}" for k in keys)
      + "   ||" + "".join(f"{k:>10s}" for k in keys))
for r in rows:
    if r[0] not in SHOW:
        continue
    print(f"{r[0]:>3d}" + "".join(f"{r[11][k]:>10.3f}" for k in keys)
          + "   ||" + "".join(f"{r[12][k]:>10.3f}" for k in keys))

# ═══════════════════════ ХТО ВИНЕН ═══════════════════════
print("\n🔎 ТОП-8 НЕЙРОНІВ ЗА |e_лок| НА ШАРІ 31 (і їхні числа)")
pl = per_layer[DEPTH - 1]
idx = np.argsort(-np.abs(pl["e_loc"]))[:8]
print(f"{'j':>5s}{'e_лок':>12s}{'b':>12s}{'t':>9s}{'s_h':>11s}"
      f"{'δs/s':>10s}{'γ₃':>9s}{'γ₄':>9s}{'n_eff':>8s}")
for j in idx:
    print(f"{j:>5d}{pl['e_loc'][j]:>12.4e}{pl['b'][j]:>12.4e}"
          f"{pl['t'][j]:>9.3f}{pl['s_h'][j]:>11.4e}"
          f"{100*(pl['s_h'][j]/pl['s_t'][j]-1):>9.3f}%"
          f"{pl['g3'][j]:>9.3f}{pl['g4'][j]:>9.3f}{pl['n_eff'][j]:>8.1f}")

# концентрація: скільки нейронів дають половину енергії e_лок
e2 = np.sort(pl["e_loc"] ** 2)[::-1]
cum = np.cumsum(e2) / e2.sum()
print(f"\n  нейронів на 50% енергії e_лок: {int(np.searchsorted(cum,0.5))+1}"
      f" / {W_}   (рівномірно було б {W_//2})")
print(f"  нейронів на 90%:                {int(np.searchsorted(cum,0.9))+1}")

# ═══════════════════════ ФІНАЛ ═══════════════════════
mfin = per_layer[DEPTH - 1]["m_h"]
for a in (1.0, 1.0 - 0.00751):
    print(f"\nвихід (α={a:.5f}): MSE проти істини банку = "
          f"{float(((a*mfin-truth)**2).mean()):.6e}")
aopt = float((mfin @ truth) / (mfin @ mfin))
print(f"  оптимальний масштаб a = {aopt:.6f} ⟹ "
      f"MSE = {float(((aopt*mfin-truth)**2).mean()):.6e}")

np.savez_compressed(f"/mnt/win_d/llm/White-Box/relay/walk100_net{NET}.npz",
                    **{f"{k}_{L}": v for L, d in per_layer.items()
                       for k, v in d.items()})
print(f"\nповні масиви → relay/walk100_net{NET}.npz")
