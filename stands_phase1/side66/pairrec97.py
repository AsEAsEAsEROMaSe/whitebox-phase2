"""pairrec97 — ЧИ ВИЖИВАЄ ПАРНА ПОПРАВКА `λ₂₁` У 31-КРОКОВІЙ РЕКУРСІЇ.
СТЕЛЯ КАНАЛУ: ВПОРСКУЄМО ІСТИННИЙ `λ₂₁` (N=1e8) НА КОЖНОМУ КРОЦІ.

════════════════ ЧОМУ САМЕ ОРАКУЛ ════════════════
Одноступенево поправка знімає 84–92% похибки позадіагоналі, і вільний α не
забирає нічого (α=0.9999). Але розрив один-крок/рекурсія в цьому проєкті
зміряно ТРИЧІ: ×8.5 (С65) · ×4.88→×0.83 (С95) · ×4.53→×0.61 (С94), і закон
С83 каже «в рекурсії ВСІ поправки шкодять».
⟹ ОРАКУЛ — найдешевший вирішальний тест: якщо гине він, гине все під ним
(будь-яка доставка може бути лише ГІРШОЮ), і гілка закрита за одну годину.

════════════════ ЩО РАХУЄМО ════════════════
Стан = `(m, C)` активацій. Крок: `mp = Wᵀm`, `Cp = WᵀCW`, тоді ReLU-моменти.
```
ГАУС  : m = s·ψ(t) · E[a_j a_k] = s_j s_k·Прайс(t_j,t_k,ρ) · E[a²]=s²[(1+t²)Φ+tφ]
+κ₃   : те саме + (3λ_jjk B_j + 3λ_jkk B_k)/6 у позадіагональ
        + γφ/3 у діагональ · −(γ/6)tφ у `m`  ← вимикається окремо (PAIRONLY)
```
`λ₂₁`, `γ` беруться ІСТИННІ, з `pre_M21`/`pre_m3` датасету для ЦЬОГО ж шару.
Ціль — `mean[31]` (N=1e8). Метрика — енергетично зважена відносна RMS.

🚨 ВАЖЛИВЕ ЗАСТЕРЕЖЕННЯ ПРО ОРАКУЛ: істинні `λ₂₁` належать ІСТИННІЙ
траєкторії, а прикладаються до НАКОПИЧЕНОЇ (закон С74 про перенос). Це не
робить тест некоректним — це і є визначення стелі «локальної» поправки.

════════════════ ПЛОМБИ ════════════════
 P1 🔒 λ=0 ⟹ РІВНО база (побітово)
 P2 🔒 шар 0: вхід точно гаусів ⟹ κ₃=0 ⟹ поправка на 1-му кроці = 0
 P3 🔒 контроль «ЧУЖА λ₂₁» (з іншого шару) мусить погіршити
 P4 🔒 λ-скан: своя поправка мусить мати оптимум при λ>0

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) оракул у рекурсії дасть КРАЩЕ за базу                     60%
 (б) дасть > ×2                                                25%
 (в) дасть ГІРШЕ за базу (як усі попередні поправки)           30%
 (г) вільний α з'їсть менше половини виграшу (бо одноступенево
     α=0.9999, поправка є формою)                              55%
 (д) PAIRONLY (без поправки до m) буде КРАЩЕ за повну          50%
"""
import glob
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
SCR = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
NPAN, NNOD = int(sys.argv[1]) if len(sys.argv) > 1 else 30, 24


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def psi(x):
    return phi(x) + x * ndtr(x)


def gl(a, b, npan, nnod):
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    return ((0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel(),
            (0.5 * (hi - lo) * ws[None, :]).ravel())


ZQ, WQ = gl(0.0, 40.0, NPAN, NNOD)
print(f"pairrec97 · квадратура {ZQ.size} вузлів", flush=True)


def price_mat(t, rho, chunk=4096):
    """матриця E[(x+t_j)₊(y+t_k)₊] для всіх пар (векторно, спільні вузли)."""
    n = t.size
    jj = np.repeat(np.arange(n), n)
    kk = np.tile(np.arange(n), n)
    tj, tk = t[jj], t[kk]
    rr = rho.ravel()
    out = np.empty(n * n)
    q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-12))
    for o in range(0, n * n, chunk):
        sl = slice(o, min(o + chunk, n * n))
        x = ZQ[None, :] - tj[sl, None]
        u = (rr[sl, None] * x + tk[sl, None]) / q[sl, None]
        out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                   * psi(u)).sum(1)
    return out.reshape(n, n)


def corr_mat(t, rho, ljjk, ljkk):
    """поправка κ₃ до E[a_j a_k], лише крос-члени (матрично)."""
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    tj, tk = t[:, None], t[None, :]
    uk = (tk - rho * tj) / q
    uj = (tj - rho * tk) / q
    Bj = phi(tj) * ndtr(uk)
    Bk = phi(tk) * ndtr(uj)
    return (3.0 * ljjk * Bj + 3.0 * ljkk * Bk) / 6.0


# ─────────────── дані ───────────────
d = np.load(f"{SCR}/mlp_00000.npz")
gidx = int(d["global_index"])
W = None
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    ids = t.column("mlp_id").to_pylist()
    if gidx not in ids:
        continue
    i = ids.index(gidx)
    x = t.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    W = np.asarray(x).reshape(t.num_rows, 32, 256, 256)[i].astype(np.float64)
    break
assert W is not None
TRUTH = d["mean"][31].astype(np.float64)

# істинні стандартизовані кумулянти передактивацій, пошарово
LJ, LK, GG = [], [], []
for L in range(32):
    m = d["pre_mean"][L].astype(np.float64)
    m2 = d["pre_m2"][L].astype(np.float64)
    M11 = d["pre_M11"][L].astype(np.float64)
    M21 = d["pre_M21"][L].astype(np.float64)
    m3 = d["pre_m3"][L].astype(np.float64)
    s = np.sqrt(np.maximum(m2 - m * m, 1e-300))
    sj, sk = s[:, None], s[None, :]
    k3jjk = (M21 - m[None, :] * m2[:, None] - 2 * m[:, None] * M11
             + 2 * (m ** 2)[:, None] * m[None, :])
    LJ.append(np.where(s[:, None] * s[None, :] > 1e-12,
                       k3jjk / (sj * sj * sk), 0.0))
    LK.append(LJ[-1].T.copy())
    GG.append(np.where(s > 1e-9, (m3 - 3 * m * m2 + 2 * m ** 3) / s ** 3, 0.0))


def run(lam=0.0, pair_only=False, foreign=0):
    """31-крокова рекурсія. lam — сила поправки. foreign — зсув шару λ (контроль)."""
    m = np.zeros(256)
    C = np.eye(256)
    for L in range(32):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        s2 = np.maximum(np.diag(Cp), 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.9999, 0.9999)
        P = price_mat(t, rho)
        if lam != 0.0:
            src = (L + foreign) % 32
            P = P + lam * corr_mat(t, rho, LJ[src], LK[src])
        M2 = P * (s[:, None] * s[None, :])
        g = GG[(L + foreign) % 32] if lam != 0.0 else np.zeros(256)
        d2 = s2 * ((1 + t * t) * ndtr(t) + t * phi(t) + lam * g * phi(t) / 3.0)
        ma = s * psi(t)
        if lam != 0.0 and not pair_only:
            ma = ma - lam * s * g / 6.0 * t * phi(t)
        np.fill_diagonal(M2, d2)
        C = M2 - np.outer(ma, ma)
        C = 0.5 * (C + C.T)
        m = ma
    return m


def err(m):
    return float(np.sqrt(((m - TRUTH) ** 2).mean())
                 / np.sqrt((TRUTH ** 2).mean()))


def err_a(m):
    a = float((m * TRUTH).sum() / max((m * m).sum(), 1e-300))
    return float(np.sqrt(((a * m - TRUTH) ** 2).mean())
                 / np.sqrt((TRUTH ** 2).mean())), a


t0 = time.time()
base = run(0.0)
eb = err(base)
eba, ab = err_a(base)
print(f"\n🔒 P1 база: похибка {100*eb:.4f}% · після α {100*eba:.4f}% "
      f"(α={ab:.4f}) · {time.time()-t0:.0f} с", flush=True)

print("\n" + "=" * 92)
print("λ-СКАН ОРАКУЛЬНОЇ ПАРНОЇ ПОПРАВКИ У 31-КРОКОВІЙ РЕКУРСІЇ")
print("=" * 92)
print(f"{'λ':>6s}{'режим':>12s}{'похибка':>11s}{'× до бази':>11s}"
      f"{'після α':>11s}{'× після α':>12s}{'α':>9s}")
for lam in (0.25, 0.5, 1.0, 1.5):
    for po in (False, True):
        mm = run(lam, pair_only=po)
        e = err(mm)
        ea, aa = err_a(mm)
        print(f"{lam:>6.2f}{'ЛИШЕ ПАРА' if po else 'ПАРА+m+діаг':>12s}"
              f"{100*e:>10.4f}%{eb/e:>11.4f}{100*ea:>10.4f}%"
              f"{eba/ea:>12.4f}{aa:>9.4f}", flush=True)

print("\n🔒 P3 КОНТРОЛЬ «ЧУЖА λ₂₁» (кумулянти зі зсунутого шару):")
for sh in (7, 16):
    mm = run(1.0, pair_only=True, foreign=sh)
    e = err(mm)
    ea, _ = err_a(mm)
    print(f"  зсув {sh:>2d}: похибка {100*e:>9.4f}%  × до бази {eb/e:>7.4f}"
          f"  · після α × {eba/ea:>7.4f}")
print(f"\n  час {time.time()-t0:.0f} с")
