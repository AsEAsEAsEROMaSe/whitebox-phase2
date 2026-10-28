"""decomp99 — З ЧОГО СКЛАДАЄТЬСЯ `E_i²`, ВЕЛИЧИНА, ЯКУ МИ ОПТИМІЗУЄМО.

За 99 сесій ми жодного разу не розклали похибку гілки замикання сабміту на
доданки — оптимізували агрегат наосліп. `[ЗМІРЯНО A, 50 мереж мінісплиту]`
щойно з'ясувалось, що вона **ортогональна** істинній похибці замикання
(`corr = −0.018`), тобто ми міряли не ту величину.

СПОСІБ: відтворити шлях `_closure_parts` сабміту в numpy ПОЗА пісочницею (там
немає ні бюджету, ні заборони на fp64) і вимикати по одному наближенню.

🔒 ПЛОМБА, БЕЗ ЯКОЇ ВИМІР НЕ РАХУЄТЬСЯ: варіант «усе як у сабміті» мусить дати
   середнє `2.338512e-05` (зміряно грейдером, `_W_BLEND=1.0`, ті самі 50 мереж).
   Не зійдеться — значить відтворення неправильне, і решта рядків нічого не варта.

КОМПОНЕНТИ, ЯКІ ВИМИКАЮТЬСЯ:
   fp32 → fp64            · шум арифметики самого оцінювача
   Мелер K=2 → квадратура · головне наближення позадіагоналі
   обрізання → повні ваги · вплив викидання мертвих нейронів
   α = −0.00751 → 0       · скільки бере глобальний масштаб

ПЕРЕДБАЧЕННЯ (A, ДО ПРОГОНУ):
 (а) пломба зійдеться в межах 5%                                  75%
 (б) fp32 дасть < 2% від E²                                       80%
 (в) Мелер K=2 — найбільший доданок (> 50%)                       55%
 (г) обрізання дасть < 10%                                        60%
 (д) 🚨 сума компонент НЕ зійдеться з E² (доданки не адитивні,
     бо α перепідбирається під кожну конфігурацію)                65%
"""
import glob
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
ALPHA = -0.00751
SEED = 12345
PILOT = 128
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/mini")
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 50


def phi(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / SQ2PI


def psi(x):
    x = np.asarray(x, float)
    return phi(x) + x * ndtr(x)


def gl(a, b, npan, nnod):
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    return ((0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel(),
            (0.5 * (hi - lo) * ws[None, :]).ravel())


ZQ, WQ = gl(0.0, 40.0, 12, 20)


def pair_quad(t, rho, n, chunk=8192):
    """точні гаусові парні моменти relu — те, що Мелер K=2 наближає."""
    tj = np.repeat(t, n)
    tk = np.tile(t, n)
    rr = rho.ravel()
    out = np.empty(n * n)
    q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-14))
    for o in range(0, n * n, chunk):
        sl = slice(o, min(o + chunk, n * n))
        x = ZQ[None, :] - tj[sl, None]
        u = (rr[sl, None] * x + tk[sl, None]) / q[sl, None]
        out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                   * psi(u)).sum(1)
    return out.reshape(n, n)


def closure_est(Wc, width, dt, mehler=True):
    """ТОЧНЕ відтворення `_closure_parts` sub99 (див. рядки 218–273)."""
    m = np.zeros(width, dtype=dt)
    C = np.eye(width, dtype=dt)
    for i in range(DEPTH):
        w = Wc[i].astype(dt)
        cp = w.T @ C @ w
        s = np.sqrt(np.maximum(np.diagonal(cp), 1e-20)).astype(dt)
        mp = (m @ w).astype(dt)
        t = mp / s
        ph = np.exp(-0.5 * t.astype(np.float64) ** 2) / SQ2PI
        Ph = ndtr(t.astype(np.float64))
        ph = ph.astype(dt)
        Ph = Ph.astype(dt)
        m = mp * Ph + s * ph
        if i == DEPTH - 1:
            break
        if mehler:
            C = cp * np.outer(Ph, Ph) + (0.5 * (cp * cp)) * np.outer(ph / s,
                                                                     ph / s)
        else:
            n = w.shape[1]
            rho = np.clip(cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
            C = (pair_quad(t.astype(np.float64), rho.astype(np.float64), n)
                 * (s[:, None] * s[None, :])).astype(dt)
            C = C - np.outer(m, m)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + np.diag(d - np.diagonal(C))
    return m.astype(np.float64)


def prune(Wfull, width, dt):
    """пілотне обрізання, як у sub99: 128 пар антитетично, `max(h) >= 0`."""
    rng = np.random.default_rng(SEED)
    zp = rng.standard_normal((PILOT, width)).astype(np.float32)
    xp = np.concatenate([zp, -zp], axis=0)
    alive, a = [], xp
    for i in range(DEPTH):
        h = a @ Wfull[i].astype(np.float32)
        if i < DEPTH - 1:
            alive.append(np.max(h, axis=0) >= 0.0)
        a = np.maximum(h, 0.0)
    Wc = []
    for i in range(DEPTH):
        wi = Wfull[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < DEPTH - 1:
            wi = wi[:, alive[i]]
        Wc.append(np.ascontiguousarray(wi, dtype=dt))
    return Wc


VAR = [
    ("A  як у сабміті   fp32 · Мелер · обрізання", np.float32, True, True),
    ("B  fp64            fp64 · Мелер · обрізання", np.float64, True, True),
    ("D  без обрізання   fp64 · Мелер · ПОВНІ",     np.float64, True, False),
]
if "--quad" in sys.argv:
    VAR += [
        ("C  квадратура    fp64 · КВАДР · обрізання", np.float64, False, True),
        ("E  чисте замикання fp64 · КВАДР · ПОВНІ",   np.float64, False, False),
    ]

print(f"decomp99 · розклад E_i² · {NNET} мереж мінісплиту · істина N=1e9",
      flush=True)
res = {k[0]: [] for k in VAR}
res2 = {k[0]: [] for k in VAR}
truth_all = []
for fpath in sorted(glob.glob(SRC + "/data-00000-of-00002.arrow")):
    tb = pa.ipc.open_stream(pa.memory_map(fpath, "rb")).read_all()
    x = tb.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(tb.num_rows, DEPTH, 256, 256)
    yv = tb.column("final_means").combine_chunks()
    while not pa.types.is_primitive(yv.type):
        yv = yv.flatten()
    FM = np.asarray(yv).reshape(tb.num_rows, 256)
    for i in range(min(tb.num_rows, NNET)):
        Wf = [np.ascontiguousarray(Wall[i][L], dtype=np.float64)
              for L in range(DEPTH)]
        tm = FM[i].astype(np.float64)
        truth_all.append(tm)
        for nm, dt, meh, pr in VAR:
            Wc = prune(Wf, 256, dt) if pr else [w.astype(dt) for w in Wf]
            m = closure_est(Wc, 256, dt, mehler=meh)
            res[nm].append(m)
            res2[nm].append(float((((1.0 + ALPHA) * m - tm) ** 2).mean()))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{NNET}", flush=True)

print("\n" + "=" * 86)
print(f"{'варіант':<44s}{'MSE (α сабміту)':>18s}{'α перепідібр.':>16s}")
print("=" * 86)
for nm, dt, meh, pr in VAR:
    M = np.array(res[nm])
    T = np.array(truth_all)
    mse = float(np.mean(res2[nm]))
    a = float((M * T).sum() / max((M * M).sum(), 1e-300))     # МНК-масштаб
    mse_a = float((((a * M) - T) ** 2).mean())
    tag = ""
    if nm.startswith("A"):
        tag = f"   🔒 ПЛОМБА: грейдер 2.338512e-05 ⟹ Δ {100*(mse/2.338512e-05-1):+.2f}%"
    print(f"{nm:<44s}{mse:>18.6e}{mse_a:>16.6e}{tag}")

print("\nРІЗНИЦІ (внесок КОЖНОГО наближення, при α сабміту):")
base = float(np.mean(res2[VAR[0][0]]))
for nm, dt, meh, pr in VAR[1:]:
    v = float(np.mean(res2[nm]))
    print(f"  вимкнули «{nm.split()[1]:<14s}» : {base:.4e} → {v:.4e}"
          f"   ×{base/v:.4f}   ⟹ {100*(1-v/base):+.1f}% від E²")
