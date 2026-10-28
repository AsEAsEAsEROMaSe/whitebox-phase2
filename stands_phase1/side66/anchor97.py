"""anchor97 — ЯКІР `n=256` НА БАГАТЬОХ МЕРЕЖАХ. ЗНІМАЄ «ОДНА МЕРЕЖА, ±40%»,
ЯКЕ Я САМА НАЗВАЛА СЛАБКИМ МІСЦЕМ ЗАКОНУ `n^(−1.46)`.

════════════ ЧОМУ НЕ n=48/64, ЯК ПРОПОНУВАВ КОНТРОЛЕР ════════════
Перевірка «чи це рівно −3/2» через додавання `n=48,64` НЕ МАЄ ПОТУЖНОСТІ:
```
n=48: при −1.459 → 5.3192% · при −1.5 → 5.2315%  ⟹ гіпотези різняться на 1.65%
n=64: при −1.459 → 3.4959% · при −1.5 → 3.3979%  ⟹                        2.80%
невизначеність медіани при n=32 (бутстреп) = −46%/+40%
```
⟹ сигнал 2.8% проти шуму 40%. Щоб гіпотези розійшлись на 20%, потрібне
відношення ширин **85**, а в нас максимум 256/12 = 21. Тобто розрізнити
−1.46 від −1.5 нашими засобами НЕМОЖЛИВО В ПРИНЦИПІ, і додатковий прогін
на 48/64 дав би лише ілюзію перевірки.

════════════ ЩО МОЖНА І ЩО РОБИМО ════════════
Єдиний уже наявний важіль — плече `32→256` (r=8). Його точність зараз
обмежує ОДНА мережа на якорі:
```
одна мережа  ±40%  ⟹ невизначеність нахилу ln(1.45)/ln(8) = ±0.179
16 мереж     ±10%  ⟹                                        ±0.046
```
⟹ 16 мереж на якорі стискають вуса плеча ВЧЕТВЕРО. Це не розсудить
−1.46/−1.5 (не розсудить ніщо), але прибирає єдине справді слабке місце
закону — і робить показник вимірюваним, а не збігом однієї мережі.

════════════ ІСТИНА ════════════
`final_means` ПУБЛІЧНОГО БАНКУ: **N = 1e9** семплів, уже на диску, без
завантажень. Це в 10× точніша істина за датасет вищих моментів (1e8) і в
2500× за наш робочий МК.
🔒 ПЛОМБА: мережа `mlp_id=0` мусить відтворити 0.4647% з `pairrec97`
(та сама конструкція, той самий об'єкт) — інакше стенд міряє щось інше.

════════════ ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ) ════════════
 (а) пломба на mlp_id=0 зійдеться в межах 2%                      85%
 (б) медіана по 16 мережах ляже в 0.35…0.60%                      70%
 (в) плече 32→256 після цього дасть нахил у [−1.55, −1.35]        60%
 (г) розкид МІЖ мережами при n=256 буде ВУЖЧИЙ, ніж при n=32
     (IQR/медіана 0.79 там) — бо велика ширина усереднює            65%
"""
import glob
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
NNET = 16
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


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


ZQ, WQ = gl(0.0, 40.0, 12, 20)          # 240 вузлів: ~1e-9, вистачає з запасом


def closure(W, n, chunk=8192):
    m = np.zeros(n)
    C = np.eye(n)
    for L in range(DEPTH):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        s2 = np.maximum(np.diag(Cp), 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
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
        M2 = out.reshape(n, n) * (s[:, None] * s[None, :])
        np.fill_diagonal(M2, s2 * ((1 + t * t) * ndtr(t) + t * phi(t)))
        ma = s * psi(t)
        Cn = M2 - np.outer(ma, ma)
        C = 0.5 * (Cn + Cn.T)
        m = ma
    return m


print(f"anchor97 · якір n=256 на {NNET} мережах · істина final_means N=1e9",
      flush=True)
t0 = time.time()
errs = []
for f in sorted(glob.glob(SRC + "/data-*.arrow")):
    t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
    x = t.column("weights").combine_chunks()
    while not pa.types.is_primitive(x.type):
        x = x.flatten()
    Wall = np.asarray(x).reshape(t.num_rows, DEPTH, 256, 256)
    y = t.column("final_means").combine_chunks()
    while not pa.types.is_primitive(y.type):
        y = y.flatten()
    FM = np.asarray(y).reshape(t.num_rows, 256)
    ids = t.column("mlp_id").to_pylist()
    for i in range(t.num_rows):
        if len(errs) >= NNET:
            break
        W = [np.ascontiguousarray(Wall[i][L], dtype=np.float64)
             for L in range(DEPTH)]
        cm = closure(W, 256)
        tm = FM[i].astype(np.float64)
        sc = math.sqrt(float((tm * tm).mean()))
        e = float(np.sqrt(((cm - tm) ** 2).mean()) / sc)
        errs.append(e)
        tag = ""
        if ids[i] == 0:
            tag = (f"  🔒 ПЛОМБА mlp_id=0: {100*e:.4f}% проти pairrec97 0.4647%"
                   f" ⟹ Δ {100*abs(e-0.004647)/0.004647:.2f}%")
        print(f"  мережа {len(errs):>2d} (id {ids[i]:>3d}): {100*e:.4f}%"
              f"  ({time.time()-t0:.0f} с){tag}", flush=True)
    if len(errs) >= NNET:
        break

E = np.array(errs)
rng = np.random.default_rng(97)
boot = np.array([np.median(rng.choice(E, E.size, replace=True))
                 for _ in range(4000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
med = float(np.median(E))
print("\n" + "=" * 84)
print(f"ЯКІР n=256 · {NNET} мереж · істина N=1e9")
print(f"  медіана {100*med:.4f}%  ·  95% вуса [{100*lo:.4f}%, {100*hi:.4f}%]"
      f"  ·  IQR {100*np.percentile(E,25):.4f}…{100*np.percentile(E,75):.4f}%")
print(f"  відн. ширина вусів {(hi-lo)/med:.3f}   (при n=32 було 0.86)")

M32 = 0.096108
print("\nПЛЕЧЕ 32→256 З ВУСАМИ:")
b32lo, b32hi = 0.052251, 0.134794          # бутстреп-вуса медіани при n=32
sl_pt = math.log(med / M32) / math.log(8.0)
sl_lo = math.log(lo / b32hi) / math.log(8.0)
sl_hi = math.log(hi / b32lo) / math.log(8.0)
print(f"  нахил = {sl_pt:.3f}   95% вуса приблизно [{sl_lo:.3f}, {sl_hi:.3f}]")
print(f"  накриває −1.0 : {'ТАК' if sl_lo <= -1.0 <= sl_hi else 'НІ'}")
print(f"  накриває −1.5 : {'ТАК' if sl_lo <= -1.5 <= sl_hi else 'НІ'}")
print(f"\n  фіт n=12..32 давав −1.459 ⟹ передбачення n=256 = "
      f"{100*M32*8**-1.459:.4f}%  проти зміряних {100*med:.4f}%"
      f"  ⟹ промах ×{max(med, M32*8**-1.459)/min(med, M32*8**-1.459):.3f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/anchor97.npz", E=E)
