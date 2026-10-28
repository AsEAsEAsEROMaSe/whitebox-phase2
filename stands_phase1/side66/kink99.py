"""kink99 — КОНТРОЛЬ НА МЕХАНІЗМ, А НЕ НА КОРЕЛЯЦІЮ (закон С95 №16).

ГІПОТЕЗА, яку дала `wfeat99`: `corr(медіана|t|, log e) = −0.60` ⟹ похибку
замикання ТВОРЯТЬ нейрони біля зламу ReLU; де ворота насичені, замикання точне.

🚨 Кореляція цього НЕ доводить: медіана |t| корелює з десятком інших величин.
Механізм дає ПЕРЕВІРЮВАНЕ передбачення, якого кореляція не дає:

    при |t| → ∞ ворота фіксуються ⟹ мережа стає ЛІНІЙНОЮ на носії ⟹
    вихід точно гаусів ⟹ похибка гаусового замикання → РІВНО 0.

Ручка: зсув входу `z ~ N(m₀, I)`, `m₀ = c·u`. При c=0 маємо базовий режим;
зі зростанням c усі `t` ростуть ~c і ворота насичуються.

ПЕРЕДБАЧЕННЯ (ДО ПРОГОНУ):
 (а) e монотонно спадає з c                                          80%
 (б) при c=16 буде e < 1/10 базового                                 65%
 (в) 🔒 при c=0 відтворить медіану 9.28% з wfeat99 (пломба режиму)    95%
 (г) якщо (а) провалиться — механізм ХИБНИЙ, а `corr=−0.60` є збігом
     через треті величини, і вирок wfeat99 треба переписати            —
🔒 ПЛОМБА ШУМУ: МК щоразу з розщепленням навпіл; вирок лише там, де
   e ≫ шуму. Якщо e впаде В шум — це НЕ «нуль», це «нижче роздільної».
"""
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
NWID = 32
NNET = 12
NSAMP = 1 << 20


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


ZQ, WQ = gl(0.0, 40.0, 24, 20)


def price_pairs(t, rho):
    n = t.size
    tj = np.repeat(t, n)
    tk = np.tile(t, n)
    rr = rho.ravel()
    q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-14))
    x = ZQ[None, :] - tj[:, None]
    u = (rr[:, None] * x + tk[:, None]) / q[:, None]
    return ((WQ[None, :] * ZQ[None, :] * phi(x) * q[:, None]
             * psi(u)).sum(1)).reshape(n, n)


def closure(W, n, m0):
    m = m0.copy()
    C = np.eye(n)
    tabs = []
    for L in range(DEPTH):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        s2 = np.maximum(np.diag(Cp), 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
        M2 = price_pairs(t, rho) * (s[:, None] * s[None, :])
        np.fill_diagonal(M2, s2 * ((1 + t * t) * ndtr(t) + t * phi(t)))
        ma = s * psi(t)
        Cn = M2 - np.outer(ma, ma)
        C = 0.5 * (Cn + Cn.T)
        m = ma
        tabs.append(np.abs(t))
    return m, float(np.median(np.concatenate(tabs)))


def truth(W, n, N, seed, m0):
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    aA = np.zeros(n)
    aB = np.zeros(n)
    nA = nB = 0
    CH = 1 << 15
    done = 0
    k = 0
    while done < N:
        b = min(CH, N - done)
        h = rng.standard_normal((b, n)) + m0[None, :]
        for L in range(DEPTH):
            h = np.maximum(h @ W[L], 0.0)
        s = h.sum(0)
        acc += s
        if k % 2 == 0:
            aA += s
            nA += b
        else:
            aB += s
            nB += b
        done += b
        k += 1
    dif = aA / nA - aB / nB
    return acc / N, 0.5 * math.sqrt(float((dif * dif).mean()))


def make_net(n, seed):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((n, n)) * math.sqrt(2.0 / n)
            for _ in range(DEPTH)]


print(f"kink99 · насичення воріт · n={NWID} · {NNET} мереж · N={NSAMP}",
      flush=True)
print("\nc — сила зсуву входу m₀ = c·u  (c=0 — базовий режим проєкту)")
print(f"{'c':>6s}{'мед|t|':>10s}{'медіана e':>12s}{'шум МК':>10s}"
      f"{'e/шум':>9s}{'e/e(c=0)':>11s}")
t0 = time.time()
base = None
for c in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
    es, nzs, tm_ = [], [], []
    for k in range(NNET):
        W = make_net(NWID, 990000 + 31 * k)          # ТІ САМІ мережі, що wfeat99
        rng0 = np.random.default_rng(4242 + k)
        u = rng0.standard_normal(NWID)
        u /= np.linalg.norm(u)
        m0 = c * math.sqrt(NWID) * u                 # ‖m₀‖ = c·√n ⟹ порівнянно з шумом
        cm, mt = closure(W, NWID, m0)
        tv, nz = truth(W, NWID, NSAMP, 660000 + 13 * k, m0)
        sc = math.sqrt(float((tv * tv).mean()))
        if sc < 1e-12:
            continue
        es.append(float(np.sqrt(((cm - tv) ** 2).mean()) / sc))
        nzs.append(nz / sc)
        tm_.append(mt)
    e = float(np.median(es))
    nzm = float(np.median(nzs))
    if base is None:
        base = e
    print(f"{c:>6.1f}{np.median(tm_):>10.2f}{100*e:>11.4f}%{100*nzm:>9.4f}%"
          f"{e/max(nzm,1e-30):>9.1f}{e/base:>11.4f}", flush=True)

print(f"\n  час {time.time()-t0:.0f} с")
print("🔒 пломба (в): c=0 мусить дати ≈9.28% (wfeat99, 48 мереж) — тут 12 мереж,"
      " тож допуск ширший")
