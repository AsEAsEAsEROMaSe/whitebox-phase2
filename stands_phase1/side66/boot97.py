"""boot97 — ЗАКРИТИ ГАП, ЯКИЙ Я НАЗВАЛА САМА: НЕВИЗНАЧЕНІСТЬ МЕДІАНИ.

`nlaw97` дав нахили −1.445 · −2.433 · −1.152 · −1.096 і я записала «нахили
гуляють». Але медіана бралась по 12–24 мережах із важкими хвостами, і її
власна невизначеність НЕ МІРЯЛАСЬ. Якщо вуса широкі, правильне твердження —
«нахил ≈ −1.2 з широкими вусами», а це ЗОВСІМ ІНШЕ твердження, ніж «гуляє».

🔒 МЕТОД: ті самі сіди ⟹ ті самі мережі й та сама істина (детерміновано),
плюс бутстреп ПО МЕРЕЖАХ (2000 розіграшів) на медіану і на локальні нахили.
Нових даних не потрібно — потрібна лише чесна дисперсія наявних.

ПЕРЕДБАЧЕННЯ (до прогону):
 (а) вуса на нахилі при n=12→16 накриють −1                     55%
 (б) вуса накриють −1 на ВСІХ чотирьох переходах ⟹ вирок
     пом'якшується до «узгоджено з 1/n, точність фіту замала»   35%
 (в) при n=24→32 вуса будуть найвужчі (найбільш стабільний режим) 45%
     ⚠️ насправді там НАЙМЕНШЕ мереж (16 і 12) ⟹ можуть бути найширші
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32


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


ZQ, WQ = gl(0.0, 40.0, 30, 24)


def make_net(n, seed):
    rng = np.random.default_rng(seed)
    sd = math.sqrt(2.0 / n)
    return [rng.standard_normal((n, n)) * sd for _ in range(DEPTH)]


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


def closure(W, n):
    m = np.zeros(n)
    C = np.eye(n)
    for L in range(DEPTH):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        s2 = np.maximum(np.diag(Cp), 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
        M2 = price_pairs(t, rho) * (s[:, None] * s[None, :])
        d2 = s2 * ((1 + t * t) * ndtr(t) + t * phi(t))
        ma = s * psi(t)
        np.fill_diagonal(M2, d2)
        Cn = M2 - np.outer(ma, ma)
        C = 0.5 * (Cn + Cn.T)
        m = ma
    return m


def truth_mc(W, n, N, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    CH = min(1 << 16, N)
    done = 0
    while done < N:
        b = min(CH, N - done)
        h = rng.standard_normal((b, n))
        for L in range(DEPTH):
            h = np.maximum(h @ W[L], 0.0)
        acc += h.sum(0)
        done += b
    return acc / N


NETS = {8: 24, 12: 24, 16: 24, 24: 16, 32: 12}
NSAMP = {8: 1 << 20, 12: 1 << 20, 16: 1 << 20, 24: 1 << 19, 32: 1 << 19}
ORDER = (8, 12, 16, 24, 32)

print("boot97 — довірчі вуса на медіану і на локальні нахили", flush=True)
E = {}
for n in ORDER:
    v = []
    for k in range(NETS[n]):
        W = make_net(n, 900000 + 1000 * n + k)          # ТІ САМІ сіди, що в nlaw97
        cm = closure(W, n)
        tm = truth_mc(W, n, NSAMP[n], 500000 + 7 * k + n)
        sc = math.sqrt(float((tm * tm).mean()))
        if sc <= 0:
            continue                                     # мертва мережа
        v.append(float(np.sqrt(((cm - tm) ** 2).mean()) / sc))
    E[n] = np.array(v)
    print(f"  n={n:>2d}: {len(v)} живих · медіана {100*np.median(v):.4f}%"
          f" · 25–75% {100*np.percentile(v,25):.3f}…"
          f"{100*np.percentile(v,75):.3f}%", flush=True)

rng = np.random.default_rng(4242)
B = 4000
boot = {n: np.array([np.median(rng.choice(E[n], E[n].size, replace=True))
                     for _ in range(B)]) for n in ORDER}

print("\n" + "=" * 88)
print("МЕДІАНА З ДОВІРЧИМИ ВУСАМИ (бутстреп по мережах, 4000 розіграшів)")
print("=" * 88)
print(f"{'n':>4s}{'мереж':>7s}{'медіана':>11s}{'2.5%':>11s}{'97.5%':>11s}"
      f"{'відн. ширина':>15s}")
for n in ORDER:
    lo, hi = np.percentile(boot[n], [2.5, 97.5])
    print(f"{n:>4d}{E[n].size:>7d}{100*np.median(E[n]):>10.4f}%"
          f"{100*lo:>10.4f}%{100*hi:>10.4f}%{(hi-lo)/np.median(E[n]):>14.2f}")

print("\n" + "=" * 88)
print("ЛОКАЛЬНІ НАХИЛИ З ВУСАМИ — ЧИ НАКРИВАЮТЬ ВОНИ −1")
print("=" * 88)
print(f"{'перехід':>12s}{'нахил':>10s}{'2.5%':>10s}{'97.5%':>10s}"
      f"{'накриває −1?':>15s}")
for a, b in zip(ORDER, ORDER[1:]):
    s = np.log(boot[b] / boot[a]) / math.log(b / a)
    lo, hi = np.percentile(s, [2.5, 97.5])
    pt = math.log(np.median(E[b]) / np.median(E[a])) / math.log(b / a)
    print(f"{f'{a}→{b}':>12s}{pt:>10.3f}{lo:>10.3f}{hi:>10.3f}"
          f"{('ТАК' if lo <= -1.0 <= hi else 'НІ'):>15s}")

print("\n" + "-" * 88)
print("ГЛОБАЛЬНИЙ НАХИЛ по n=12..32 (без напівмертвого n=8), з вусами:")
xs = np.log(np.array([12.0, 16.0, 24.0, 32.0]))
sl = []
for i in range(B):
    ys = np.log(np.array([boot[n][i] for n in (12, 16, 24, 32)]))
    sl.append(np.polyfit(xs, ys, 1)[0])
sl = np.array(sl)
lo, hi = np.percentile(sl, [2.5, 97.5])
pt = np.polyfit(xs, np.log([np.median(E[n]) for n in (12, 16, 24, 32)]), 1)[0]
print(f"  нахил = {pt:.3f}  ·  95% вуса [{lo:.3f}, {hi:.3f}]  ·  "
      f"накриває −1: {'ТАК' if lo <= -1.0 <= hi else 'НІ'}")
print(f"  🎯 ЯКІР n=256 (pairrec97, та сама конструкція) = 0.4647% ⟹ "
      f"ефективний нахил 32→256 = "
      f"{math.log(0.004647/np.median(E[32]))/math.log(8.0):.3f}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/boot97.npz",
         **{f"n{n}": E[n] for n in ORDER})
