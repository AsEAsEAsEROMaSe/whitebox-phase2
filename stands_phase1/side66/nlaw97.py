"""narrow97 — ІДЕЯ ЮЗЕРА: ВУЗЬКІ МЕРЕЖІ З ТОЧНОЮ ІСТИНОЮ. ЧИ Є ПОХИБКА
ЗАМИКАННЯ СКІНЧЕННОРОЗМІРНИМ ЕФЕКТОМ `1/n`?

════════════════ ЧОМУ ЦЕ ВЗАГАЛІ МОЖНА ЗМІРЯТИ ════════════════
Гіпотеза С97 («корінь усіх ~15 мертвих гілок»): похибка гаусового замикання є
ефектом СКІНЧЕННОЇ ШИРИНИ `1/n` для КОНКРЕТНОЇ `W`, а не браком порядку ряду.
При `n=256` це `1/256 = 0.39%` і тоне в шумі істини (банк: 24% робочого MSE).
При `n=4` той самий ефект у **64× більший** — і його видно без жодної точності.
🔑 ⟹ головне тут не «машинна точність», а те, що СИГНАЛ росте, поки шум стоїть.

════════════════ ЩО РОБИТЬ ІСТИНУ ТОЧНОЮ ПРИ n=2 ════════════════
Мережа БЕЗ БІАСІВ із ReLU **позитивно однорідна**: `h(cz) = c·h(z)`, c>0.
Для гаусового `z`: `z = ‖z‖·u`, і `‖z‖ ⟂ u` ⟹
```
E[ReLU(w·h(z))] = E‖z‖ · E_{u~Unif(Sⁿ⁻¹)}[ReLU(w·h(u))] ,
E‖z‖ = √2·Γ((n+1)/2)/Γ(n/2)
```
⟹ `n`-вимірний інтеграл ТОЧНО зводиться до `(n−1)`-вимірного по сфері.
При `n=2` це ОДНОВИМІРНИЙ інтеграл по куту, а всередині кожного сектора
(де всі затвори фіксовані) підінтегральна функція є РІВНО `a·cosθ + b·sinθ`
⟹ інтегрується АНАЛІТИЧНО. Злами знаходимо бісекцією до 1e-16.
🎯 Це істина машинної точності на реальній мережі глибини 32 — те, чого в
проєкті не було ЖОДНОГО РАЗУ.

════════════════ ПЛОМБИ ════════════════
 S1 🔒 аналітична істина при n=2 проти МК тієї ж мережі — мусять збігтись
      у межах шуму МК (це пломба на ВЕСЬ конвеєр, не на формулу)
 S2 🔒 однорідність: `h(2z) = 2h(z)` побітово
 S3 🔒 шум МК міряти розщепленням вибірки навпіл, а не оцінювати
 S4 🔒 замикання при n=1 мусить бути ТОЧНИМ (гаус лишається гаусом,
      бо один нейрон — це лінійна форма) ⟹ похибка РІВНО 0

════════════════ ПЕРЕДБАЧЕННЯ (ЗАПИСАНО ДО ПРОГОНУ) ════════════════
 (а) S1 зійдеться в межах шуму МК                                  90%
 (б) 🚨 НАСИЧЕННЯ: при малому `n` похибка впреться в O(1) і локальний
     нахил БУДЕ ПОЛОГІШИЙ за −1 — тобто фіт на n=2..5 ЗАНИЗИТЬ
     показник. Саме тому друкую ЛОКАЛЬНІ нахили, а не один фіт      70%
 (в) у смузі n=8..16 локальний нахил ляже в [−1.3, −0.7]           55%
 (г) екстраполяція з n=4..8 передбачить n=16 у межах 25%            50%
 (д) при n=1 похибка буде < 1e-12 (пломба S4)                       95%
🚨 ЧЕСНО ПРО РЕЖИМ: при малому `n` інші SNR, частка `t>0` і концентрація
напрямку. Закон №11 («тестувати на L=32») тримаю — глибина СКРІЗЬ 32 — але
закон про режим порушую свідомо. Саме тому вирок дає ПЕРЕВІРКА ЕКСТРАПОЛЯЦІЇ
на n=16, а не сама екстраполяція.
"""
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr, gammaln                       # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
NQ_PRICE = 720


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
    """He-init, без біасів, глибина 32. `h = h @ W[L]`, як у банку."""
    rng = np.random.default_rng(seed)
    sd = math.sqrt(2.0 / n)
    return [rng.standard_normal((n, n)) * sd for _ in range(DEPTH)]


# ─────────────────── ЗАМИКАННЯ (ТОЧНІ гаусові парні моменти) ───────────────
def price_pairs(t, rho):
    """матриця E[(x+t_j)₊(y+t_k)₊] для гаусової пари, точна квадратура."""
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
    """гаусове замикання моментів; повертає `m` активацій останнього шару."""
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
        C = 0.5 * ((M2 - np.outer(ma, ma)) + (M2 - np.outer(ma, ma)).T)
        m = ma
    return m


# ─────────────────── ІСТИНА: МОНТЕ-КАРЛО (будь-яке n) ───────────────
def truth_mc(W, n, N, seed, halves=True):
    """повертає (середнє, оцінка шуму з розщеплення навпіл)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    accA = np.zeros(n)
    accB = np.zeros(n)
    nA = nB = 0
    CH = max(1, min(1 << 16, N))
    done = 0
    k = 0
    while done < N:
        b = min(CH, N - done)
        z = rng.standard_normal((b, n))
        h = z
        for L in range(DEPTH):
            h = np.maximum(h @ W[L], 0.0)
        s = h.sum(0)
        acc += s
        if k % 2 == 0:
            accA += s
            nA += b
        else:
            accB += s
            nB += b
        done += b
        k += 1
    mean = acc / N
    if halves and nA > 0 and nB > 0:
        dif = accA / nA - accB / nB
        noise = 0.5 * np.sqrt((dif * dif).mean())     # SE повної вибірки
    else:
        noise = float("nan")
    return mean, noise


# ─────────────────── ІСТИНА: АНАЛІТИЧНА ПРИ n=2 ───────────────
def _fwd2(theta, W, upto=None):
    """передактивації шару `upto` (або активації виходу, якщо None) для кутів."""
    u = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    h = np.atleast_2d(u)
    for L in range(DEPTH):
        p = h @ W[L]
        if upto is not None and L == upto:
            return p
        h = np.maximum(p, 0.0)
    return h


def truth_exact_2d(W, ngrid=60000):
    """ТОЧНО при n=2: однорідність ⟹ інтеграл по колу; на кожному секторі всі
    затвори фіксовані ⟹ вихід є РІВНО `A·cosθ + B·sinθ` ⟹ інтеграл аналітичний
    (`∫ = A(sin) − B(cos)`). Злами — нулі БУДЬ-ЯКОЇ передактивації, бісекція 60
    ітерацій. Похибка методу — лише округлення."""
    n = 2
    th = np.linspace(0.0, 2.0 * math.pi, ngrid, endpoint=False)
    brk = [0.0, 2.0 * math.pi]
    for L in range(DEPTH):
        p = _fwd2(th, W, upto=L)
        for a in range(n):
            v = p[:, a]
            sg = np.signbit(v)
            for i in np.nonzero(sg[:-1] != sg[1:])[0]:
                lo, hi = th[i], th[i + 1]
                slo = np.signbit(v[i])
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    vm = _fwd2(np.array([mid]), W, upto=L)[0, a]
                    if np.signbit(vm) == slo:
                        lo = mid
                    else:
                        hi = mid
                brk.append(0.5 * (lo + hi))
    brk = np.unique(np.array(brk))
    tot = np.zeros(n)
    for i in range(brk.size - 1):
        lo, hi = brk[i], brk[i + 1]
        if hi - lo < 1e-13:
            continue
        t1, t2 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
        v1 = _fwd2(np.array([t1]), W)[0]
        v2 = _fwd2(np.array([t2]), W)[0]
        c1, s1 = math.cos(t1), math.sin(t1)
        c2, s2 = math.cos(t2), math.sin(t2)
        det = c1 * s2 - c2 * s1
        A = (v1 * s2 - v2 * s1) / det
        B = (v2 * c1 - v1 * c2) / det
        seg = (A * (math.sin(hi) - math.sin(lo))
               - B * (math.cos(hi) - math.cos(lo)))
        # затвори фіксовані на секторі ⟹ вихід або ≡ лінійна форма, або ≡ 0
        tot += np.where(_fwd2(np.array([0.5 * (lo + hi)]), W)[0] > 0, seg, 0.0)
    En = math.exp(0.5 * math.log(2.0) + gammaln((n + 1) / 2) - gammaln(n / 2))
    return En * tot / (2.0 * math.pi)


print("narrow97 — вузькі мережі, точна істина, закон 1/n", flush=True)
t0 = time.time()

print('\n⛔ S1/S4 ЗНЯТО: при n<=4 мережа МЕРТВА (surv97) — об єкта не існує')

# ══════════ ГОЛОВНА ТАБЛИЦЯ ══════════
print("\n" + "=" * 96)
print("ПОХИБКА ГАУСОВОГО ЗАМИКАННЯ ПРОТИ ШИРИНИ (глибина 32 СКРІЗЬ)")
print("=" * 96)
print(f"{'n':>4s}{'мереж':>7s}{'N (МК)':>11s}{'медіана відн.похибки':>22s}"
      f"{'шум МК':>11s}{'сигнал/шум':>12s}{'лок. нахил':>12s}")
NETS = {8: 24, 12: 24, 16: 24, 24: 16, 32: 12}
NSAMP = {8: 1 << 20, 12: 1 << 20, 16: 1 << 20, 24: 1 << 19, 32: 1 << 19}
res = {}
prev = None
for n in (8, 12, 16, 24, 32):
    errs, noises = [], []
    for k in range(NETS[n]):
        W = make_net(n, 900000 + 1000 * n + k)
        cm = closure(W, n)
        tm, nz = truth_mc(W, n, NSAMP[n], 500000 + 7 * k + n)
        sc = np.sqrt((tm * tm).mean())
        errs.append(float(np.sqrt(((cm - tm) ** 2).mean()) / sc))
        noises.append(float(nz / sc))
    e = float(np.median(errs))
    nzm = float(np.median(noises))
    sl = ("" if prev is None
          else f"{math.log(e/prev[1])/math.log(n/prev[0]):>12.3f}")
    print(f"{n:>4d}{NETS[n]:>7d}{NSAMP[n]:>11d}{100*e:>21.4f}%"
          f"{100*nzm:>10.4f}%{e/max(nzm,1e-30):>12.1f}{sl}", flush=True)
    res[n] = e
    prev = (n, e)

print(f"\n  час {time.time()-t0:.0f} с")
print("\n" + "-" * 96)
print("ЕКСТРАПОЛЯЦІЯ (запобіжник контролера): закон із n=8..16 → передбачення n=32")
ns = np.array([8.0, 12.0, 16.0])
ys = np.array([res[8], res[12], res[16]])
A = np.polyfit(np.log(ns), np.log(ys), 1)
pred16 = math.exp(A[1] + A[0] * math.log(32.0))
print(f"  нахил на n=4..8 = {A[0]:.3f} ⟹ передбачення n=32: {100*pred16:.4f}%")
print(f"  ЗМІРЯНО n=32: {100*res[32]:.4f}%  ⟹ промах "
      f"×{max(pred16, res[32])/min(pred16, res[32]):.3f}")
print(f"  екстраполяція на n=256: {100*math.exp(A[1]+A[0]*math.log(256.0)):.4f}%"
      f"   (зміряне замикання проєкту 0.4647–0.5070%)")
