"""jseal89 — ПЛОМБА НА ВИВЕДЕНІ ІНТЕГРАЛИ `J_p` ДО ТОГО, ЯК ЇХ ВЖИВАТИ.

Для Едвортової густини `f(x) = φ_g(x)[1 + γ₃He₃(z)/6 + …]`, `z=(x−m)/s`,
моменти зрізаної величини `relu(X)` набувають вигляду
    E[relu^p] = G_p + (γ₃/6)·J_p ,  J_p = ∫_{−t}^∞ (m+sz)^p He₃(z)φ(z)dz
ВИВЕДЕНО (через `He_{n}φ = −d/dz[He_{n−1}φ]` і `z·He_n = He_{n+1}+n·He_{n−1}`):
    J_0 = (t²−1)φ(t)      J_1 = −s·t·φ(t)
    J_2 = 2s²φ(t)         J_3 = 6s³Φ(t)
🚨 ТРИ З ЧОТИРЬОХ ЗГОРНУЛИСЬ В ОДНОЧЛЕН — надто красиво, щоб вірити алгебрі,
яку перевіряє та сама алгебра. Потрібен НЕЗАЛЕЖНИЙ шлях до тих самих чисел.

ПЛОМБА: пряма чисельна квадратура на щільній сітці, у `float64`, для набору
`(m, s)`, що НАКРИВАЄ РОБОЧУ ТОЧКУ (`|t|` від 0.002 до 3.4 — зміряний
профіль `tprof88`), а не лише зручні `t≈0`.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ: збіг ≤1e-10 відносних скрізь (шанс 80%). Якщо
розійдеться саме на великих `|t|` — виведення підганяли під малі `t`, і вся
рекурсія κ₃ поїде саме на глибині, де живуть гроші.
"""
import numpy as np

SQ2PI = np.sqrt(2.0 * np.pi)


def phi(t):
    return np.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    from math import erf
    return np.array([0.5 * (1.0 + erf(x / np.sqrt(2.0))) for x in np.atleast_1d(t)])


def He(n, z):
    if n == 0:
        return np.ones_like(z)
    if n == 1:
        return z
    hm, h = np.ones_like(z), z
    for k in range(1, n):
        hm, h = h, z * h - k * hm
    return h


def J_analytic(p, m, s):
    t = m / s
    if p == 0:
        return (t * t - 1.0) * phi(t)
    if p == 1:
        return -s * t * phi(t)
    if p == 2:
        return 2.0 * s * s * phi(t)
    if p == 3:
        return 6.0 * s ** 3 * Phi(np.array([t]))[0]
    raise ValueError(p)


def J_quad(p, m, s, half=40.0, n=4_000_001):
    """Пряма квадратура по z від −t до +∞ (зрізано на 40σ, крок ~2e-05)."""
    t = m / s
    z = np.linspace(-t, half, n)
    g = (m + s * z) ** p * He(3, z) * phi(z)
    return np.trapezoid(g, z)


print("=" * 78)
print("ПЛОМБА `J_p`: АНАЛІТИКА ПРОТИ КВАДРАТУРИ (float64)")
print("=" * 78)
print(f"{'t':>8s} {'s':>6s} {'p':>3s} {'аналітика':>16s} {'квадратура':>16s} {'віднос.':>11s}")
worst = 0.0
for t in (0.002, 0.25, 1.0, 1.9, 2.674, 3.378, -1.5, -3.0):
    for s in (0.7, 1.0, 2.3):
        m = t * s
        for p in (0, 1, 2, 3):
            a = float(J_analytic(p, m, s))
            q = float(J_quad(p, m, s))
            rel = abs(a - q) / max(abs(q), 1e-300)
            worst = max(worst, rel)
            if s == 1.0:
                print(f"{t:>8.3f} {s:>6.1f} {p:>3d} {a:>16.9e} {q:>16.9e} {rel:>11.2e}")

print()
print(f"🔬 НАЙГІРШЕ ВІДХИЛЕННЯ ПО ВСІХ 96 ТОЧКАХ: {worst:.3e}")
print(f"   вирок: {'✅ ПЛОМБА ТРИМАЄ' if worst < 1e-8 else '⛔ ВИВЕДЕННЯ ХИБНЕ'}")

print()
print("=" * 78)
print("ДРУГА ПЛОМБА: ЧИ ЗБЕРІГАЄ ПОПРАВКА НОРМУВАННЯ І ЧИ ДАЄ ПРАВИЛЬНИЙ κ₃?")
print("=" * 78)
print("   якщо густина Едворта правильна, то для НЕзрізаної величини")
print("   E[(X−m)³] мусить дорівнювати РІВНО γ₃s³ (за визначенням γ₃).")
for g3 in (0.0, 0.3, 0.8):
    for s in (1.0, 2.3):
        z = np.linspace(-40, 40, 4_000_001)
        f = phi(z) * (1.0 + g3 * He(3, z) / 6.0)
        norm = np.trapezoid(f, z)
        mu3 = np.trapezoid((s * z) ** 3 * f, z)
        print(f"   γ₃={g3:.1f} s={s:.1f}:  ∫f = {norm:.12f}   "
              f"E[(X−m)³] = {mu3:.9f}   очік {g3 * s ** 3:.9f}   "
              f"Δ={abs(mu3 - g3 * s ** 3):.2e}")
