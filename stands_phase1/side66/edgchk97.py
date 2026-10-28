"""edgchk97 — НЕЗАЛЕЖНА ЧИСЕЛЬНА ПЛОМБА на 4 замкнені форми Еджворта 4-го порядку.

ОБ'ЄКТ. X ~ p(x) = phi(x)*[1 + g4*He4(x)/24]  (gamma3 = 0);  Y = sigma*X + mu;  t = mu/sigma.
Перевіряються ЧОТИРИ твердження сесії ВИМІР:
 (1) E[ReLU(Y)]        = s*[phi(t) + t*Phi(t)]            + s *(g4/24)*(t^2-1)*phi(t)
 (2) E[ReLU(Y)^2]      = s^2*[(1+t^2)*Phi(t) + t*phi(t)]  - s^2*(g4/12)*t*phi(t)
 (3) E[ReLU(Y)*He1(X)] = s*Phi(t)                         - s *(g4/24)*phi(t)*(t^3 + t)
 (4) E[ReLU(Y)*He2(X)] = s*phi(t)                         + s *(g4/24)*phi(t)*(t^4 + 2t^2 + 7)

МЕТОД. Ліва частина — ПРЯМА квадратура інтеграла по [-t, +inf) (кінк ReLU стоїть
РІВНО на межі, тож підінтегральна функція гладка всередині). ДВА незалежні
квадратурні шляхи, щоб хвороба приладу не видала себе за хворобу формули:
  A) mpmath.quad (tanh-sinh / Gauss-Legendre adaptive), mp.dps=50 — еталон;
  B) numpy Gauss-Legendre 600 вузлів на [-t, T] з T = t + 40 — незалежна реалізація
     на іншій арифметиці (float64) і з іншим правилом.
Розбіжність A vs B — це роздільна здатність приладу; вирок виносимо ЛИШЕ по A.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (протокол п.6). Я вивів усі чотири вручну НЕЗАЛЕЖНО від
ВИМІРУ (через x*He_n = He_{n+1} + n*He_{n-1} і int_{-t}^inf He_n phi = He_{n-1}(-t)*phi(t)):
(1) (t^2-1), (2) -2t/24 = -t/12, (3) -(t^3+t), (4) +(t^4+2t^2+7) — збіглось усе чотири.
⚠️ У (4) я СПОЧАТКУ отримав (t^4-10t^2+19) через помилку в x^3*He4 (36He3 замість 48He3);
    саме тому потрібна КВАДРАТУРА, а не другий олівець. Тож передбачення:
    всі чотири проходять, max rel.err ~1e-40 (A) і ~1e-14 (B). Шанс 90%.
    Найімовірніше місце провалу, якби він був — (4) при великих t (там поліном 4-го
    степеня множить phi(t), а гаусів член s*phi(t) малий => катастрофічне скорочення НЕ
    відбувається, але відносна вага поправки росте).

ПЛОМБИ (мусять пройти ДО вироку):
 P1  g4 = 0  =>  усі чотири = чисто гаусові вирази (ловить хворобу самої квадратури);
 P2  s=1, t=0  =>  E[ReLU] = phi(0) - (g4/24)*phi(0), phi(0)=0.3989422804014327;
 P3  1-однорідність: відносна похибка при s=1 і s=0.37 збігається до останньої цифри;
 P4  A vs B: два незалежні квадратурні правила дають те саме число.
"""
import sys
import numpy as np
from mpmath import mp, mpf, quad, exp, sqrt, pi, erfc, inf

mp.dps = 50

# ---------- еталонні спецфункції у високій точності ----------
def phi(x):
    x = mpf(x)
    return exp(-x * x / 2) / sqrt(2 * pi)

def Phi(x):
    x = mpf(x)
    return erfc(-x / sqrt(2)) / 2

def He(n, x):
    """Ймовірнісні поліноми Ерміта через рекурсію He_{n+1} = x He_n - n He_{n-1}."""
    h0, h1 = mpf(1), mpf(x)
    if n == 0:
        return h0
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1

def dens(x, g4):
    """Густина Еджворта 4-го порядку, gamma3 = 0."""
    return phi(x) * (1 + mpf(g4) * He(4, x) / 24)

# ---------- ЛІВІ частини: пряма квадратура (шлях A, mpmath) ----------
def lhs_mp(kind, s, t, g4):
    s, t = mpf(s), mpf(t)
    def f(x):
        r = s * (x + t)                       # ReLU(Y) на x > -t;  Y = s*(X+t)
        if kind == 1:
            w = r
        elif kind == 2:
            w = r * r
        elif kind == 3:
            w = r * He(1, x)
        elif kind == 4:
            w = r * He(2, x)
        return w * dens(x, g4)
    # межа кінка -t стоїть точкою розбиття; хвіст беремо до нескінченності
    return quad(f, [-t, -t + 1, -t + 5, inf])

# ---------- ЛІВІ частини: незалежне правило (шлях B, numpy Gauss-Legendre) ----------
_GLX, _GLW = np.polynomial.legendre.leggauss(40)
_NPAN = 60

def lhs_np(kind, s, t, g4):
    """Складене правило Гауса-Лежандра: 60 панелей x 40 вузлів на [-t, 14].
    Один 600-вузловий GL на широкому інтервалі ПРОВАЛЮЄТЬСЯ (6.7e-04) — вузький
    гаусів пік на довгому відрізку; це хвороба ПРИЛАДУ B, не формул."""
    s, t = float(s), float(t)
    a, b = -t, max(-t, 0.0) + 14.0
    edges = np.linspace(a, b, _NPAN + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    x = (0.5 * (hi - lo) * _GLX + 0.5 * (hi + lo)).ravel()
    w = (0.5 * (hi - lo) * _GLW).ravel()
    ph = np.exp(-x * x / 2) / np.sqrt(2 * np.pi)
    he4 = x**4 - 6 * x**2 + 3
    p = ph * (1 + g4 * he4 / 24)
    r = s * (x + t)
    if kind == 1:
        v = r
    elif kind == 2:
        v = r * r
    elif kind == 3:
        v = r * x
    elif kind == 4:
        v = r * (x * x - 1)
    terms = w * v * p
    val = float(np.sum(terms))
    # число обумовленості підсумовування: sum|terms| / |sum terms|.
    # Воно і є ПЕРЕДБАЧУВАНОЮ стелею точності float64: kappa * 2.2e-16.
    kappa = float(np.sum(np.abs(terms)) / abs(val)) if val != 0 else float("inf")
    return val, kappa

# ---------- ПРАВІ частини: замкнені форми ВИМІРУ ----------
def rhs(kind, s, t, g4):
    s, t, g = mpf(s), mpf(t), mpf(g4)
    p, P = phi(t), Phi(t)
    if kind == 1:
        return s * (p + t * P) + s * (g / 24) * (t * t - 1) * p
    if kind == 2:
        return s * s * ((1 + t * t) * P + t * p) - s * s * (g / 12) * t * p
    if kind == 3:
        return s * P - s * (g / 24) * p * (t**3 + t)
    if kind == 4:
        return s * p + s * (g / 24) * p * (t**4 + 2 * t * t + 7)

def rhs_gauss(kind, s, t):
    """Чисто гаусова частина — для пломби P1."""
    return rhs(kind, s, t, 0)

NAMES = {1: "E[ReLU]", 2: "E[ReLU^2]", 3: "a1=E[ReLU*He1]", 4: "a2=E[ReLU*He2]"}

TS = [-1, 0, 1, 2, 3.0, 3.5, 4.2, 5, 7]
G4S = [0.0383, 0.10, 0.2736, 0.3694, 0.5]
SIGMAS = [1.0, 0.37]


def relerr(l, r):
    l, r = mpf(l), mpf(r)
    if l == 0:
        return abs(l - r)
    return abs(l - r) / abs(l)


def main():
    print("=" * 96)
    print("edgchk97 — незалежна квадратурна пломба, mp.dps=%d" % mp.dps)
    print("=" * 96)

    # ---------------- P1: gamma4 = 0 => чисто гаусові вирази ----------------
    print("\n[P1] ПЛОМБА g4=0: квадратура vs гаусова замкнена форма (ловить хворобу приладу)")
    worst_p1 = mpf(0)
    for kind in (1, 2, 3, 4):
        for s in SIGMAS:
            for t in TS:
                e = relerr(lhs_mp(kind, s, t, 0.0), rhs_gauss(kind, s, t))
                worst_p1 = max(worst_p1, e)
    print("     max rel.err по 4x2x%d = %.3e   %s" % (len(TS), float(worst_p1),
          "PASS" if worst_p1 < mpf("1e-30") else "FAIL"))

    # ---------------- P2: s=1, t=0 ----------------
    print("\n[P2] ПЛОМБА s=1,t=0: E[ReLU] = phi(0)*(1 - g4/24), phi(0)=0.3989422804014327")
    worst_p2 = mpf(0)
    for g4 in G4S:
        got = lhs_mp(1, 1.0, 0.0, g4)
        want = phi(0) * (1 - mpf(g4) / 24)
        e = relerr(got, want)
        worst_p2 = max(worst_p2, e)
        print("     g4=%-7.4f  квадратура=%.16f  очікуване=%.16f  rel=%.2e"
              % (g4, float(got), float(want), float(e)))
    print("     %s" % ("PASS" if worst_p2 < mpf("1e-30") else "FAIL"))

    # ---------------- P4 + основна сітка ----------------
    print("\n[ОСНОВНА СІТКА] |ліва - права| / |ліва|,  ліва = квадратура mpmath")
    results = {}          # (kind, s) -> {(t,g4): err}
    ab = []               # (kind, s, t, g4, rel.diff A-vs-B, kappa)
    for kind in (1, 2, 3, 4):
        for s in SIGMAS:
            tab = {}
            for t in TS:
                for g4 in G4S:
                    L = lhs_mp(kind, s, t, g4)
                    R = rhs(kind, s, t, g4)
                    tab[(t, g4)] = relerr(L, R)
                    B, kap = lhs_np(kind, s, t, g4)
                    ab.append((kind, s, t, g4, relerr(L, B), kap))
            results[(kind, s)] = tab

    for kind in (1, 2, 3, 4):
        for s in SIGMAS:
            tab = results[(kind, s)]
            print("\n  (%d) %-16s sigma=%.2f" % (kind, NAMES[kind], s))
            print("      t \\ g4  " + "".join("%12.4f" % g for g in G4S))
            for t in TS:
                print("      %6.2f  " % t + "".join("%12.2e" % float(tab[(t, g)]) for g in G4S))
            mx = max(tab.values())
            print("      max = %.3e" % float(mx))

    print("\n[P4] ПЛОМБА двох квадратур: mpmath(dps=50) vs складений GL у float64.")
    print("     Критерій — НЕ абсолютний поріг, а СТЕЛЯ float64: kappa*2.2e-16, де")
    print("     kappa = sum|доданки| / |сума| (число обумовленості підсумовування).")
    worst_raw = max(r[4] for r in ab)
    worst_ratio, arg_ratio = mpf(0), None
    for kind, s, t, g4, d, kap in ab:
        ceil_ = mpf(kap) * mpf("2.2e-16")
        ratio = d / ceil_ if ceil_ > 0 else mpf(0)
        if ratio > worst_ratio:
            worst_ratio, arg_ratio = ratio, (kind, s, t, g4, float(d), kap)
    print("     max сира rel.diff = %.3e (без урахування kappa)" % float(worst_raw))
    print("     max diff/стеля    = %.3f  при (%d, sigma=%.2f, t=%.2f, g4=%.4f), "
          "diff=%.2e, kappa=%.2e" % (float(worst_ratio), arg_ratio[0], arg_ratio[1],
          arg_ratio[2], arg_ratio[3], arg_ratio[4], arg_ratio[5]))
    print("     %s" % ("PASS: обидві квадратури узгоджені В МЕЖАХ float64"
                       if worst_ratio < 100 else "FAIL: розбіжність ПОНАД стелю float64"))
    print("     найгірші 5 точок за сирою розбіжністю:")
    for kind, s, t, g4, d, kap in sorted(ab, key=lambda r: -r[4])[:5]:
        print("       (%d) sigma=%.2f t=%.2f g4=%.4f  diff=%.2e  kappa=%.2e  стеля=%.2e"
              % (kind, s, t, g4, float(d), kap, kap * 2.2e-16))

    # ---------------- P3: 1-однорідність ----------------
    print("\n[P3] ПЛОМБА 1-однорідності: rel.err при s=1 vs s=0.37 (мусить збігатись)")
    worst_hom = mpf(0)
    for kind in (1, 2, 3, 4):
        a, b = results[(kind, 1.0)], results[(kind, 0.37)]
        d = max(abs(a[k] - b[k]) for k in a)
        worst_hom = max(worst_hom, d)
        print("     (%d) %-16s max|err(s=1) - err(s=0.37)| = %.3e" % (kind, NAMES[kind], float(d)))
    print("     %s" % ("PASS" if worst_hom < mpf("1e-35") else
                       "УВАГА: відрізняються (див. числа вище)"))

    # ------- ПОБІЧНЕ: наскільки поправка велика відносно гаусового члена -------
    print("\n[ПОБІЧНЕ] |поправка| / |гаусів член| у робочій точці (не пломба, а масштаб ефекту)")
    print("      t \\ формула " + "".join("%14s" % NAMES[k] for k in (1, 2, 3, 4)))
    for t in (3.0, 3.5, 4.2, 5.0):
        for g4 in (0.2736, 0.3694):
            row = []
            for kind in (1, 2, 3, 4):
                G = rhs_gauss(kind, 1.0, t)
                F = rhs(kind, 1.0, t, g4)
                row.append(float(abs(F - G) / abs(G)))
            print("  t=%.1f g4=%.4f " % (t, g4) + "".join("%14.3e" % v for v in row))

    # ---------------- ВИРОК ----------------
    print("\n" + "=" * 96)
    print("ВИРОК ПО КОЖНІЙ ФОРМУЛІ (max відносна похибка по 2 sigma x %d t x %d g4 = %d точок)"
          % (len(TS), len(G4S), 2 * len(TS) * len(G4S)))
    TOL = mpf("1e-30")
    verdicts = {}
    for kind in (1, 2, 3, 4):
        mx = mpf(0)
        arg = None
        for s in SIGMAS:
            for k, v in results[(kind, s)].items():
                if v > mx:
                    mx, arg = v, (s, k[0], k[1])
        verdicts[kind] = (mx, arg)
        print("  (%d) %-16s max=%.3e  при (sigma=%.2f, t=%.2f, g4=%.4f)   %s"
              % (kind, NAMES[kind], float(mx), arg[0], arg[1], arg[2],
                 "ПРОХОДИТЬ" if mx < TOL else ">>> НЕ ПРОХОДИТЬ <<<"))
    bad = [k for k in verdicts if verdicts[k][0] >= TOL]
    print("\n  НЕ ПРОХОДЯТЬ: %s" % (bad if bad else "жодна — усі чотири підтверджені"))
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
