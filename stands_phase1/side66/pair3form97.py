"""pair3form97 — ЗАМКНЕНА ФОРМА `E[ReLU(p_j)·ReLU(p_k)]` З КРОС-КУМУЛЯНТАМИ
3-го ПОРЯДКУ. Запит КОНТРОЛЕРА: дати D формулу замість двовимірної квадратури.

════════════════ ВИВІД (не вгадано — виведено з тотожності) ════════════════
Розклад густини навколо гауса з ТИМИ САМИМИ (m, C), перший порядок по κ₃:
    p = p_G − (1/6)[κ₃₀₀∂ₓ³ + 3κ₂₁₀∂ₓ²∂_y + 3κ₁₂₀∂ₓ∂_y² + κ₀₃₀∂_y³] p_G
інтегрування частинами ⟹ для БУДЬ-ЯКОЇ f:
    E[f] = E_G[f] + (1/6)[γ_j·E_G[f_xxx] + 3λ_jjk·E_G[f_xxy]
                          + 3λ_jkk·E_G[f_xyy] + γ_k·E_G[f_yyy]]

🔑 КЛЮЧОВЕ СПОСТЕРЕЖЕННЯ, ЩО РОБИТЬ ФОРМУ ЗАМКНЕНОЮ:
    f = (x+t_j)₊(y+t_k)₊  ⟹  ∂ₓ²f = δ(x+t_j)·(y+t_k)₊ ,  ∂ₓ³f = δ′(x+t_j)·(…)
δ-функція ЗГОРТАЄ один вимір ТОЧНО ⟹ поправка виражається ЛИШЕ через
ОДНОВИМІРНІ φ, Φ, ψ=φ+tΦ. Поправка ПРОСТІША за сам гаусів член.

    q   = √(1−ρ²) ,  u_k = (t_k−ρt_j)/q ,  u_j = (t_j−ρt_k)/q
    E_G[f_xxy] = φ(t_j)·Φ(u_k)                      ≡ B_j
    E_G[f_xyy] = φ(t_k)·Φ(u_j)                      ≡ B_k
    E_G[f_xxx] = −φ(t_j)[t_j·q·ψ(u_k) + ρ·Φ(u_k)]   ≡ A_j
    E_G[f_yyy] = −φ(t_k)[t_k·q·ψ(u_j) + ρ·Φ(u_j)]   ≡ A_k

════════════════ ПЛОМБИ ════════════════
 S1 🔒 ЗАПИТ КОНТРОЛЕРА: κ₃ = 0 ⟹ РІВНО гаусів Прайс (за побудовою + чисельно)
 S2 🔒 ρ→0 мусить відтворити ЗАПЕЧАТАНУ 1-D форму `ψ(t) − (γ₃/6)tφ(t)`
      — це пломба на АЛГЕБРУ проти результату, вже запечатаного до 1e-44
 S3 🔒 кожна з чотирьох замкнених форм проти НЕЗАЛЕЖНОЇ 1-D квадратури
 S4 🔒 наскрізна: справжня НЕгаусова пара з ТОЧНО відомими кумулянтами,
      істина машинної точності ⟹ формула мусить бити гаусову
 S5 🔒 ПЛОМБА МАСШТАБУВАННЯ: залишок O(κ₃²) ⟹ при κ₃ ×s залишок/поправка ∝ s³

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) S2 зійдеться АНАЛІТИЧНО, до 1e-14                            шанс 90%
 (б) S4: формула зніме >70% похибки гауса в робочій точці         шанс 65%
 (в) поправка в робочій точці (|λ₂₁|=0.2835, t≈3.5, ρ≈0.97) буде
     1…5% від гаусового значення — тобто того ж порядку, що й
     зміряна похибка позадіагоналі 4.08%                          шанс 50%
 (г) 🚨 при ρ→1 форма стане ЧУТЛИВОЮ (q→0, u→±∞) — можлива втрата
     точності саме в робочій точці ρ=0.974                        шанс 40%
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)


def phi(x):
    return np.exp(-0.5 * np.asarray(x, dtype=np.float64) ** 2) / SQ2PI


def Phi(x):
    from scipy.special import ndtr
    return ndtr(np.asarray(x, dtype=np.float64))


def psi(x):
    return phi(x) + np.asarray(x, dtype=np.float64) * Phi(x)


# ───────────────────── КВАДРАТУРА (складена, вузли на зламах) ─────────────
def gl(a, b, npan=64, nnod=40):
    """складене Гауса–Лежандра: вузли й ваги на [a,b]."""
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    X = (0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel()
    W = (0.5 * (hi - lo) * ws[None, :]).ravel()
    return X, W


def price_gauss(tj, tk, rho, npan=80, nnod=40, span=42.0):
    """ТОЧНЕ E[(x+t_j)₊(y+t_k)₊] для стандартної гаусової пари.
    Зведено до ОДНОВИМІРНОГО інтеграла аналітично ⟹ злам поза областю:
        ∫_{−t_j}^∞ (x+t_j)φ(x)·q·ψ((ρx+t_k)/q) dx
    """
    q = math.sqrt(max(1.0 - rho * rho, 0.0))
    if q == 0.0:
        raise ValueError("ρ=±1 вироджено")
    X, W = gl(-tj, -tj + span, npan, nnod)
    return float(np.sum(W * (X + tj) * phi(X) * q * psi((rho * X + tk) / q)))


# ───────────────────── ЗАМКНЕНА ФОРМА ─────────────────────
def price_k3(tj, tk, rho, gj=0.0, ljjk=0.0, ljkk=0.0, gk=0.0, lam=1.0,
             pg=None):
    """Гаусів Прайс + поправка 1-го порядку по третіх кумулянтах.
    Усі аргументи СТАНДАРТИЗОВАНІ: t=m/s, ρ=corr, γ=κ₃₀₀/s³,
    λ_jjk = κ(x,x,y)/(s_j² s_k), λ_jkk = κ(x,y,y)/(s_j s_k²).
    `lam` — сила поправки (для сканів і контролю «чужа»).
    """
    q = math.sqrt(max(1.0 - rho * rho, 0.0))
    uk = (tk - rho * tj) / q
    uj = (tj - rho * tk) / q
    Bj = phi(tj) * Phi(uk)
    Bk = phi(tk) * Phi(uj)
    Aj = -phi(tj) * (tj * q * psi(uk) + rho * Phi(uk))
    Ak = -phi(tk) * (tk * q * psi(uj) + rho * Phi(uj))
    corr = (gj * Aj + 3.0 * ljjk * Bj + 3.0 * ljkk * Bk + gk * Ak) / 6.0
    base = price_gauss(tj, tk, rho) if pg is None else pg
    return base + lam * corr, base, corr


print("pair3form97 — замкнена форма для пари з κ₃\n" + "=" * 84)

# ═════════════ S3: КОЖНА ЗАМКНЕНА ФОРМА ПРОТИ НЕЗАЛЕЖНОЇ 1-D КВАДРАТУРИ ═══
print("S3 🔒 чотири замкнені форми проти прямої 1-D квадратури")
print(f"{'t_j':>6s}{'t_k':>6s}{'ρ':>7s}{'відн.похибка B_j':>18s}"
      f"{'A_j':>14s}{'B_k':>12s}{'A_k':>12s}")
worst3 = 0.0
for tj, tk, rho in ((3.5, 3.5, 0.974), (3.5, 2.0, 0.90), (4.2, 4.2, 0.99),
                    (1.0, -0.5, 0.30), (0.0, 0.0, 0.0), (2.0, 3.0, -0.60)):
    q = math.sqrt(1.0 - rho * rho)
    uk, uj = (tk - rho * tj) / q, (tj - rho * tk) / q
    # B_j = E_G[δ(x+t_j)·1_{y>−t_k}] = ∫_{−t_k}^∞ p_G(−t_j, y) dy
    Y, WY = gl(-tk, -tk + 42.0, 80, 40)
    pg_line = phi(-tj) * phi((Y - rho * (-tj)) / q) / q
    Bj_q = float(np.sum(WY * pg_line))
    # B_k симетрично
    X, WX = gl(-tj, -tj + 42.0, 80, 40)
    pg_lin2 = phi(-tk) * phi((X - rho * (-tk)) / q) / q
    Bk_q = float(np.sum(WX * pg_lin2))
    # A_j = −∂ₓ∫(y+t_k)₊p_G dy |_{x=−t_j}; ∂ₓp_G = −((x−ρy)/q²)p_G
    Y2, WY2 = gl(-tk, -tk + 42.0, 80, 40)
    pg2 = phi(-tj) * phi((Y2 - rho * (-tj)) / q) / q
    Aj_q = -float(np.sum(WY2 * (Y2 + tk) * (-((-tj) - rho * Y2) / q ** 2)
                         * pg2))
    X2, WX2 = gl(-tj, -tj + 42.0, 80, 40)
    pg3 = phi(-tk) * phi((X2 - rho * (-tk)) / q) / q
    Ak_q = -float(np.sum(WX2 * (X2 + tj) * (-((-tk) - rho * X2) / q ** 2)
                         * pg3))
    Bj = float(phi(tj) * Phi(uk))
    Bk = float(phi(tk) * Phi(uj))
    Aj = float(-phi(tj) * (tj * q * psi(uk) + rho * Phi(uk)))
    Ak = float(-phi(tk) * (tk * q * psi(uj) + rho * Phi(uj)))

    def rel(a, b):
        return abs(a - b) / max(abs(b), 1e-300)
    e = (rel(Bj, Bj_q), rel(Aj, Aj_q), rel(Bk, Bk_q), rel(Ak, Ak_q))
    worst3 = max(worst3, max(e))
    print(f"{tj:>6.2f}{tk:>6.2f}{rho:>7.3f}{e[0]:>18.2e}{e[1]:>14.2e}"
          f"{e[2]:>12.2e}{e[3]:>12.2e}")
print(f"  ⟹ найгірша відносна похибка {worst3:.2e}   "
      f"{'✅' if worst3 < 1e-10 else '❌'}")

# ═════════════ S2: ρ→0 МУСИТЬ ВІДТВОРИТИ ЗАПЕЧАТАНУ 1-D ФОРМУ ═════════════
print("\nS2 🔒 ρ→0 проти запечатаної 1-D форми  E[ReLU]=ψ(t)−(γ₃/6)tφ(t)")
print(f"{'t_j':>6s}{'t_k':>6s}{'γ_j':>7s}{'формула пари':>16s}"
      f"{'добуток 1-D':>16s}{'відн.':>11s}")
worst2 = 0.0
for tj, tk, gj in ((3.5, 3.0, 0.30), (2.0, 4.0, -0.40), (0.5, 0.5, 0.20)):
    rho = 1e-9
    v, _, _ = price_k3(tj, tk, rho, gj=gj)
    ref = (psi(tj) - gj / 6.0 * tj * phi(tj)) * psi(tk)
    r = abs(v - float(ref)) / abs(float(ref))
    worst2 = max(worst2, r)
    print(f"{tj:>6.2f}{tk:>6.2f}{gj:>7.2f}{v:>16.10f}{float(ref):>16.10f}"
          f"{r:>11.2e}")
print(f"  ⟹ {worst2:.2e}  "
      f"{'✅ алгебра збігається з незалежно запечатаною' if worst2 < 1e-9 else '❌'}")

# ═════════════ S1: κ₃ = 0 ⟹ РІВНО ГАУСІВ ПРАЙС ═════════════
v0, b0, c0 = price_k3(3.5, 3.5, 0.974, 0.0, 0.0, 0.0, 0.0)
print(f"\nS1 🔒 κ₃=0: поправка = {c0:.3e} · |нове−гаусове| = {abs(v0-b0):.3e}"
      f"   {'✅' if c0 == 0.0 else '❌'}")

# ═════════════ S4/S5: НАСКРІЗНА НА СПРАВЖНІЙ НЕГАУСОВІЙ ПАРІ ═════════════
# Побудова з ТОЧНО відомими кумулянтами:
#   X = g1 + c·w ,  Y = ρ′g1 + √(1−ρ′²)g2 + d·w ,  w = g3²−1  (κ₂=2, κ₃=8)
# Умовно на g3 пара ГАУСОВА ⟹ істина = 1-D інтеграл від price_gauss.
def truth_pair(Mj, Mk, c, d, rp, npan=60, nnod=32, span=9.0):
    G, WG = gl(-span, span, npan, nnod)
    out = 0.0
    for g, wt in zip(G, WG):
        w = g * g - 1.0
        out += wt * phi(g) * price_gauss(Mj + c * w, Mk + d * w, rp)
    return out


def moments_pair(Mj, Mk, c, d, rp):
    sj = math.sqrt(1.0 + 2.0 * c * c)
    sk = math.sqrt(1.0 + 2.0 * d * d)
    cov = rp + 2.0 * c * d
    rho = cov / (sj * sk)
    return (Mj / sj, Mk / sk, rho,
            8.0 * c ** 3 / sj ** 3, 8.0 * c * c * d / (sj * sj * sk),
            8.0 * c * d * d / (sj * sk * sk), 8.0 * d ** 3 / sk ** 3,
            sj, sk)


print("\nS4/S5 🔒 наскрізна: справжня негаусова пара, істина машинної точності")
print(f"{'масштаб s':>10s}{'|λ₂₁|':>9s}{'похиб.ГАУС':>13s}"
      f"{'похиб.+κ₃':>12s}{'знято %':>10s}{'залишок/попр':>14s}")
Mj, Mk, rp = 3.5, 3.4, 0.90
prev = None
for s in (1.00, 0.60, 0.35, 0.20):
    c, dd = 0.45 * s, 0.30 * s
    T = truth_pair(Mj, Mk, c, dd, rp)
    tj, tk, rho, gj, ljjk, ljkk, gk, sj, sk = moments_pair(Mj, Mk, c, dd, rp)
    vk, vg, corr = price_k3(tj, tk, rho, gj, ljjk, ljkk, gk)
    eg = abs(vg * sj * sk - T) / T
    ek = abs(vk * sj * sk - T) / T
    ratio = abs(vk * sj * sk - T) / max(abs(corr * sj * sk), 1e-300)
    print(f"{s:>10.2f}{abs(ljjk):>9.4f}{eg:>13.3e}{ek:>12.3e}"
          f"{100*(1-ek/eg):>9.1f}%{ratio:>14.3e}")
    prev = (s, ratio)
print("  🔒 S5: якщо залишок Є O(κ₃²), стовпець «залишок/попр» мусить падати "
      "≈ як s³")

# ═════════════ РОБОЧА ТОЧКА: ЧИ ВЗАГАЛІ ВАРТО ═════════════
print("\n" + "=" * 84)
print("РОБОЧА ТОЧКА (t≈3.5, ρ=0.974, |λ₂₁| медіана 0.2835 з датасету 1e8)")
print(f"{'ρ':>7s}{'λ_jjk':>9s}{'гаусів Прайс':>15s}{'поправка κ₃':>14s}"
      f"{'частка':>9s}")
for rho in (0.974, 0.90, 0.70):
    for L in (0.2835, -0.2835):
        v, b, cr = price_k3(3.5, 3.5, rho, 0.0, L, L, 0.0)
        print(f"{rho:>7.3f}{L:>9.4f}{b:>15.8f}{cr:>14.3e}"
              f"{100*abs(cr)/b:>8.3f}%")
print("\n📌 порівняти з зміряною похибкою ПОЗАДІАГОНАЛІ замикання = 4.08%")
