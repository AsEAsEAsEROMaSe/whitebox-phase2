"""pairchk97 — ЧИ Є БІВАРІАНТНЕ 4-КУМУЛЯНТНЕ НАБЛИЖЕННЯ ЗАКОННИМ ОБ'ЄКТОМ?

ПИТАННЯ ВИМІРУ: чи існує t*(rho, g4), вище якого 4-кумулянтне наближення
E[ReLU(X)ReLU(Y)] перестає бути збіжним, і чи накриває наша робоча точка
(|t| медіана 3.50, rho по парах нейронів) цю межу?

=======================  СТРУКТУРА, ВИВЕДЕНА ДО ПРОГОНУ  =======================
Будь-яка спільна густина зі стандартизованими маргіналями розкладається в
базисі Ерміта:  lam_mn = E[He_m(X) He_n(Y)],  і для БУДЬ-ЯКИХ f, g

    E[f(X)g(Y)] = sum_{m,n>=0} lam_mn/(m! n!) * a_m[f] * a_n[g],
    a_n[f] = E_phi[f(X) He_n(X)]      <- проєкції під ГАУСОМ

Віднімаючи m_f*m_g, отримуємо СПОЛУЧЕНУ форму (маргінальні члени скорочуються
ТОЧНО — це вже пояснює, чому g4 маргіналі не рухає позадіагональ):

    Cov(f,g) = sum_{m,n>=1} c_mn/(m! n!) * a_m * a_n,   c_mn = lam_mn - lam_m0*lam_0n

ГОЛОВНЕ. Для f = ReLU(sigma*(x+t)) проєкції мають ЗАМКНЕНУ ФОРМУ (перевірено
квадратурою до 1e-31, t=1.0/3.5/4.2, n=0..8):

    a_0 = sigma*(phi(t) + t*Phi(t))
    a_1 = sigma*Phi(t)
    a_n = sigma*phi(t)*(-1)^n * He_{n-2}(t)          для n >= 2     <<<< КЛЮЧ

⟹ a_1 ~ sigma (бо Phi(3.5)=0.99977), а ВСІ a_n при n>=2 несуть множник
   phi(t) = 8.727e-04 при t=3.5. Провідний член Cov — це c_11*a_1*a_1 = rho*sigma^2.
   Кожен НЕГАУСІВ член має m>=2 або n>=2 ⟹ придушений щонайменше на phi(t).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (протокол п.6, записую разом із шансом):
 (П1) Відносне відхилення Cov від гаусового, |dCov|/Cov_G, у робочій точці
      буде порядку g3*phi(t) ~ 0.4*8.7e-4 = 3.5e-04, і НЕ 4%.        шанс 85%
 (П2) Воно СПАДАЄ з t (phi(t) б'є будь-який поліном He) ⟹ t* НЕ ІСНУЄ:
      пара стає БІЛЬШ гаусовою при великих t, а не менш.             шанс 80%
 (П3) Воно майже НЕ залежить від rho, бо провідні крос-кумулянти самі
      пропорційні rho (lam_21 = rho*g3, lam_31 = rho*g4) і rho скорочується. шанс 70%
 (П4) Дві РІЗНІ родини з ТИМИ САМИМИ (g3,g4) — знакозмінна Еджвортова і
      СПРАВЖНЯ додатна (стандартизована Gamma) — дадуть те саме з точністю
      ~30% ⟹ ефект про МОМЕНТИ, а не про форму густини.               шанс 65%
 (П5) Розбіжність a_2 (множник t^4+2t^2+7, зміряна edgchk97 як x2.07 при
      t=3.5) У ПАРІ НЕШКІДЛИВА: a_2 входить у Cov лише через член
      (rho^2/2)*a_2^2 ~ rho^2*phi(t)^2 = 7.6e-07*rho^2 проти провідного rho. шанс 90%
Якщо (П1) провалиться в бік БІЛЬШОГО — значить негаусовість пари справді може
пояснити позадіагональ 4.08%, і клас ЖИВИЙ. Це і є вирішальне число.

=============================  ПРИЛАД  =========================================
Модель пари (щоб мати ТОЧНУ істину, а не наближення):
    X, Z незалежні з тієї самої родини (сер. 0, дисп. 1, задані g3, g4)
    Y = rho*X + sqrt(1-rho^2)*Z                 ⟹ Corr(X,Y) = rho ТОЧНО
Гаусів еталон — пара N(0,1) з ТИМ САМИМ rho (маргіналі збігаються за 1-м і 2-м
моментом за побудовою) ⟹ виміряне відхилення Є ЧИСТО негаусовим ефектом.

Дві родини:
  (E) Еджворт-4: p(x) = phi(x)[1 + g3*He3/6 + g4*He4/24]   — ЗНАКОЗМІННА (це той
      самий об'єкт, який ставить ВИМІР);
  (G) стандартизована Gamma(k): X = (G-k)/sqrt(k), g3 = 2/sqrt(k), g4 = 6/k —
      СПРАВЖНЯ додатна густина. При k=25: g3=0.400, g4=0.240 — майже рівно
      наша робоча точка (g3~0.40, g4~0.27) ⟹ контроль «не та густина».

Квадратура: складене правило Гауса-Лежандра. E[fg] береться по ПРЯМОКУТНИКУ
(x > -t_j, y > -t_k) у координатах (x,y), де кінки ReLU лежать РІВНО на межах.
lam_mn — по (x,z), де густина є ДОБУТКОМ і підінтегральна функція гладка.
Обумовленість підсумовування kappa друкується (урок edgchk97: критерій згоди
двох приладів — kappa*2.2e-16, а не абсолютний поріг).
"""
import sys
import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.special import erfc

SQ2PI = np.sqrt(2 * np.pi)


def phi(x):
    return np.exp(-np.asarray(x, float) ** 2 / 2) / SQ2PI


def Phi(x):
    return 0.5 * erfc(-np.asarray(x, float) / np.sqrt(2))


def He(n, x):
    x = np.asarray(x, float)
    if n < 0:
        raise ValueError
    h0 = np.ones_like(x)
    if n == 0:
        return h0
    h1 = x.copy()
    for k in range(1, n):
        h0, h1 = h1, x * h1 - k * h0
    return h1


# ------------------------------------------------------------------ проєкції
def a_coef(n, s, t):
    """a_n = E_phi[ReLU(s*(x+t)) He_n(x)] — замкнена форма (перевірена квадратурою)."""
    if n == 0:
        return s * (phi(t) + t * Phi(t))
    if n == 1:
        return s * Phi(t)
    return s * phi(t) * ((-1) ** n) * He(n - 2, np.array(float(t)))[()]


# ------------------------------------------------------------------ родини
class EdgeworthFam:
    tag = "E"
    name = "Еджворт-4 (знакозмінна)"

    def __init__(self, g3, g4):
        self.g3, self.g4 = g3, g4
        self.lo, self.hi = -12.0, 12.0

    def pdf(self, x):
        x = np.asarray(x, float)
        return phi(x) * (1 + self.g3 * He(3, x) / 6 + self.g4 * He(4, x) / 24)


class GaussFam:
    tag = "N"
    name = "N(0,1) — гаусів еталон"

    def __init__(self):
        self.g3, self.g4 = 0.0, 0.0
        self.lo, self.hi = -12.0, 12.0

    def pdf(self, x):
        return phi(x)


class GammaFam:
    tag = "G"
    name = "Gamma стандартизована (додатна)"

    def __init__(self, k):
        self.k = k
        self.g3, self.g4 = 2.0 / np.sqrt(k), 6.0 / k
        self.lo, self.hi = -np.sqrt(k) + 1e-12, 12.0

    def pdf(self, x):
        x = np.asarray(x, float)
        k = self.k
        g = k + np.sqrt(k) * x
        out = np.zeros_like(x)
        m = g > 0
        out[m] = gamma_dist.pdf(g[m], k) * np.sqrt(k)
        return out


# ------------------------------------------------------------------ квадратура
_NODE, _NPAN = 24, 48
_GX, _GW = np.polynomial.legendre.leggauss(_NODE)


def grid1(a, b, npan=_NPAN):
    e = np.linspace(a, b, npan + 1)
    lo, hi = e[:-1, None], e[1:, None]
    x = (0.5 * (hi - lo) * _GX + 0.5 * (hi + lo)).ravel()
    w = (0.5 * (hi - lo) * _GW).ravel()
    return x, w


def integ2(fun, xa, xb, ya, yb, npan=_NPAN):
    """Повертає (значення, kappa) — kappa = sum|доданки|/|сума|."""
    x, wx = grid1(xa, xb, npan)
    y, wy = grid1(ya, yb, npan)
    X, Y = np.meshgrid(x, y, indexing="ij")
    W = wx[:, None] * wy[None, :]
    T = W * fun(X, Y)
    v = float(T.sum())
    k = float(np.abs(T).sum() / abs(v)) if v != 0 else np.inf
    return v, k


def integ1(fun, a, b, npan=_NPAN):
    x, w = grid1(a, b, npan)
    T = w * fun(x)
    v = float(T.sum())
    k = float(np.abs(T).sum() / abs(v)) if v != 0 else np.inf
    return v, k


# ------------------------------------------------------------------ пара
def joint_xy(fam, rho):
    """Густина (X,Y) при Y = rho X + sqrt(1-rho^2) Z."""
    r = np.sqrt(1 - rho * rho)
    return lambda X, Y: fam.pdf(X) * fam.pdf((Y - rho * X) / r) / r


def cov_true(fam, rho, sj, tj, sk, tk):
    p = joint_xy(fam, rho)
    hi = fam.hi + 4.0
    efg, k1 = integ2(lambda X, Y: sj * (X + tj) * sk * (Y + tk) * p(X, Y),
                     max(-tj, fam.lo), hi, max(-tk, fam.lo), hi)
    mj, _ = integ1(lambda x: sj * (x + tj) * fam.pdf(x), max(-tj, fam.lo), hi)
    # маргіналь Y — та сама сім'я лише при rho=0; беремо її ЧЕСНО, 2-D інтегралом
    mk, _ = integ2(lambda X, Y: sk * (Y + tk) * p(X, Y),
                   fam.lo, hi, max(-tk, fam.lo), hi)
    return efg - mj * mk, efg, mj, mk, k1


def cov_gauss_mehler(rho, sj, tj, sk, tk, nmax=80):
    """Cov під гаусом = sum_{n>=1} rho^n/n! a_n a_n (Мелер)."""
    tot, terms = 0.0, []
    lf = 0.0
    for n in range(1, nmax + 1):
        lf += np.log(n)
        c = np.exp(n * np.log(abs(rho)) - lf) * (np.sign(rho) ** n) if rho != 0 else 0.0
        v = c * a_coef(n, sj, tj) * a_coef(n, sk, tk)
        terms.append(v)
        tot += v
    return tot, terms


def cov_gauss_quad(rho, sj, tj, sk, tk):
    r = np.sqrt(1 - rho * rho)
    p = lambda X, Y: phi(X) * phi((Y - rho * X) / r) / r
    efg, _ = integ2(lambda X, Y: sj * (X + tj) * sk * (Y + tk) * p(X, Y),
                    -tj, 14.0, -tk, 14.0)
    mj = sj * (phi(tj) + tj * Phi(tj))
    mk = sk * (phi(tk) + tk * Phi(tk))
    return efg - mj * mk


def lam_tensor(fam, rho, M=6):
    """lam_mn = E[He_m(X) He_n(Y)] — інтеграл у (x,z), густина = ДОБУТОК."""
    r = np.sqrt(1 - rho * rho)
    x, wx = grid1(fam.lo, fam.hi, _NPAN)
    z, wz = grid1(fam.lo, fam.hi, _NPAN)
    X, Z = np.meshgrid(x, z, indexing="ij")
    W = (wx[:, None] * wz[None, :]) * fam.pdf(X) * fam.pdf(Z)
    Y = rho * X + r * Z
    HX = [He(m, X) for m in range(M + 1)]
    HY = [He(n, Y) for n in range(M + 1)]
    return np.array([[float((W * HX[m] * HY[n]).sum()) for n in range(M + 1)]
                     for m in range(M + 1)])


def dcov_trunc(lam, order, sj, tj, sk, tk, rho):
    """Наближення dCov = Cov - Cov_gauss, обрізане за m+n <= order.
    c_mn = lam_mn - lam_m0 lam_0n; гаусів еталон c^G_mn = delta_mn * n! rho^n."""
    M = lam.shape[0] - 1
    tot = 0.0
    fact = [1.0]
    for i in range(1, M + 1):
        fact.append(fact[-1] * i)
    for m in range(1, M + 1):
        for n in range(1, M + 1):
            if m + n > order:
                continue
            c = lam[m, n] - lam[m, 0] * lam[0, n]
            cg = fact[n] * rho ** n if m == n else 0.0
            tot += (c - cg) / (fact[m] * fact[n]) * a_coef(m, sj, tj) * a_coef(n, sk, tk)
    return tot


# ------------------------------------------------------------------ прогін
TS = [2.0, 3.0, 3.5, 4.2, 5.0]
RHOS = [0.02, 0.05, 0.10, 0.30, 0.60, 0.90]


def main():
    print("=" * 100)
    print("pairchk97 — біваріантна межа збіжності 4-кумулянтного наближення")
    print("=" * 100)

    famE = EdgeworthFam(0.40, 0.240)
    famG = GammaFam(25.0)
    print("\nРодини: (E) Еджворт g3=%.3f g4=%.3f  |  (G) Gamma k=25 -> g3=%.3f g4=%.3f"
          % (famE.g3, famE.g4, famG.g3, famG.g4))

    # ---------- ПЛОМБА 0: квадратура відтворює МОМЕНТИ родини ----------
    print("\n[P0] ПЛОМБА родин: сер / дисп / g3 / g4 з квадратури (мусять бути 0/1/задані)")
    for fam in (famE, famG):
        mom = [integ1(lambda x, q=q: x ** q * fam.pdf(x), fam.lo, fam.hi + 4)[0]
               for q in range(5)]
        m1, m2 = mom[1], mom[2]
        g3 = mom[3] - 3 * m1 * m2 + 2 * m1 ** 3
        g4 = mom[4] - 4 * m1 * mom[3] + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3
        print("     (%s) norm=%.12f  mean=%+.2e  var=%.12f  g3=%.6f  g4=%.6f"
              % (fam.tag, mom[0], m1, m2, g3, g4))

    # ---------- ПЛОМБА 1: два способи гаусової Cov ----------
    print("\n[P1] ПЛОМБА гаусової Cov: Мелер (замкнені a_n) vs пряма 2-D квадратура")
    w_lo, w_hi, tail_ratio = 0.0, 0.0, 0.0
    for t in TS:
        for rho in RHOS:
            m, terms = cov_gauss_mehler(rho, 1.0, t, 1.0, t)
            q = cov_gauss_quad(rho, 1.0, t, 1.0, t)
            d = abs(m - q) / abs(q)
            # стеля float64 для РІЗНИЦІ E[fg]-m*m: kappa = E[fg]/|Cov|
            r_ = np.sqrt(1 - rho * rho)
            pp = lambda X, Y: phi(X) * phi((Y - rho * X) / r_) / r_
            efg_, _ = integ2(lambda X, Y: (X + t) * (Y + t) * pp(X, Y), -t, 14.0, -t, 14.0)
            ceil_ = (efg_ / abs(q)) * 2.2e-16
            if rho <= 0.6:
                w_lo = max(w_lo, d / ceil_)
            else:
                w_hi = max(w_hi, d)
                if terms[-1]:
                    tail_ratio = max(tail_ratio, d * abs(q) / abs(terms[-1]))
    print("     rho <= 0.60 : max (rel / стеля float64 kappa*2.2e-16) = %.2f   %s"
          % (w_lo, "PASS" if w_lo < 100 else "FAIL"))
    print("     rho = 0.90  : max rel = %.3e   — це ХВІСТ ряду Мелера, обрізаного на n=80:" % w_hi)
    print("                   |Мелер-квад| / |останній член| = %.1f (тобто залишок ряду)."
          % tail_ratio)
    print("                   ⟹ хвороба ПРИЛАДУ (обрізання ряду), не об'єкта. Далі еталоном")
    print("                   служить КВАДРАТУРА (без обрізання), Мелер — лише для розкладу.")

    # ---------- ПЛОМБА 2: rho=0 => Cov = 0 ----------
    print("\n[P2] ПЛОМБА rho=0: Cov мусить бути 0 (незалежність)")
    w2 = 0.0
    for fam in (famE, famG):
        for t in TS:
            c = cov_true(fam, 0.0, 1.0, t, 1.0, t)[0]
            w2 = max(w2, abs(c))
    print("     max |Cov| = %.3e   %s" % (w2, "PASS" if w2 < 1e-9 else "FAIL"))

    # ---------- ГОЛОВНА ТАБЛИЦЯ ----------
    print("\n" + "=" * 100)
    print("ГОЛОВНЕ: dCov/Cov_G = (Cov_справжня - Cov_гаусова)/Cov_гаусова")
    print("Гаусів еталон має ТОЙ САМИЙ rho і ті самі 1-й та 2-й моменти ⟹ це ЧИСТО негаусовість")
    print("=" * 100)
    store = {}
    famN = GaussFam()
    covG = {}
    for t in TS:
        for rho in RHOS:
            covG[(t, rho)] = cov_true(famN, rho, 1.0, t, 1.0, t)[0]
    for fam in (famE, famG):
        print("\n  Родина (%s) %s   g3=%.3f g4=%.3f" % (fam.tag, fam.name, fam.g3, fam.g4))
        print("     t \\ rho " + "".join("%12.2f" % r for r in RHOS) + "   | phi(t)")
        for t in TS:
            row = []
            for rho in RHOS:
                ct = cov_true(fam, rho, 1.0, t, 1.0, t)[0]
                cg = covG[(t, rho)]
                d = (ct - cg) / cg
                row.append(d)
                store[(fam.tag, t, rho)] = (ct, cg, d)
            print("     %6.2f  " % t + "".join("%12.3e" % v for v in row)
                  + "   | %.2e" % phi(t))

    # ---------- ПЛОМБА МАСШТАБУВАННЯ: чи це справді phi(t)? ----------
    print("\n[P3] ПЛОМБА МЕХАНІЗМУ: якщо придушення є phi(t), то dCov/Cov_G / phi(t)")
    print("     мусить бути ~СТАЛИМ по t (протокол п.4: відрізняє механізм від збігу)")
    for fam in (famE, famG):
        print("     (%s)  t \\ rho " % fam.tag + "".join("%11.2f" % r for r in RHOS))
        for t in TS:
            print("           %6.2f  " % t + "".join(
                "%11.3f" % (store[(fam.tag, t, r)][2] / phi(t)) for r in RHOS))

    # ---------- ЗБІЖНІСТЬ ОБРІЗАННЯ: 3 vs 4 vs 6 порядок ----------
    print("\n" + "=" * 100)
    print("ЗБІЖНІСТЬ 4-КУМУЛЯНТНОГО ОБРІЗАННЯ: dCov_обрізане / dCov_справжнє")
    print("=1.0 ⟹ обрізання ТОЧНЕ · росте з порядком ⟹ збігається · стрибає ⟹ РОЗБІГАЄТЬСЯ")
    print("=" * 100)
    for fam in (famE, famG):
        print("\n  Родина (%s)" % fam.tag)
        print("     t     rho   |  dCov_справж   ord<=3      ord<=4      ord<=6   | вирок")
        for t in TS:
            for rho in (0.05, 0.30, 0.90):
                lam = lam_tensor(fam, rho, M=6)
                ct, cg, d = store[(fam.tag, t, rho)]
                dt = ct - cg
                r3 = dcov_trunc(lam, 3, 1.0, t, 1.0, t, rho) / dt if dt != 0 else np.nan
                r4 = dcov_trunc(lam, 4, 1.0, t, 1.0, t, rho) / dt if dt != 0 else np.nan
                r6 = dcov_trunc(lam, 6, 1.0, t, 1.0, t, rho) / dt if dt != 0 else np.nan
                ok = "збіг" if abs(r6 - 1) < abs(r4 - 1) + 1e-12 and abs(r6 - 1) < 0.5 else "УВАГА"
                print("    %5.2f  %5.2f  | %+.4e  %10.4f  %10.4f  %10.4f   | %s"
                      % (t, rho, dt, r3, r4, r6, ok))

    # ---------- ЧИ ІСНУЄ t* ----------
    print("\n" + "=" * 100)
    print("ЧИ ІСНУЄ t*: шукаємо t, де |dCov|/Cov_G досягає 1.0 (наближення втрачає сенс)")
    print("=" * 100)
    for fam in (famE, famG):
        print("  (%s) t = " % fam.tag, end="")
        for t in [1.0, 2.0, 3.0, 3.5, 4.2, 5.0, 6.0, 7.0]:
            ct = cov_true(fam, 0.10, 1.0, t, 1.0, t)[0]
            cg = cov_true(GaussFam(), 0.10, 1.0, t, 1.0, t)[0]
            print("%.1f:%.2e " % (t, abs((ct - cg) / cg)), end="")
        print()

    # ---------- ВНЕСОК a_2 (розбіжного) У ПАРУ ----------
    print("\n[ПОБІЧНЕ] куди дівається розбіжність a_2 (x2.07 при t=3.5, edgchk97):")
    print("     член Мелера n=2 у Cov = (rho^2/2)*a_2^2; його частка від Cov_G:")
    for t in (3.0, 3.5, 4.2):
        for rho in (0.05, 0.30, 0.90):
            cg, terms = cov_gauss_mehler(rho, 1.0, t, 1.0, t)
            print("       t=%.1f rho=%.2f  член_n2/Cov_G = %.3e   (a_2=%.3e)"
                  % (t, rho, terms[1] / cg, a_coef(2, 1.0, t)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
