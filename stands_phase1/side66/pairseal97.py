"""pairseal97 — ДВІ ЗВІРКИ, ЯКИХ ВИМАГАВ КОНТРОЛЕР. ОБИДВІ РОЗСУДЖУЮТЬ ЧИСЛОМ.

════════════ ЗВІРКА 1: СУПЕРЕЧКА ПРО `Φ₂` ════════════
D каже: коефіцієнт `t₁t₂+ρ` не нуль ⟹ `Φ₂` НЕ прибирається.
Я кажу:  злам виходить за межі області ⟹ 2-D квадратура не потрібна.
🔑 ПІДОЗРА: ми говоримо про РІЗНІ ТВЕРДЖЕННЯ, і обидва вірні.
   D — про ПОВНІСТЮ замкнену форму (там `Φ₂` справді лишається).
   Я — про зведення до ОДНОВИМІРНОГО інтеграла (це не «замкнена форма»).
⟹ перевіряємо МОЄ твердження, бо саме воно не пломбоване: 1-D зведення
   проти ЧЕСНОЇ 2-D квадратури на 6 точках, включно з ρ=0.99.

════════════ ЗВІРКА 2: 88% (я) ПРОТИ 9.5% (D) — РОЗРИВ ×9 ════════════
Гіпотеза контролера: різні метрики. Перевіряємо ПРЯМО, на ОДНОМУ зрізі,
друкуючи чотири метрики поруч:
   (M1) частка знятої RMS похибки впорскуванням поправки (моя)
   (M2) R² регресії похибки на СИРИЙ λ₂₁ (ймовірно її)
   (M3) R² регресії похибки на ЧЛЕН ПОПРАВКИ λ₂₁·B_j (той самий об'єкт,
        але з теоретичним ядром)
   (M4) те саме для маргінального скосу γ
🔑 ЯКЩО M2 ≪ M3, причина названа: ядро `φ(t_j)Φ(u_k)` гуляє на порядки
   по парах, і лінійна регресія на сирий `λ₂₁` його не бачить.

ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ)
 (а) 1-D зведення зійдеться з 2-D квадратурою до <1e-12        шанс 92%
 (б) M2 буде в рази менша за M1 і M3                            шанс 75%
 (в) M3 буде близька до M1 (обидві бачать той самий об'єкт)     шанс 65%
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
SCR = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")


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


def price_1d(tj, tk, rho, npan=80, nnod=40, span=44.0):
    q = math.sqrt(1.0 - rho * rho)
    X, W = gl(-tj, -tj + span, npan, nnod)
    return float(np.sum(W * (X + tj) * phi(X) * q * psi((rho * X + tk) / q)))


def price_2d(tj, tk, rho, npan=110, nnod=40, span=26.0):
    """ЧЕСНА двовимірна квадратура по чверті {x>−t_j, y>−t_k}, де
    підінтегральна функція гладка (злами РІВНО на межі)."""
    q2 = 1.0 - rho * rho
    X, WX = gl(-tj, -tj + span, npan, nnod)
    Y, WY = gl(-tk, -tk + span, npan, nnod)
    tot = 0.0
    for i in range(0, X.size, 2048):
        sl = slice(i, min(i + 2048, X.size))
        x = X[sl][:, None]
        wx = WX[sl][:, None]
        Q = (x * x - 2 * rho * x * Y[None, :] + Y[None, :] ** 2) / q2
        pg = np.exp(-0.5 * Q) / (2 * math.pi * math.sqrt(q2))
        tot += float(np.sum(wx * WY[None, :] * (x + tj) * (Y[None, :] + tk)
                            * pg))
    return tot


print("pairseal97\n" + "=" * 88)
print("ЗВІРКА 1 🔒 1-D зведення гаусового Прайса проти ЧЕСНОЇ 2-D квадратури")
print(f"{'t_j':>6s}{'t_k':>6s}{'ρ':>8s}{'1-D зведення':>18s}"
      f"{'2-D квадратура':>18s}{'відн.':>12s}")
worst = 0.0
for tj, tk, rho in ((3.5, 3.5, 0.974), (3.5, 2.0, 0.90), (4.2, 4.2, 0.99),
                    (1.0, -0.5, 0.30), (0.0, 0.0, 0.0), (2.0, 3.0, -0.60)):
    a, b = price_1d(tj, tk, rho), price_2d(tj, tk, rho)
    r = abs(a - b) / abs(b)
    worst = max(worst, r)
    print(f"{tj:>6.2f}{tk:>6.2f}{rho:>8.3f}{a:>18.12f}{b:>18.12f}{r:>12.2e}")
print(f"  ⟹ найгірше {worst:.2e}   "
      f"{'✅ зведення ТОЧНЕ, 2-D квадратура не потрібна' if worst < 1e-10 else '❌'}")
print("  📌 це НЕ спростовує D: у ПОВНІСТЮ замкненій формі `Φ₂` справді")
print("     лишається (коеф. `t_j t_k + ρ`). Ми доводили РІЗНІ твердження.")

# ═══════════════════ ЗВІРКА 2: ЧОТИРИ МЕТРИКИ ПОРУЧ ═══════════════════
ZQ, WQ = gl(0.0, 44.0, 60, 40)


def price_vec(tj, tk, rho, chunk=512):
    out = np.empty(tj.shape[0])
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    for o in range(0, tj.shape[0], chunk):
        sl = slice(o, min(o + chunk, tj.shape[0]))
        x = ZQ[None, :] - tj[sl, None]
        u = (rho[sl, None] * x + tk[sl, None]) / q[sl, None]
        out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                   * psi(u)).sum(1)
    return out


def r2(y, x):
    """частка дисперсії y, пояснена ЛІНІЙНОЮ регресією на x (з вільним нахилом)."""
    x = x - x.mean()
    y = y - y.mean()
    den = float((x * x).sum() * (y * y).sum())
    return 0.0 if den <= 0 else float((x * y).sum() ** 2 / den)


d = np.load(f"{SCR}/mlp_00000.npz")
print("\n" + "=" * 88)
print("ЗВІРКА 2 🔒 ЧОТИРИ МЕТРИКИ НА ОДНОМУ ЗРІЗІ (mlp_00000, істина 1e8)")
print(f"{'шар':>4s}{'M1 знята RMS':>15s}{'M2 R² на СИРИЙ λ₂₁':>21s}"
      f"{'M3 R² на член λ₂₁·B':>22s}{'M4 R² на член γ':>18s}")
for L in (1, 8, 16, 24, 31):
    m = d["pre_mean"][L].astype(np.float64)
    m2 = d["pre_m2"][L].astype(np.float64)
    M11 = d["pre_M11"][L].astype(np.float64)
    M21 = d["pre_M21"][L].astype(np.float64)
    m3 = d["pre_m3"][L].astype(np.float64)
    s = np.sqrt(np.maximum(m2 - m * m, 1e-300))
    t = m / s
    vm, vM11 = d["mean"][L].astype(np.float64), d["M11"][L].astype(np.float64)
    j = np.repeat(np.arange(256), 256)
    k = np.tile(np.arange(256), 256)
    ok = (j != k) & (s[j] > 1e-6) & (s[k] > 1e-6)
    j, k = j[ok], k[ok]
    tj, tk, sj, sk = t[j], t[k], s[j], s[k]
    rho = np.clip((M11[j, k] - m[j] * m[k]) / (sj * sk), -0.999, 0.999)
    ljjk = (M21[j, k] - m[k] * m2[j] - 2 * m[j] * M11[j, k]
            + 2 * m[j] ** 2 * m[k]) / (sj * sj * sk)
    ljkk = (M21[k, j] - m[j] * m2[k] - 2 * m[k] * M11[j, k]
            + 2 * m[k] ** 2 * m[j]) / (sj * sk * sk)
    g = (m3 - 3 * m * m2 + 2 * m ** 3) / s ** 3
    gj, gk = g[j], g[k]
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    uk, uj = (tk - rho * tj) / q, (tj - rho * tk) / q
    Bj, Bk = phi(tj) * ndtr(uk), phi(tk) * ndtr(uj)
    Aj = -phi(tj) * (tj * q * psi(uk) + rho * ndtr(uk))
    Ak = -phi(tk) * (tk * q * psi(uj) + rho * ndtr(uj))

    pg = price_vec(tj, tk, rho)
    true = vM11[j, k] - vm[j] * vm[k]
    cg = (pg - psi(tj) * psi(tk)) * sj * sk
    err = true - cg                                    # похибка гауса
    term_l = (3.0 * ljjk * Bj + 3.0 * ljkk * Bk) / 6.0 * sj * sk
    term_g = (gj * Aj + gk * Ak) / 6.0 * sj * sk
    M1 = 1.0 - (np.sqrt(((err - term_l) ** 2).mean())
                / np.sqrt((err ** 2).mean()))
    print(f"{L:>4d}{100*M1:>14.1f}%{100*r2(err, ljjk):>20.1f}%"
          f"{100*r2(err, term_l):>21.1f}%{100*r2(err, term_g):>17.1f}%")

print("\n📌 M1 і M3 міряють ОДИН об'єкт по-різному; M2 міряє СИРИЙ кумулянт")
print("   без теоретичного ядра `φ(t_j)Φ(u_k)`, яке гуляє на порядки по парах.")
