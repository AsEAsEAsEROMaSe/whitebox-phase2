"""pairdiag97 — ЛОКАЛІЗАЦІЯ НЕСУМІСНОСТІ З D. ТРИ ЧИСЛА, ЯКИХ ВОНА ПРОСИЛА,
ПЛЮС ГІПОТЕЗА ПРО ПРИЧИНУ, ПЕРЕВІРЕНА ПРЯМО.

════════════ НЕСУМІСНІСТЬ (формулювання контролера) ════════════
мій ОДИН член λ₂₁·ядро ш31: R² 98.5% · її НАЙКРАЩА комбінація 4: R² 87.2%
Один член не може перевищити найкращу комбінацію, що його містить ⟹ ЦІЛІ РІЗНІ.

════════════ ГІПОТЕЗА ПРО ПРИЧИНУ (до прогону) ════════════
Її ціль: `D = ІСТИНА E[a_j a_k] − ГАУС` — СИРИЙ другий момент.
Моя ціль: `Cov = ІСТИНА(M11−m⊗m) − (Прайс − ψψ)·s s` — ЦЕНТРАЛЬНА коваріація.
🔑 Вони відрізняються НЕ нормуванням, а тим, що в сирому моменті лишається
похибка КАНАЛУ `m`: гаусове `m = sψ(t)` не дорівнює істинному `m`, і різниця
`m_іст⊗m_іст − ψψ·s s` входить у її `D` і НЕ входить у мою.
`λ₂₁` цю компоненту не пояснює за побудовою ⟹ її R² розбавлений.
📏 Перевірка: її RMS = 3.2% (≈ записані 4.08%), моя 0.963% — розрив ×3.3.
Якщо гіпотеза вірна, СИРИЙ варіант на МОЄМУ зрізі дасть ≈3%, а не 0.96%.

ПЕРЕДБАЧЕННЯ: сирий варіант дасть RMS 2.5…4% і R²(λ₂₁) впаде нижче 20%  75%
"""
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
SCR = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")


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


def r2_slope(y, x):
    """R² і НАХИЛ МНК (без вільного члена — так само, як діє поправка з коеф.1)."""
    den = float((x * x).sum())
    if den <= 0:
        return 0.0, 0.0
    b = float((x * y).sum() / den)
    ss = float((y * y).sum())
    return (1.0 - float(((y - b * x) ** 2).sum()) / ss if ss > 0 else 0.0), b


d = np.load(f"{SCR}/mlp_00000.npz")
print("pairdiag97 · mlp_00000 · істина 1e8 · УСІ 65280 пар (j≠k)\n" + "=" * 104)
print("ДВІ ЦІЛІ ПОРУЧ. «СИРА» = E[a_j a_k] (ціль D) · «ЦЕНТР» = Cov (моя ціль)")
print(f"{'шар':>4s}{'RMS СИРА':>11s}{'RMS ЦЕНТР':>11s}{'відн.':>8s}  ‖ "
      f"{'R²(λ) СИРА':>12s}{'нахил':>8s}{'знято@1':>9s}  ‖ "
      f"{'R²(λ) ЦЕНТР':>13s}{'нахил':>8s}{'знято@1':>9s}")

for L in (1, 8, 16, 24, 31):
    m = d["pre_mean"][L].astype(np.float64)
    m2 = d["pre_m2"][L].astype(np.float64)
    M11 = d["pre_M11"][L].astype(np.float64)
    M21 = d["pre_M21"][L].astype(np.float64)
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
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    uk, uj = (tk - rho * tj) / q, (tj - rho * tk) / q
    Bj, Bk = phi(tj) * ndtr(uk), phi(tk) * ndtr(uj)

    pg = price_vec(tj, tk, rho)
    term = (3.0 * ljjk * Bj + 3.0 * ljkk * Bk) / 6.0 * sj * sk

    raw_true = vM11[j, k]
    raw_pred = pg * sj * sk
    cen_true = vM11[j, k] - vm[j] * vm[k]
    cen_pred = (pg - psi(tj) * psi(tk)) * sj * sk
    err_raw = raw_true - raw_pred
    err_cen = cen_true - cen_pred
    scale = np.sqrt((cen_true ** 2).mean())     # спільне нормування: RMS Cov

    rr, br = r2_slope(err_raw, term)
    rc, bc = r2_slope(err_cen, term)
    kr = 1 - np.sqrt(((err_raw - term) ** 2).mean() / (err_raw ** 2).mean())
    kc = 1 - np.sqrt(((err_cen - term) ** 2).mean() / (err_cen ** 2).mean())
    print(f"{L:>4d}{100*np.sqrt((err_raw**2).mean())/scale:>10.3f}%"
          f"{100*np.sqrt((err_cen**2).mean())/scale:>10.3f}%"
          f"{np.sqrt((err_raw**2).mean())/np.sqrt((err_cen**2).mean()):>8.2f}  ‖ "
          f"{100*rr:>11.1f}%{br:>8.3f}{100*kr:>8.1f}%  ‖ "
          f"{100*rc:>12.1f}%{bc:>8.3f}{100*kc:>8.1f}%")

print("\n" + "-" * 104)
print("ВІДПОВІДІ НА ТРИ ПИТАННЯ D:")
print("  (а) RMS моєї нев'язки, нормованої на RMS Cov — стовпці вище")
print("  (б) пари j=k ВИКЛЮЧЕНІ (маска j != k), як і мертві (s < 1e-6)")
print("  (в) 'знято@1' — коефіцієнт ТЕОРЕТИЧНИЙ (=1, поправка віднімається")
print("      як є); 'R²' і 'нахил' — МНК із вільним нахилом.")
print("      ⟹ пломба 'знято@1 ≈ 1−√(1−R²)' НЕ тавтологія РІВНО ТОДІ,")
print("      коли нахил ≈ 1. Стовпець 'нахил' і є цією перевіркою.")
