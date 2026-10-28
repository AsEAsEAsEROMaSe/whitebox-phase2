"""pair3alpha97 — ОБОВ'ЯЗКОВІ СТОВПЦІ ДО ЧИСЛА «84% ЗНЯТО» (pair3real97).
Мій власний закон С97: «ПІСЛЯ ВІЛЬНОГО МАСШТАБУ» — обов'язковий стовпець у
БУДЬ-ЯКІЙ таблиці поправок; понад половина похибки замикання це чиста довжина
вектора, і `_ALPHA` у сабміті бере її ДАРОМ. Поправка, що не міняє ФОРМУ,
дає рівно ×1.00 на грейдері попри будь-яке число на стенді.

Плюс два контролі, без яких вирок неповний:
  (1) ДІАГОНАЛЬ: чи та сама машинерія лагодить `E[v_j²]` (гейт узгодженості С95:
      поправка, що чіпає позадіагональ БЕЗ діагоналі, дала ×0.83)
  (2) ТІЛЬКИ γ (маргінальний скіс) проти ТІЛЬКИ λ₂₁ (крос) — щоб знати,
      яка саме частина працює, а не приписувати все крос-кумулянту

ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ)
 (а) вільний α з'їсть частину виграшу, але НЕ весь (тут поправка змінює
     ФОРМУ по парах, а не масштаб)                                шанс 65%
 (б) внесок ЛИШЕ λ₂₁ (без γ) буде > половини                      шанс 55%
 (в) діагональ теж покращиться (1-D форма вже запечатана)          шанс 75%
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
    return np.exp(-0.5 * x * x) / SQ2PI


def psi(x):
    return phi(x) + x * ndtr(x)


def gl_nodes(a, b, npan, nnod):
    xs, ws = np.polynomial.legendre.leggauss(nnod)
    ed = np.linspace(a, b, npan + 1)
    lo, hi = ed[:-1, None], ed[1:, None]
    return ((0.5 * (hi - lo) * xs[None, :] + 0.5 * (hi + lo)).ravel(),
            (0.5 * (hi - lo) * ws[None, :]).ravel())


ZQ, WQ = gl_nodes(0.0, 44.0, 60, 40)


def price_gauss_vec(tj, tk, rho, chunk=512):
    out = np.empty(tj.shape[0], dtype=np.float64)
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    for o in range(0, tj.shape[0], chunk):
        sl = slice(o, min(o + chunk, tj.shape[0]))
        x = ZQ[None, :] - tj[sl, None]
        u = (rho[sl, None] * x + tk[sl, None]) / q[sl, None]
        out[sl] = (WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                   * psi(u)).sum(1)
    return out


def corr_k3(tj, tk, rho, gj, ljjk, ljkk, gk):
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    uk, uj = (tk - rho * tj) / q, (tj - rho * tk) / q
    Bj, Bk = phi(tj) * ndtr(uk), phi(tk) * ndtr(uj)
    Aj = -phi(tj) * (tj * q * psi(uk) + rho * ndtr(uk))
    Ak = -phi(tk) * (tk * q * psi(uj) + rho * ndtr(uj))
    return (gj * Aj + 3.0 * ljjk * Bj + 3.0 * ljkk * Bk + gk * Ak) / 6.0


def err_free_alpha(pred, true):
    """RMS похибки ПІСЛЯ підбору одного вільного скаляра (як `_ALPHA`)."""
    a = float((pred * true).sum() / max((pred * pred).sum(), 1e-300))
    return (float(np.sqrt(((a * pred - true) ** 2).mean())), a)


d = np.load(f"{SCR}/mlp_00000.npz")
print(f"pair3alpha97 · {str(d['name'])} · істина 1e8 · УСІ 65280 пар на шар")
print("=" * 104)
print(f"{'шар':>4s}{'ГАУС':>9s}{'+κ₃':>9s}{'знято':>8s}  ‖  "
      f"{'ГАУС+α':>9s}{'+κ₃+α':>9s}{'знято ПІСЛЯ α':>15s}  ‖  "
      f"{'лише γ':>9s}{'лише λ₂₁':>10s}{'α(κ₃)':>8s}")

for L in (8, 16, 24, 31):
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
    msk = (j != k) & (s[j] > 1e-6) & (s[k] > 1e-6)
    j, k = j[msk], k[msk]
    tj, tk, sj, sk = t[j], t[k], s[j], s[k]
    rho = np.clip((M11[j, k] - m[j] * m[k]) / (sj * sk), -0.999, 0.999)
    k3jjk = (M21[j, k] - m[k] * m2[j] - 2 * m[j] * M11[j, k]
             + 2 * m[j] ** 2 * m[k])
    k3jkk = (M21[k, j] - m[j] * m2[k] - 2 * m[k] * M11[j, k]
             + 2 * m[k] ** 2 * m[j])
    k3d = m3 - 3 * m * m2 + 2 * m ** 3
    ljjk, ljkk = k3jjk / (sj * sj * sk), k3jkk / (sj * sk * sk)
    g = k3d / s ** 3
    gj, gk = g[j], g[k]
    Z = np.zeros_like(gj)

    pg = price_gauss_vec(tj, tk, rho)
    mg_j, mg_k = psi(tj), psi(tk)
    mk_j = mg_j - gj / 6.0 * tj * phi(tj)
    mk_k = mg_k - gk / 6.0 * tk * phi(tk)

    true = vM11[j, k] - vm[j] * vm[k]
    cg = (pg - mg_j * mg_k) * sj * sk
    c3 = (pg + corr_k3(tj, tk, rho, gj, ljjk, ljkk, gk)
          - mk_j * mk_k) * sj * sk
    cG = (pg + corr_k3(tj, tk, rho, gj, Z, Z, gk) - mk_j * mk_k) * sj * sk
    cL = (pg + corr_k3(tj, tk, rho, Z, ljjk, ljkk, Z)
          - mg_j * mg_k) * sj * sk

    sc = np.sqrt((true ** 2).mean())
    e_g = np.sqrt(((cg - true) ** 2).mean()) / sc
    e_3 = np.sqrt(((c3 - true) ** 2).mean()) / sc
    e_G = np.sqrt(((cG - true) ** 2).mean()) / sc
    e_L = np.sqrt(((cL - true) ** 2).mean()) / sc
    ag, _ = err_free_alpha(cg, true)
    a3, alp = err_free_alpha(c3, true)
    ag, a3 = ag / sc, a3 / sc
    print(f"{L:>4d}{100*e_g:>8.3f}%{100*e_3:>8.3f}%{100*(1-e_3/e_g):>7.1f}%  ‖  "
          f"{100*ag:>8.3f}%{100*a3:>8.3f}%{100*(1-a3/ag):>14.1f}%  ‖  "
          f"{100*e_G:>8.3f}%{100*e_L:>9.3f}%{alp:>8.4f}")

print("\n" + "-" * 104)
print("ДІАГОНАЛЬ (гейт узгодженості С95: поправка мусить чіпати ОБИДВІ частини)")
print(f"{'шар':>4s}{'Var ГАУС':>11s}{'Var +κ₃':>11s}{'знято':>8s}"
      f"{'ГАУС+α':>10s}{'+κ₃+α':>10s}{'знято ПІСЛЯ α':>15s}")
for L in (8, 16, 24, 31):
    m = d["pre_mean"][L].astype(np.float64)
    m2 = d["pre_m2"][L].astype(np.float64)
    m3 = d["pre_m3"][L].astype(np.float64)
    s = np.sqrt(np.maximum(m2 - m * m, 1e-300))
    ok = s > 1e-6
    t = (m / s)[ok]
    g = ((m3 - 3 * m * m2 + 2 * m ** 3) / s ** 3)[ok]
    ss = s[ok]
    vm = d["mean"][L].astype(np.float64)[ok]
    vm2 = d["m2"][L].astype(np.float64)[ok]
    true = vm2 - vm * vm
    # 1-D запечатані форми: E[ReLU]/σ = ψ − (γ/6)tφ ;
    #                       E[ReLU²]/σ² = (1+t²)Φ + tφ + γφ/3
    e2g = ((1 + t * t) * ndtr(t) + t * phi(t)) * ss ** 2
    mgv = psi(t) * ss
    m3v = (psi(t) - g / 6.0 * t * phi(t)) * ss
    e23 = (((1 + t * t) * ndtr(t) + t * phi(t)) + g * phi(t) / 3.0) * ss ** 2
    vg, v3 = e2g - mgv ** 2, e23 - m3v ** 2
    sc = np.sqrt((true ** 2).mean())
    eg = np.sqrt(((vg - true) ** 2).mean()) / sc
    e3 = np.sqrt(((v3 - true) ** 2).mean()) / sc
    ag = err_free_alpha(vg, true)[0] / sc
    a3 = err_free_alpha(v3, true)[0] / sc
    print(f"{L:>4d}{100*eg:>10.3f}%{100*e3:>10.3f}%{100*(1-e3/eg):>7.1f}%"
          f"{100*ag:>9.3f}%{100*a3:>9.3f}%{100*(1-a3/ag):>14.1f}%")
