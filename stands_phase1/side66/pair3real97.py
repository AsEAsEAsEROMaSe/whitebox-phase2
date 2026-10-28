"""pair3real97 — СКІЛЬКИ ПОХИБКИ ПОЗАДІАГОНАЛІ СПРАВДІ ПОЯСНЮЄ `λ₂₁`.
ПРЯМО НА ІСТИНІ 1e8, ПО РОЗПОДІЛУ `t`, А НЕ В МЕДІАНІ.

════════════════ ЧОМУ ЦЕ ТРЕБА ЗРОБИТИ ДО РЕАЛІЗАЦІЇ ════════════════
`[ЗМІРЯНО pair3form97]` у медіанній точці (t=3.5, ρ=0.974, λ₂₁=0.2835)
поправка κ₃ до `E[ReLU_j·ReLU_k]` = 1.62e-04 на значенні 13.22 = **0.001%**,
тоді як зміряна похибка позадіагоналі замикання = **4.08%**. Розрив ×4000.
🚨 АЛЕ РІВНО ЦЕЙ РАХУНОК Я ВЖЕ РОБИЛА ХИБНО: «придушено на φ(t)», пораховане
в МЕДІАНІ, занижує ефект на 1–2 порядки, бо похибка живе на нейронах із МАЛИМ
`|t|`, де `φ` велике (закон С97, спійманий двічі за день). ⟹ вирок виносити
ЛИШЕ по РОЗПОДІЛУ (t_j, t_k, ρ, λ) реальної мережі.
🚨 І ДРУГЕ: цільовий об'єкт — НЕ `E[v_jv_k]`, а `Cov(v)_jk = E[v_jv_k]−m_jm_k`,
різниця майже рівних великих чисел (13.22 проти 12.25 при t=3.5) ⟹ відносна
вага поправки там ІНША. Поправку до `m` треба вносити УЗГОДЖЕНО (гейт С95).

════════════════ ЩО МІРЯЄМО (одноступенево, істина 1e8) ════════════════
Вхід — ПЕРЕДактивації шару L із датасету: (t_j, t_k, ρ_jk, λ_jjk, λ_jkk).
Вихід — ПОСТактивації того ж шару: `Cov(v)_jk` з `M11`/`mean`. Три прогнози:
    ГАУС   : price_gauss(t_j,t_k,ρ) − ψ(t_j)ψ(t_k)
    +κ₃    : те саме + поправка пари − УЗГОДЖЕНА поправка до m_j·m_k
    істина : з датасету
🔒 ПЛОМБИ: (1) λ=0 ⟹ рівно ГАУС · (2) контроль «ЧУЖА λ» (перемішані пари)
мусить погіршити · (3) енергетично зважена метрика, не відносна понейронна
(інакше майже мертві нейрони дають 36548%, як у P0).

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) поправка зніме <10% похибки позадіагоналі                   шанс 70%
 (б) вона зніме >50%                                             шанс 12%
 (в) розподіл `t` підніме ефект проти медіанної оцінки у 3…30×
     (як у k3diag97, де промах медіани був ×26)                  шанс 60%
 (г) контроль «чужа λ» дасть ~×1.00                              шанс 80%
"""
import math
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
SCR = ("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
       "97ba0e9f-c49d-48e8-8d0a-5a2ae0c00729/scratchpad")
LAYERS = (8, 16, 24, 31)
NPAIR = 24000
NQ = 2400                        # вузлів квадратури на пару


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


ZQ, WQ = gl_nodes(0.0, 44.0, NQ // 40, 40)


def price_gauss_vec(tj, tk, rho, chunk=512):
    """E[(x+t_j)₊(y+t_k)₊] векторно. Заміна x = z − t_j ⟹ вузли СПІЛЬНІ:
        ∫₀^∞ z·φ(z−t_j)·q·ψ((ρ(z−t_j)+t_k)/q) dz
    """
    out = np.empty(tj.shape[0], dtype=np.float64)
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    for o in range(0, tj.shape[0], chunk):
        sl = slice(o, min(o + chunk, tj.shape[0]))
        x = ZQ[None, :] - tj[sl, None]
        u = (rho[sl, None] * x + tk[sl, None]) / q[sl, None]
        out[sl] = ((WQ[None, :] * ZQ[None, :] * phi(x) * q[sl, None]
                    * psi(u)).sum(1))
    return out


def corr_k3(tj, tk, rho, gj, ljjk, ljkk, gk):
    """поправка 1-го порядку по κ₃ до E[ReLU_j·ReLU_k] (замкнена форма)."""
    q = np.sqrt(np.maximum(1.0 - rho * rho, 1e-12))
    uk = (tk - rho * tj) / q
    uj = (tj - rho * tk) / q
    Bj = phi(tj) * ndtr(uk)
    Bk = phi(tk) * ndtr(uj)
    Aj = -phi(tj) * (tj * q * psi(uk) + rho * ndtr(uk))
    Ak = -phi(tk) * (tk * q * psi(uj) + rho * ndtr(uj))
    return (gj * Aj + 3.0 * ljjk * Bj + 3.0 * ljkk * Bk + gk * Ak) / 6.0


d = np.load(f"{SCR}/mlp_00000.npz")
rng = np.random.default_rng(970309)
print(f"pair3real97 · {str(d['name'])} · істина N={int(d['n_samples']):,}"
      f" · {NPAIR} пар на шар\n" + "=" * 100)
print(f"{'шар':>4s}{'|t| мед':>9s}{'ρ мед':>8s}{'|λ₂₁| мед':>11s}"
      f"{'RMS ГАУС':>11s}{'RMS +κ₃':>10s}{'знято':>9s}"
      f"{'🔒чужа λ':>10s}{'попр/похиб':>12s}")

for L in LAYERS:
    m = d["pre_mean"][L].astype(np.float64)
    m2 = d["pre_m2"][L].astype(np.float64)
    M11 = d["pre_M11"][L].astype(np.float64)
    M21 = d["pre_M21"][L].astype(np.float64)
    s = np.sqrt(np.maximum(m2 - m * m, 1e-300))
    t = m / s
    # істина ПОСТактивацій того ж шару
    vm = d["mean"][L].astype(np.float64)
    vM11 = d["M11"][L].astype(np.float64)

    jj, kk = [], []
    while len(jj) < NPAIR:
        a = rng.integers(0, 256, NPAIR)
        b = rng.integers(0, 256, NPAIR)
        ok = (a != b) & (s[a] > 1e-6) & (s[b] > 1e-6)
        jj.extend(a[ok].tolist())
        kk.extend(b[ok].tolist())
    j = np.array(jj[:NPAIR])
    k = np.array(kk[:NPAIR])

    tj, tk, sj, sk = t[j], t[k], s[j], s[k]
    cov = M11[j, k] - m[j] * m[k]
    rho = np.clip(cov / (sj * sk), -0.999, 0.999)
    k3jjk = M21[j, k] - m[k] * m2[j] - 2 * m[j] * M11[j, k] + 2 * m[j] ** 2 * m[k]
    k3jkk = M21[k, j] - m[j] * m2[k] - 2 * m[k] * M11[j, k] + 2 * m[k] ** 2 * m[j]
    k3jjj = d["pre_m3"][L].astype(np.float64) - 3 * m * m2 + 2 * m ** 3
    ljjk = k3jjk / (sj * sj * sk)
    ljkk = k3jkk / (sj * sk * sk)
    gj, gk = (k3jjj / s ** 3)[j], (k3jjj / s ** 3)[k]

    # ── прогнози (у стандартизованих одиницях, тоді × s_j s_k) ──
    pg = price_gauss_vec(tj, tk, rho)
    cr = corr_k3(tj, tk, rho, gj, ljjk, ljkk, gk)
    # УЗГОДЖЕНА поправка до m: 1-D Еджворт ψ(t) − (γ/6)tφ(t)
    mg_j, mg_k = psi(tj), psi(tk)
    mk_j = mg_j - gj / 6.0 * tj * phi(tj)
    mk_k = mg_k - gk / 6.0 * tk * phi(tk)

    cov_true = (vM11[j, k] - vm[j] * vm[k])
    cov_g = (pg - mg_j * mg_k) * sj * sk
    cov_3 = (pg + cr - mk_j * mk_k) * sj * sk

    # ── контроль «чужа λ»: перемішані значення поправки ──
    per = rng.permutation(NPAIR)
    cov_x = (pg + cr[per] - mk_j[per] * mk_k[per] * 0 - mg_j * mg_k) * sj * sk

    sc = np.sqrt((cov_true ** 2).mean())          # енергетичне нормування
    eg = np.sqrt(((cov_g - cov_true) ** 2).mean()) / sc
    e3 = np.sqrt(((cov_3 - cov_true) ** 2).mean()) / sc
    ex = np.sqrt(((cov_x - cov_true) ** 2).mean()) / sc
    ratio = np.sqrt((((cov_3 - cov_g)) ** 2).mean()) / np.sqrt(
        ((cov_g - cov_true) ** 2).mean())
    print(f"{L:>4d}{np.median(np.abs(tj)):>9.3f}{np.median(rho):>8.4f}"
          f"{np.median(np.abs(ljjk)):>11.4f}{100*eg:>10.3f}%{100*e3:>9.3f}%"
          f"{100*(1-e3/eg):>8.1f}%{eg/ex:>10.4f}{ratio:>12.4f}")

print("\n" + "-" * 100)
print("🔒 ПЛОМБА λ=0 (поправка вимкнена ⟹ мусить дати РІВНО стовпець ГАУС):")
L = 31
m = d["pre_mean"][L].astype(np.float64)
m2 = d["pre_m2"][L].astype(np.float64)
s = np.sqrt(np.maximum(m2 - m * m, 1e-300))
print(f"  corr_k3(…, 0,0,0,0) = {np.abs(corr_k3(np.array([3.5]), np.array([3.4]), np.array([0.9]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))).max():.3e}")

print("\n" + "-" * 100)
print("РОЗКЛАД ЗА |t| — ЧИ ЖИВЕ ПОХИБКА ТАМ, ДЕ `φ` ВЕЛИКЕ (закон С97)")
L = 31
m = d["pre_mean"][L].astype(np.float64)
m2 = d["pre_m2"][L].astype(np.float64)
M11 = d["pre_M11"][L].astype(np.float64)
M21 = d["pre_M21"][L].astype(np.float64)
s = np.sqrt(np.maximum(m2 - m * m, 1e-300))
t = m / s
vm, vM11 = d["mean"][L].astype(np.float64), d["M11"][L].astype(np.float64)
j = np.repeat(np.arange(256), 256)
k = np.tile(np.arange(256), 256)
msk = (j != k) & (s[j] > 1e-6) & (s[k] > 1e-6)
j, k = j[msk], k[msk]
tj, tk, sj, sk = t[j], t[k], s[j], s[k]
cov = M11[j, k] - m[j] * m[k]
rho = np.clip(cov / (sj * sk), -0.999, 0.999)
k3jjk = M21[j, k] - m[k] * m2[j] - 2 * m[j] * M11[j, k] + 2 * m[j] ** 2 * m[k]
k3jkk = M21[k, j] - m[j] * m2[k] - 2 * m[k] * M11[j, k] + 2 * m[k] ** 2 * m[j]
k3d = d["pre_m3"][L].astype(np.float64) - 3 * m * m2 + 2 * m ** 3
ljjk, ljkk = k3jjk / (sj * sj * sk), k3jkk / (sj * sk * sk)
gj, gk = (k3d / s ** 3)[j], (k3d / s ** 3)[k]
pg = price_gauss_vec(tj, tk, rho)
cr = corr_k3(tj, tk, rho, gj, ljjk, ljkk, gk)
mg_j, mg_k = psi(tj), psi(tk)
mk_j = mg_j - gj / 6.0 * tj * phi(tj)
mk_k = mg_k - gk / 6.0 * tk * phi(tk)
cov_true = vM11[j, k] - vm[j] * vm[k]
cov_g = (pg - mg_j * mg_k) * sj * sk
cov_3 = (pg + cr - mk_j * mk_k) * sj * sk
mn = np.minimum(np.abs(tj), np.abs(tk))
qs = np.quantile(mn, [0, .2, .4, .6, .8, 1.0])
print(f"{'смуга min|t|':>18s}{'частка пар':>12s}{'частка ПОХИБКИ²':>18s}"
      f"{'RMS ГАУС':>11s}{'RMS +κ₃':>10s}{'знято':>9s}")
tot = ((cov_g - cov_true) ** 2).sum()
sc = np.sqrt((cov_true ** 2).mean())
for i in range(5):
    b = (mn >= qs[i]) & (mn <= qs[i + 1])
    eg = np.sqrt(((cov_g[b] - cov_true[b]) ** 2).mean()) / sc
    e3 = np.sqrt(((cov_3[b] - cov_true[b]) ** 2).mean()) / sc
    print(f"{qs[i]:>8.2f}…{qs[i+1]:<9.2f}{100*b.mean():>11.1f}%"
          f"{100*((cov_g[b]-cov_true[b])**2).sum()/tot:>17.1f}%"
          f"{100*eg:>10.3f}%{100*e3:>9.3f}%{100*(1-e3/eg):>8.1f}%")
print(f"\n  УСІ пари шару 31: ГАУС {100*np.sqrt(((cov_g-cov_true)**2).mean())/sc:.3f}%"
      f" · +κ₃ {100*np.sqrt(((cov_3-cov_true)**2).mean())/sc:.3f}%")
