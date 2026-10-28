"""wfeat99 — ЧИ ІСНУЄ ДЕТЕРМІНОВАНА ФУНКЦІЯ ВІД `W`, ЯКА ПЕРЕДБАЧАЄ ПОХИБКУ
ГАУСОВОГО ЗАМИКАННЯ? ПЕРША ЧЕСНА ПЕРЕВІРКА КЛАСУ (режим n=32, шум 0.9% сигналу).

════════════ ЧОМУ ЦЕ НЕ ПОВТОРЕННЯ ДЕСЯТКІВ СПРОБ ════════════
При n=256 чверть ЦІЛІ була шумом (a_true банку = 24% робочого MSE) ⟹ R²
обвалювався навіть при ідеальному предикторі, і «сигналу нема» було не
відрізнити від «сигнал захований». С98: при n=32 сигнал/шум = 116.
🔑 Шум цілі входить у ЗНАМЕННИК R². Тут він 0.9% від сигналу ⟹ стеля R² = 0.9998.

════════════ ГОЛОВНИЙ ПРЕДИКТОР ВИВЕДЕНО, А НЕ ВГАДАНО ════════════
Структура похибки замкнена ТОЧНО (С70, `recon.py`, пломба лінійності 0.00000):
    b_ℓ = Φ(t_ℓ)⊙(b_{ℓ−1} @ W_ℓ) + φ(t_ℓ)·δs_ℓ + e_лок,ℓ
⟹ перенос є ЛІНІЙНИМ оператором `T_ℓ = W_ℓ · diag(Φ(t_ℓ))`, і
    b_32 = Σ_ℓ src_ℓ @ M_{ℓ→32},   M_{ℓ→32} = T_{ℓ+1}···T_{32}
Локальне джерело універсальне (0.139% / 1.80% на КОЖНОМУ шарі, С96) ⟹ уся
міжмережева дисперсія `e` мусить сидіти в ПІДСИЛЕННІ. При некогерентних
джерелах:
    A = Σ_ℓ rms(m_ℓ)²·‖M_{ℓ→32}‖_F²/n  /  rms(m_32)²      ⟹  F1 = ½·log A
🔑 `T_ℓ` рахується З ЗАМИКАННЯ, тобто ДЕТЕРМІНОВАНО з `W`, без жодної вибірки.
Це не «ще одна ознака з ваг» — це та сама величина, що стоїть у рівнянні.

════════════ ПЛОМБИ (усі обов'язкові, вирок без них не рахується) ════════════
 S1 🔒 ЧУЖА `W`: перемішати відповідність (ознаки ↔ цілі) 400 разів, LOO-R²
      мусить лягти навколо ≤0. Дає ЕМПІРИЧНИЙ нуль, а не теоретичний.
 S2 🔒 КРОС-ВАЛІДАЦІЯ ПО МЕРЕЖАХ (LOO), не по семплах. Вибір ознаки —
      ВСЕРЕДИНІ тренувальної згортки, інакше вимір нечесний.
 S3 🔒 ШУМ ЦІЛІ — з розщеплення вибірки МК навпіл, а не оцінений.
      Мусить бути ≪ міжмережевого розкиду, інакше стеля R² невідома.
 S4 🔒 БЕЗРОЗМІРНІСТЬ: усі ознаки — відношення/частки/логарифми відношень,
      осмислені при БУДЬ-ЯКОМУ n. Без цього перенесення на n=256 неможливе.
 S5 🔒 МЕРТВІ МЕРЕЖІ відкидати (rms(m_32) ≈ 0) — об'єкта там немає.

════════════ ПЕРЕДБАЧЕННЯ (ЗАПИСАНО ДО ПРОГОНУ) ════════════
 (а) S1 дасть медіану LOO-R² в [−0.15, +0.05]                        85%
 (б) F1 (виведений перенос) буде НАЙСИЛЬНІШОЮ одиночною ознакою      55%
 (в) |corr(F1, log e)| > 0.5                                         45%
 (г) LOO-R² повного набору > 0.3 (вирок «гілка відкрита»)            20%
 (д) sd(log e) між мережами буде 0.35…0.75                           70%
 (е) 🚨 ЯКЩО щось спрацює — це буде НЕ спектр `WᵀW` (він майже
     детермінований за МП і має розкид ~1/n), а ВОРІТНІ величини
     (частка живих, ‖M‖) — бо саме вони несуть реалізацію             —
"""
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32
NWID = 32
NNET = 48
NSAMP = 1 << 20


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


ZQ, WQ = gl(0.0, 40.0, 24, 20)          # 480 вузлів — з запасом при n=32


def make_net(n, seed):
    """He-init, без біасів, глибина 32 — конструкція банку."""
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


def closure_full(W, n):
    """Гаусове замикання. Повертає `m` виходу + ДЕТЕРМІНОВАНІ побічні величини:
    ворота Φ(t) пошарово, rms(m) пошарово, `t` пошарово."""
    m = np.zeros(n)
    C = np.eye(n)
    gates, rms, tt, prC = [], [], [], []
    for L in range(DEPTH):
        mp = W[L].T @ m
        Cp = W[L].T @ C @ W[L]
        s2 = np.maximum(np.diag(Cp), 1e-300)
        s = np.sqrt(s2)
        t = mp / s
        rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
        M2 = price_pairs(t, rho) * (s[:, None] * s[None, :])
        np.fill_diagonal(M2, s2 * ((1 + t * t) * ndtr(t) + t * phi(t)))
        ma = s * psi(t)
        Cn = M2 - np.outer(ma, ma)
        C = 0.5 * (Cn + Cn.T)
        m = ma
        gates.append(ndtr(t))
        tt.append(t.copy())
        rms.append(math.sqrt(float((m * m).mean())))
        ev = np.maximum(np.diag(C), 0.0)
        prC.append(float(ev.sum() ** 2 / max((C * C).sum(), 1e-300)))
    return m, gates, np.array(rms), tt, np.array(prC)


def truth_mc(W, n, N, seed):
    """МК-істина + шум зі СПРАВЖНЬОГО розщеплення вибірки навпіл (пломба S3)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    accA = np.zeros(n)
    accB = np.zeros(n)
    nA = nB = 0
    CH = 1 << 15
    done = 0
    k = 0
    while done < N:
        b = min(CH, N - done)
        h = rng.standard_normal((b, n))
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
    dif = accA / nA - accB / nB
    return mean, 0.5 * math.sqrt(float((dif * dif).mean()))


# ═════════════════ ОЗНАКИ — УСІ БЕЗРОЗМІРНІ (пломба S4) ═════════════════
FNAMES = [
    "F1_transport",     # ½log Σ rms_ℓ²‖M_ℓ‖²_F/n / rms_32²   ← ВИВЕДЕНИЙ
    "F2_transp_max",    # ½log max того самого доданка
    "F3_transp_spec",   # те саме зі спектральною нормою
    "F4_gate_mean",     # ⟨Φ(t)⟩ по всіх шарах
    "F5_alive_frac",    # частка нейронів із Φ(t)>0.01, глибокі шари
    "F6_neff_gate",     # ⟨(ΣΦ)²/ΣΦ²⟩/n — ефективна ширина воріт
    "F7_absT_med",      # медіана |t| по всіх шарах
    "F8_smallT_frac",   # частка |t|<1 (там живе похибка)
    "F9_prC_last",      # PR(C) останнього шару / n
    "F10_spec_kappa",   # ⟨log λmax/λmean(WᵀW)⟩
    "F11_spec_pr",      # ⟨PR(спектр WᵀW)⟩/n
    "F12_logdet",       # ⟨log λ⟩ = 2·⟨log|det W|⟩/n  — флуктуація реалізації
    "F13_align",        # ⟨cos(v1(W_ℓ), v1(W_{ℓ+1}))⟩ — вирівнювання сусідів
    "F14_gate_drift",   # sd по шарах від ⟨Φ⟩ — нерівномірність воріт з глибиною
]


def features(W, n, gates, rms, tt, prC):
    f = {}
    # ── перенос: M_{ℓ→32} = T_{ℓ+1}···T_{32},  T_ℓ = W_ℓ diag(Φ(t_ℓ))
    P = np.eye(n)
    terms, sterms = [], []
    for L in range(DEPTH - 1, -1, -1):
        fro = float((P * P).sum())
        spc = float(np.linalg.norm(P, 2))
        terms.append(rms[L] ** 2 * fro / n)
        sterms.append(rms[L] ** 2 * spc ** 2 / n)
        P = (W[L] * gates[L][None, :]) @ P
    den = max(rms[DEPTH - 1] ** 2, 1e-300)
    terms = np.array(terms) / den
    sterms = np.array(sterms) / den
    f["F1_transport"] = 0.5 * math.log(max(terms.sum(), 1e-300))
    f["F2_transp_max"] = 0.5 * math.log(max(terms.max(), 1e-300))
    f["F3_transp_spec"] = 0.5 * math.log(max(sterms.sum(), 1e-300))
    # ── ворітні
    G = np.array(gates)                              # (DEPTH, n)
    f["F4_gate_mean"] = float(G.mean())
    deep = G[DEPTH // 2:]
    f["F5_alive_frac"] = float((deep > 0.01).mean())
    f["F6_neff_gate"] = float(np.mean(
        G.sum(1) ** 2 / np.maximum((G * G).sum(1), 1e-300)) / n)
    T = np.array(tt)
    f["F7_absT_med"] = float(np.median(np.abs(T)))
    f["F8_smallT_frac"] = float((np.abs(T) < 1.0).mean())
    f["F9_prC_last"] = float(prC[-1] / n)
    f["F14_gate_drift"] = float(np.std(G.mean(1)))
    # ── спектральні
    ka, pr, ld, al = [], [], [], []
    v_prev = None
    for L in range(DEPTH):
        u, sv, vt = np.linalg.svd(W[L])
        lam = sv ** 2
        ka.append(math.log(lam.max() / lam.mean()))
        pr.append(lam.sum() ** 2 / (lam * lam).sum() / n)
        ld.append(float(np.log(lam).mean()))
        if v_prev is not None:
            al.append(abs(float(v_prev @ u[:, 0])))
        v_prev = vt[0]
    f["F10_spec_kappa"] = float(np.mean(ka))
    f["F11_spec_pr"] = float(np.mean(pr))
    f["F12_logdet"] = float(np.mean(ld))
    f["F13_align"] = float(np.mean(al))
    return np.array([f[k] for k in FNAMES])


# ═════════════════ ЧЕСНА ОЦІНКА: LOO З ВИБОРОМ ВСЕРЕДИНІ ═════════════════
def loo_ridge(X, y, lam):
    """LOO-R² гребеневої регресії; стандартизація — на ТРЕНУВАЛЬНІЙ згортці."""
    m = X.shape[0]
    pred = np.empty(m)
    for i in range(m):
        tr = np.ones(m, bool)
        tr[i] = False
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        Z = (Xt - mu) / sd
        ym = yt.mean()
        A = Z.T @ Z + lam * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (yt - ym))
        pred[i] = ym + ((X[i] - mu) / sd) @ w
    ss = float(((y - pred) ** 2).sum())
    return 1.0 - ss / float(((y - y.mean()) ** 2).sum()), pred


def loo_best_single(X, y):
    """LOO, де ВИБІР ознаки робиться всередині кожної згортки (пломба S2)."""
    m, p = X.shape
    pred = np.empty(m)
    picks = []
    for i in range(m):
        tr = np.ones(m, bool)
        tr[i] = False
        Xt, yt = X[tr], y[tr]
        best, bj = -np.inf, 0
        for j in range(p):
            c = np.corrcoef(Xt[:, j], yt)[0, 1]
            c = 0.0 if not np.isfinite(c) else c
            if c * c > best:
                best, bj = c * c, j
        sx, sy = Xt[:, bj], yt
        b = np.cov(sx, sy)[0, 1] / max(np.var(sx), 1e-300)
        pred[i] = sy.mean() + b * (X[i, bj] - sx.mean())
        picks.append(bj)
    ss = float(((y - pred) ** 2).sum())
    return 1.0 - ss / float(((y - y.mean()) ** 2).sum()), picks


# ═════════════════════════════ ПРОГІН ═════════════════════════════
print(f"wfeat99 · n={NWID} · глибина {DEPTH} · {NNET} мереж · МК N={NSAMP}",
      flush=True)
t0 = time.time()
Xs, ys, noises = [], [], []
dead = 0
for k in range(NNET):
    W = make_net(NWID, 990000 + 31 * k)
    cm, gates, rms, tt, prC = closure_full(W, NWID)
    tm, nz = truth_mc(W, NWID, NSAMP, 660000 + 13 * k)
    sc = math.sqrt(float((tm * tm).mean()))
    if sc < 1e-8:                                   # пломба S5
        dead += 1
        continue
    e = float(np.sqrt(((cm - tm) ** 2).mean()) / sc)
    Xs.append(features(W, NWID, gates, rms, tt, prC))
    ys.append(math.log(e))
    noises.append(nz / sc)
    if k % 8 == 0:
        print(f"  мережа {k:>3d}: e={100*e:.3f}%  шум={100*nz/sc:.4f}%"
              f"  ({time.time()-t0:.0f} с)", flush=True)

X = np.array(Xs)
y = np.array(ys)
NZ = np.array(noises)
M = X.shape[0]
print(f"\n  {M} живих мереж (мертвих {dead}) · {time.time()-t0:.0f} с")

print("\n" + "=" * 86)
print("ЦІЛЬ")
print("=" * 86)
print(f"  медіана e = {100*math.exp(np.median(y)):.4f}%"
      f"   (nlaw97 при n=32 дала 9.61% — пломба режиму)")
print(f"  sd(log e) = {y.std():.4f}   ⟹ розкид між мережами "
      f"×{math.exp(2*y.std()):.2f} (±1σ)")
print(f"  🔒 S3 шум МК = {100*np.median(NZ):.4f}% від сигналу "
      f"⟹ sd(log) від шуму = {np.median(NZ)/math.exp(np.median(y)):.5f}")
cap = 1.0 - (np.median(NZ) / math.exp(np.median(y))) ** 2 / y.var()
print(f"  ⟹ СТЕЛЯ R² через шум цілі = {cap:.5f}")

print("\n" + "=" * 86)
print("ОДИНИЧНІ ОЗНАКИ (corr із log e; знак важить)")
print("=" * 86)
order = []
for j, nm in enumerate(FNAMES):
    c = np.corrcoef(X[:, j], y)[0, 1]
    c = 0.0 if not np.isfinite(c) else c
    order.append((abs(c), j, c))
    print(f"  {nm:<18s} r = {c:+.4f}   r² = {c*c:.4f}")
order.sort(reverse=True)

print("\n" + "=" * 86)
print("ЧЕСНИЙ ВИМІР — LOO ПО МЕРЕЖАХ (пломба S2)")
print("=" * 86)
r2s, _ = loo_best_single(X, y)
print(f"  найкраща ОДНА ознака, вибір усередині згортки: LOO-R² = {r2s:+.4f}")
for lam in (1.0, 3.0, 10.0, 30.0, 100.0):
    r2, _ = loo_ridge(X, y, lam)
    print(f"  гребінь усіх {len(FNAMES)} ознак, λ={lam:>5.1f}: "
          f"LOO-R² = {r2:+.4f}")
r2f1, _ = loo_ridge(X[:, :1], y, 1e-6)
print(f"  ЛИШЕ F1 (виведений перенос):            LOO-R² = {r2f1:+.4f}")

print("\n" + "=" * 86)
print("🔒 S1 КОНТРОЛЬ «ЧУЖА W» — 400 перемішувань відповідності")
print("=" * 86)
rng = np.random.default_rng(99)
null_s, null_r, null_1 = [], [], []
for _ in range(400):
    p = rng.permutation(M)
    null_s.append(loo_best_single(X, y[p])[0])
    null_r.append(loo_ridge(X, y[p], 10.0)[0])
    null_1.append(loo_ridge(X[:, :1], y[p], 1e-6)[0])
null_s = np.array(null_s)
null_r = np.array(null_r)
null_1 = np.array(null_1)
r2r, _ = loo_ridge(X, y, 10.0)
print(f"  ЛИШЕ F1 (БЕЗ підбору): нуль медіана {np.median(null_1):+.4f}  "
      f"95-й перц. {np.percentile(null_1,95):+.4f}  "
      f"⟹ p = {(null_1 >= r2f1).mean():.4f}")
print(f"  одна ознака (вибір): нуль медіана {np.median(null_s):+.4f}  "
      f"95-й перц. {np.percentile(null_s,95):+.4f}  "
      f"⟹ p = {(null_s >= r2s).mean():.4f}")
print(f"  гребінь λ=10:        нуль медіана {np.median(null_r):+.4f}  "
      f"95-й перц. {np.percentile(null_r,95):+.4f}  "
      f"⟹ p = {(null_r >= r2r).mean():.4f}")

print("\n" + "=" * 86)
print("ВИРОК")
print("=" * 86)
print(f"  🎯 ГОЛОВНЕ ЧИСЛО — F1 сам, ЖОДНОГО підбору: LOO-R² = {r2f1:+.4f}")
print(f"     (ознака виведена з рівняння переносу ДО прогону, λ не тюниться,"
      f" вибору ознак немає)")
print(f"  поріг гілки 0.30 ⟹ {'ВІДКРИТО' if r2f1 > 0.30 else 'НЕ ВІДКРИТО'}"
      f" · контроль «чужа W» "
      f"{'ПРОЙДЕНО' if np.median(null_1) < 0.05 else 'ПРОВАЛЕНО'}")
print(f"  ⚠️ числа з підбором (гребінь/вибір ознаки) — ДОВІДКОВІ, не вирок")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/wfeat99.npz",
         X=X, y=y, NZ=NZ, null_s=null_s, null_r=null_r,
         names=np.array(FNAMES))
print(f"\n  час {time.time()-t0:.0f} с")
