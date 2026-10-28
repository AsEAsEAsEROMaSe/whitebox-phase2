"""calib99 — КАЛІБРУВАННЯ СТЕНДУ МАЛОЇ ШИРИНИ. ЧИ ПЕРЕНОСЯТЬСЯ ВИРОКИ n=32 → n=256?

════════════════ НАВІЩО ════════════════
С98 збудувала стенд n=24..32 (глибина 32 СКРІЗЬ), де сигнал/шум = 116 замість
~1 при n=256. Ним збираються виносити вироки, нерозв'язні при n=256. Закон
проєкту: «стенд, що рахує БАЛ, мусить відтворити ВІДОМЕ число ДО висновків».
Прилад НЕ відкалібровано ⟹ усе, що на ньому побудують, — під питанням.

════════════════ ЩО САМЕ МІРЯЄМО ════════════════
Чотири оракульні канали з ВІДОМИМИ вироками при n=256 (С88, 32 мережі):
```
оракул лише m              ×0.912   (шкодить)
оракул лише діагональ C    ×0.243   (сильно шкодить)
оракул лише позадіагональ  ×0.634   (шкодить)
оракул m + повна C РАЗОМ   ×20.97   (сильно допомагає)
```
Набір перекриває ДВА порядки величини і містить знак в ОБИДВА боки.

ОЗНАЧЕННЯ ІН'ЄКЦІЇ (те, що робить числа самоузгодженими): оракул підставляється
на шарах 0..30, а вихід шару 31 рахує ЗАМИКАННЯ ⟹ «m+C разом» лишає РІВНО
одношагову похибку вихідного відображення. Це узгоджено з «усе+вихід ×400.85».

════════════════ МЕТРИКА — ТА, ЩО ДАЄ ЦІ ЧИСЛА ════════════════
🚨 БЕЗ вільного масштабу числа НЕ відтворяться: одношагова похибка `E[relu]` з
точними (μ,σ,ρ) = 0.139%, а 0.5070%/20.97 = 0.0242% ⟹ ×5.7 з'їдає саме `α`.
⟹ LOO-α ПО МЕРЕЖАХ (закон: на зрізі з N мереж α рахується по N−1; при 6
мережах база штучно слабка ×1.618, при 32 → ×1.022) ⟹ ≥24 мережі обов'язково.

════════════════ ПЛОМБИ ════════════════
 P1 🔒 ПІДЛОГА ШУМУ ОРАКУЛА. Оракульна статистика накопичується ДВОМА
      незалежними половинами; варіант женеться тричі (повна · A · B).
      Розкид k(A) vs k(B) — це шум ін'єкції. Якщо він порівнянний із
      різницею між шириною, вирок нерозв'язний, і це треба знати ДО висновку.
 P2 🔒 СІД ІСТИНИ ≠ СІД ОРАКУЛА (закон: спільний сід підганяє шум оракула під
      шум істини і дав +7% фіктивного виграшу).
 P3 🔒 «ПОВНИЙ ОРАКУЛ» = m+C на ВСІХ шарах включно з 31 мусить дати РІВНО 0
      (тавтологія) ⟹ ін'єкція прикладається туди, куди я думаю.
 P4 🔒 діаг+позадіаг разом мусить дати РІВНО те саме, що «C повна».
 P5 🔒 база стенду при n=32 мусить збігтися з nlaw97 (9.6108%) — інакше це
      інший об'єкт, а не інша ширина.

════════════════ ПЕРЕДБАЧЕННЯ (ЗАПИСАНО ДО ПРОГОНУ) ════════════════
 (а) ЗНАК зберігається на всіх чотирьох (три <1, один ≫1)          60%
 (б) ПОРЯДОК зберігається на всіх чотирьох (у межах ×2)            35%
 (в) 🔑 «m+C разом» переноситься НАЙКРАЩЕ з чотирьох. Механізм: і база
     (накопичена за 32 кроки), і залишок (одношаговий) — обидва ефекти
     скінченної ширини ~1/n ⟹ їхнє ВІДНОШЕННЯ від `n` майже не залежить  50%
 (г) 🔑 «діагональ соло ×0.243» переноситься НАЙГІРШЕ. Механізм: це число є
     наслідком ВЗАЄМНОГО ГАСІННЯ каналів (закон «замикання = збалансована
     система похибок»), а баланс — величина другого порядку, і немає
     причини, щоб він тримався при зміні ширини у 8×                 55%
 (д) підлога шуму оракула (P1) буде <10% від сигналу при n≤48         70%
 (е) при n=24..48 монотонність по n БЕЗ зламу                        45%
"""
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np                                            # noqa: E402
from scipy.special import ndtr                                # noqa: E402

SQ2PI = math.sqrt(2.0 * math.pi)
DEPTH = 32


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
    rng = np.random.default_rng(seed)
    sd = math.sqrt(2.0 / n)
    return [rng.standard_normal((n, n)) * sd for _ in range(DEPTH)]


def price_pairs(t, rho, chunk=64):
    """E[(x+t_j)₊(y+t_k)₊], точна квадратура; блоками по рядках (пам'ять)."""
    n = t.size
    out = np.empty((n, n))
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        tj = np.repeat(t[i0:i1], n)
        tk = np.tile(t, i1 - i0)
        rr = rho[i0:i1].ravel()
        q = np.sqrt(np.maximum(1.0 - rr * rr, 1e-14))
        x = ZQ[None, :] - tj[:, None]
        u = (rr[:, None] * x + tk[:, None]) / q[:, None]
        out[i0:i1] = ((WQ[None, :] * ZQ[None, :] * phi(x) * q[:, None]
                       * psi(u)).sum(1)).reshape(i1 - i0, n)
    return out


def step(m, C, WL):
    """один крок гаусового замикання: (m,C) активацій → (m,C) наступних."""
    mp = WL.T @ m
    Cp = WL.T @ C @ WL
    s2 = np.maximum(np.diag(Cp), 1e-300)
    s = np.sqrt(s2)
    t = mp / s
    rho = np.clip(Cp / (s[:, None] * s[None, :]), -0.999999, 0.999999)
    M2 = price_pairs(t, rho) * (s[:, None] * s[None, :])
    np.fill_diagonal(M2, s2 * ((1 + t * t) * ndtr(t) + t * phi(t)))
    ma = s * psi(t)
    Ca = M2 - np.outer(ma, ma)
    return ma, 0.5 * (Ca + Ca.T)


def run_variant(W, n, chan, TM=None, TC=None, upto=DEPTH - 1):
    """chan ⊂ {'m','diag','off'}; ін'єкція на шарах 0..upto-1, вихід — замикання.
    upto=DEPTH ⟹ повний оракул (пломба P3, мусить дати рівно істину)."""
    m = np.zeros(n)
    C = np.eye(n)
    for L in range(DEPTH):
        ma, Ca = step(m, C, W[L])
        if L < upto and chan:
            if "m" in chan:
                ma = TM[L].copy()
            if "diag" in chan and "off" in chan:
                Ca = TC[L].copy()
            elif "diag" in chan:
                Ca = Ca.copy()
                np.fill_diagonal(Ca, np.diag(TC[L]))
            elif "off" in chan:
                d = np.diag(Ca).copy()
                Ca = TC[L].copy()
                np.fill_diagonal(Ca, d)
        m, C = ma, Ca
    return m


def truth_full(W, n, N, seed, need_cov=True):
    """пошарові m,C активацій. Повертає (TM,TC, TM_A,TC_A, TM_B,TC_B, шум_виходу).
    Половини A/B — НЕЗАЛЕЖНІ підвибірки ⟹ дають підлогу шуму оракула (P1)."""
    rng = np.random.default_rng(seed)
    S = [np.zeros((2, n)) for _ in range(DEPTH)]
    Q = ([np.zeros((2, n, n)) for _ in range(DEPTH)] if need_cov else None)
    cnt = np.zeros(2)
    CH = max(1, min(1 << 15, N // 2))
    done = 0
    k = 0
    while done < N:
        b = min(CH, N - done)
        h = rng.standard_normal((b, n))
        j = k & 1
        for L in range(DEPTH):
            h = np.maximum(h @ W[L], 0.0)
            S[L][j] += h.sum(0)
            if need_cov:
                Q[L][j] += h.T @ h
        cnt[j] += b
        done += b
        k += 1
    out = {}
    for tag, sel in (("F", (0, 1)), ("A", (0,)), ("B", (1,))):
        nn = sum(cnt[i] for i in sel)
        TM = [sum(S[L][i] for i in sel) / nn for L in range(DEPTH)]
        TC = None
        if need_cov:
            TC = [sum(Q[L][i] for i in sel) / nn - np.outer(TM[L], TM[L])
                  for L in range(DEPTH)]
        out[tag] = (TM, TC)
    dif = S[DEPTH - 1][0] / cnt[0] - S[DEPTH - 1][1] / cnt[1]
    noise = 0.5 * math.sqrt(float((dif * dif).mean()))
    return out, noise


# ─────────────────── МЕТРИКА: LOO-α ПО МЕРЕЖАХ ───────────────────
def loo_alpha_err(preds, truths):
    """preds/truths: списки (n,). Повертає (медіана, RMS) відн. похибок після
    LOO-калібровки ОДНИМ глобальним скаляром (1+α)."""
    u = [p / np.linalg.norm(t) for p, t in zip(preds, truths)]
    v = [t / np.linalg.norm(t) for t in truths]
    num = np.array([float(np.dot(a, b)) for a, b in zip(u, v)])
    den = np.array([float(np.dot(a, a)) for a in u])
    errs = []
    for i in range(len(u)):
        g = (num.sum() - num[i]) / max(den.sum() - den[i], 1e-300)
        errs.append(float(np.linalg.norm(g * u[i] - v[i])))
    e = np.array(errs)
    return float(np.median(e)), float(math.sqrt(float((e * e).mean()))), e


VARIANTS = [
    ("база",              set()),
    ("оракул m",          {"m"}),
    ("оракул діаг C",     {"diag"}),
    ("оракул позадіаг C", {"off"}),
    ("оракул C повна",    {"diag", "off"}),
    ("оракул m + C",      {"m", "diag", "off"}),
]

if __name__ == "__main__":
    WIDTHS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [32]
    NNETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else (1 << 21)
    print(f"calib99 · ширини {WIDTHS} · {NNETS} мереж · N={NSAMP}", flush=True)
    t0 = time.time()

    for n in WIDTHS:
        P = {v: [] for v, _ in VARIANTS}
        PA = {v: [] for v, _ in VARIANTS}
        PB = {v: [] for v, _ in VARIANTS}
        T = []
        noises = []
        seal3 = []
        seal4 = []
        for k in range(NNETS):
            W = make_net(n, 900000 + 1000 * n + k)
            # P2: сід ОРАКУЛА і сід ІСТИНИ різні
            orc, _ = truth_full(W, n, NSAMP, 200000 + 31 * k + n)
            tru, nz = truth_full(W, n, NSAMP, 700000 + 17 * k + n,
                                 need_cov=False)
            tm = tru["F"][0][DEPTH - 1]
            T.append(tm)
            sc = float(np.linalg.norm(tm))
            noises.append(nz * math.sqrt(n) / sc)
            for name, chan in VARIANTS:
                P[name].append(run_variant(W, n, chan, *orc["F"]))
                if chan:
                    PA[name].append(run_variant(W, n, chan, *orc["A"]))
                    PB[name].append(run_variant(W, n, chan, *orc["B"]))
            # P3: повний оракул на ВСІХ шарах ⟹ рівно оракульна істина
            fo = run_variant(W, n, {"m", "diag", "off"}, *orc["F"], upto=DEPTH)
            seal3.append(float(np.linalg.norm(fo - orc["F"][0][DEPTH - 1])
                               / max(sc, 1e-300)))
            # P4: діаг+позадіаг == C повна
            seal4.append(float(np.linalg.norm(P["оракул C повна"][-1]
                                              - run_variant(W, n,
                                                            {"diag", "off"},
                                                            *orc["F"]))))
            if k == 0:
                print(f"  n={n}: 1-ша мережа за {time.time()-t0:.0f} с",
                      flush=True)

        print("\n" + "=" * 92)
        print(f"n = {n}   ·   {NNETS} мереж   ·   N = {NSAMP}   "
              f"·   глибина {DEPTH}")
        print("=" * 92)
        print(f"🔒P3 повний оракул (мусить бути 0): {max(seal3):.3e}")
        print(f"🔒P4 діаг+поза == C повна (мусить бути 0): {max(seal4):.3e}")
        print(f"   шум МК на виході: {100*float(np.median(noises)):.4f}%")
        raw = float(np.median([float(np.linalg.norm(p - t) / np.linalg.norm(t))
                               for p, t in zip(P["база"], T)]))
        print(f"🔒P5 сира база БЕЗ α (проти nlaw97): {100*raw:.4f}%")
        base_med, base_rms, _ = loo_alpha_err(P["база"], T)
        print(f"\n{'варіант':>20s}{'медіана':>11s}{'RMS':>11s}"
              f"{'k(мед)':>10s}{'k(RMS)':>10s}{'шум k (A|B)':>18s}")
        for name, chan in VARIANTS:
            med, rms, _ = loo_alpha_err(P[name], T)
            km = base_med / max(med, 1e-300)
            kr = base_rms / max(rms, 1e-300)
            if chan:
                ma, _, _ = loo_alpha_err(PA[name], T)
                mb, _, _ = loo_alpha_err(PB[name], T)
                sp = f"{base_med/max(ma,1e-300):.3f}|{base_med/max(mb,1e-300):.3f}"
            else:
                sp = "—"
            print(f"{name:>20s}{100*med:>10.4f}%{100*rms:>10.4f}%"
                  f"{km:>10.3f}{kr:>10.3f}{sp:>18s}", flush=True)
        print(f"\n  час {time.time()-t0:.0f} с")
