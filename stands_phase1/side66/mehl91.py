"""mehl91 — РОЗКЛАД ІН'ЄКЦІЇ В `σ` НА ОБРИВ РЯДУ vs НЕГАУСОВІСТЬ.

════════════════ ВИВЕДЕННЯ (перевірене вручну) ════════════════
`g(u) = max(m + s·u, 0)`, `u ~ N(0,1)`, `t = m/s`. Гермітові коефіцієнти
`a_n = E[g·He_n]/n!` через `∫_{−t}^∞ He_n φ = He_{n−1}(−t)φ(t)` і
`u·He_n = He_{n+1} + n·He_{n−1}`:
```
a_0 = s·M(t) = s(tΦ+φ)      a_1 = s·Φ(t)      a_n = s·φ(t)·He_{n−2}(−t)/n!  (n≥2)
```
пломба n=2: `sφ/2` ✅ · пломба n=3: прямо `(1/6)[He_3(−t)+3He_1(−t)+tHe_2(−t)]φ
= (1/6)(−t)φ` = `sφ·He_1(−t)/3!` ✅
Тоді за Мелером (`E[He_n(u)He_n(v)] = n!·r^n`):
```
Cov(a_j,a_k) = s_j s_k [ Φ_jΦ_k·r + φ_jφ_k·Σ_{n≥2} He_{n−2}(−t_j)He_{n−2}(−t_k)·r^n/n! ]
```
🔑 **НАША ФОРМУЛА ЗАМИКАННЯ — ЦЕ РІВНО ОБРИВ НА `n=2`**:
`Ch·outer(p,p) + ½·Ch²·outer(f/s,f/s)` ≡ перші два члени. Решта ряду —
лише ПОЕЛЕМЕНТНІ степені `Ch`, жодної нової матриці ⟹ ~0.015% B.

════════════════ ЧОМУ ЦЕ ТРЕБА МІРЯТИ ЗАНОВО ════════════════
Пам'ять: «Мелер K=20 дає ×1.001» — але це вимір НА БАЛІ. За законом каналів
(С91) виправлення ОДНОГО каналу ЛАМАЄ БАЛАНС і на балі дає ~1.0 навіть коли
канал виправлено ідеально (`C` соло ×0.53!). ⟹ **вимір на балі про цей
об'єкт нічого не каже; міряти треба САМУ ІН'ЄКЦІЮ.**

════════════════ ЩО РОЗДІЛЯЄ ЦЕЙ СТЕНД ════════════════
```
ρ(K) = ½·log diag(Wᵀ Cov_Mehler-K(a) W) / diag(Wᵀ Ĉov_іст(a) W)
K=2   наша формула        = обрив + негаусовість
K→∞   точний гаусів парний = САМА негаусовість
```
Якщо `ρ(20) ≪ ρ(2)` — вимога до `σ` знімається ДАРОМ.
Якщо `ρ(20) ≈ ρ(2)` — обрив ні до чого, лишається лише `κ₂₁`.

════════════════ ПЛОМБИ ════════════════
P1 `K=2` мусить збігтися з `relu_moments` ПОБІТОВО (та сама алгебра).
P2 повна `Cov` при `r=1` (діагональ) — не міряємо, діагональ замінюємо на
   точну `va`, як у замиканні; інакше ряд на `r=1` сходиться повільно.
P3 ЧУЖА: коефіцієнти `He(−t)` з ПЕРЕМІШАНИХ нейронів — мусить ЗАШКОДИТИ.
P4 МАСШТАБУВАННЯ по `N`: справжня ін'єкція стоїть, шум `Ĉov` падає як `1/√N`.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) ρ(20)/ρ(2) < 0.5 — обрив дає більше половини ін'єкції   шанс 40%
 (б) ряд зійдеться до K≈8 (далі зміни <2%)                    шанс 75%
 (в) повна σ(h_31) з K=20 впаде нижче 5% (проти 7.7–8.9%)     шанс 45%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
KS = (2, 3, 4, 6, 10, 20)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def relu_mom_K(mh, Ch, K, perm=None):
    """(m,Cov) передактивації → (m,Cov) активації, ряд Мелера до n=K.

    K=2 ≡ формула замикання (пломба P1). perm — контроль «чужа».
    """
    s = Ch.diagonal().clamp(min=1e-20).sqrt()
    t = mh / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mh * p + s * f
    va = ((mh * mh + s * s) * p + mh * s * f - ma * ma).clamp(min=0.0)
    S = torch.outer(s, s)
    R = Ch / S
    acc = torch.outer(p, p) * R
    x = -t if perm is None else -t[perm]
    # He_0=1, He_1=x, He_{m+1} = x·He_m − m·He_{m−1};  член n використовує He_{n−2}
    Hpre = torch.zeros_like(x)                  # He_{−1}, множник m=0 його гасить
    Hm = torch.ones_like(x)                     # He_0(−t)
    Rn = R * R                                  # r²
    fact = 2.0                                  # 2!
    for n in range(2, K + 1):
        m = n - 2
        acc = acc + torch.outer(f * Hm, f * Hm) * Rn / fact
        Hpre, Hm = Hm, x * Hm - float(m) * Hpre
        Rn = Rn * R
        fact *= (n + 1)
    Cn = S * acc
    Ca = Cn + torch.diag(va - Cn.diagonal())
    return ma, Ca


def relu_moments(mh, Ch):
    """Формула замикання як у сабміті — еталон для P1."""
    s = Ch.diagonal().clamp(min=1e-20).sqrt()
    t = mh / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mh * p + s * f
    va = ((mh * mh + s * s) * p + mh * s * f - ma * ma).clamp(min=0.0)
    Cn = Ch * torch.outer(p, p) + (0.5 * (Ch * Ch)) * torch.outer(f / s, f / s)
    return ma, Cn + torch.diag(va - Cn.diagonal())


def empirical(ws, N, seed):
    g = torch.Generator().manual_seed(seed)
    n = ws[0].shape[0]
    MH, CH, CA = [], [], []
    x = torch.randn(N, n, generator=g)
    for k in range(32):
        h = (x @ ws[k].float()).double()
        mh = h.mean(0)
        hc = h - mh
        MH.append(mh); CH.append((hc.T @ hc) / (N - 1))
        a = h.clamp(min=0.0)
        ac = a - a.mean(0)
        CA.append((ac.T @ ac) / (N - 1))
        x = a.float()
    return MH, CH, CA


def chain(ws, K):
    """Повне замикання з рядом Мелера до K. Повертає σ(h_31)."""
    n = ws[0].shape[0]
    Ch = ws[0].T @ ws[0]
    mh = torch.zeros(n, dtype=torch.float64)
    for k in range(32):
        if k == 31:
            return Ch.diagonal().clamp(min=1e-30).sqrt()
        ma, Ca = relu_mom_K(mh, Ch, K)
        mh = ma @ ws[k + 1]
        Ch = ws[k + 1].T @ (Ca @ ws[k + 1])


RHO = {k: [] for k in KS}
RHO['чужа'] = []
CHN = {k: [] for k in KS}
P1 = []
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])).double()
              for i in range(32)]
        MH, CH, CA = empirical(ws, NS, 880000 + 13 * q)
        gp = torch.Generator().manual_seed(9000 + q)
        for K in KS:
            r = torch.zeros(31, 256, dtype=torch.float64)
            for ell in range(31):
                _, Cap = relu_mom_K(MH[ell], CH[ell], K)
                wd = ws[ell + 1]
                dp = (wd.T @ (Cap @ wd)).diagonal().clamp(min=1e-30)
                dt = (wd.T @ (CA[ell] @ wd)).diagonal().clamp(min=1e-30)
                r[ell] = 0.5 * torch.log(dp / dt)
                if K == 2:
                    _, Cb = relu_moments(MH[ell], CH[ell])
                    P1.append(float((Cap - Cb).abs().max()))
            RHO[K].append(r)
            CHN[K].append(torch.log(chain(ws, K)
                                    / CH[31].diagonal().clamp(min=1e-30).sqrt()))
        # ⛔P3 чужа: He(−t) з перемішаних нейронів
        rc = torch.zeros(31, 256, dtype=torch.float64)
        for ell in range(31):
            pm = torch.randperm(256, generator=gp)
            _, Cap = relu_mom_K(MH[ell], CH[ell], 20, perm=pm)
            wd = ws[ell + 1]
            dp = (wd.T @ (Cap @ wd)).diagonal().clamp(min=1e-30)
            dt = (wd.T @ (CA[ell] @ wd)).diagonal().clamp(min=1e-30)
            rc[ell] = 0.5 * torch.log(dp / dt)
        RHO['чужа'].append(rc)
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с · мереж {NNET} · семплів {NS}")
print(f"⛔P1 K=2 ≡ relu_moments: max |Δ| = {max(P1):.3e} "
      f"{'✅' if max(P1) < 1e-12 else '🚨'}\n")

print("=" * 98)
print(f"ІН'ЄКЦІЯ ρ = δσ/σ ЗА ПОРЯДКОМ РЯДУ МЕЛЕРА · {NNET} мереж · N={NS}")
print("=" * 98)
print(f"{'K':>8s}{'|спільна| сер.':>18s}{'розкид сер.':>16s}"
      f"{'розкид ш.30':>15s}{'σ(h_31) спільна':>18s}{'σ(h_31) розкид':>17s}")
base = None
for K in list(KS) + ['чужа']:
    A = torch.stack(RHO[K])
    c = float(A.mean(2).abs().mean())
    s = float(A.std(2).mean())
    s30 = float(A[:, 30].std(1).mean())
    if K == 'чужа':
        print(f"{'⛔чужа':>8s}{100*c:>17.4f}%{100*s:>15.4f}%{100*s30:>14.4f}%"
              f"{'—':>18s}{'—':>17s}")
        continue
    F = torch.stack(CHN[K])
    fc, fs = float(F.mean(1).abs().mean()), float(F.std(1).mean())
    print(f"{K:>8d}{100*c:>17.4f}%{100*s:>15.4f}%{100*s30:>14.4f}%"
          f"{100*fc:>17.3f}%{100*fs:>16.3f}%")
    if K == 2:
        base = (s, fs)

A2, A20 = torch.stack(RHO[2]), torch.stack(RHO[20])
print(f"\n📣 ВІДНОШЕННЯ ρ(20)/ρ(2) по розкиду = "
      f"{float(A20.std(2).mean())/float(A2.std(2).mean()):.4f}")
F2, F20 = torch.stack(CHN[2]), torch.stack(CHN[20])
print(f"📣 повна σ(h_31) розкид: K=2 {100*float(F2.std(1).mean()):.3f}% → "
      f"K=20 {100*float(F20.std(1).mean()):.3f}%  "
      f"⟹ ×{float(F2.std(1).mean())/max(float(F20.std(1).mean()),1e-30):.2f}")
print(f"📣 вимога карти ladder91: розкид σ ≤ ~1% ⟹ "
      f"{'✅ ДОСЯГНУТО' if float(F20.std(1).mean()) < 0.01 else 'ще не досягнуто, лишається ×%.1f' % (float(F20.std(1).mean())/0.01)}")
np.savez(f"/mnt/win_d/llm/White-Box/arena62/logs/mehl91_N{NS}.npz",
         **{f"rho_{k}": torch.stack(RHO[k]).numpy() for k in KS},
         rho_foreign=torch.stack(RHO['чужа']).numpy(),
         **{f"chain_{k}": torch.stack(CHN[k]).numpy() for k in KS})
