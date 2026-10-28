"""accum95 — НАКОПИЧЕНА РЕКУРСІЯ З ОДНОВИМІРНИМ ЕДЖВОРТОМ ПО (γ₃,γ₄)

════════════════ ЗВІДКИ (ланцюг вимірів С95) ════════════════
`[ЗМІРЯНО newalg95d, 4 мережі × 2¹⁸, ОДИН крок, ТОЧНІ 1-D моменти]`
```
шар   σ гаус   σ+γ₃    σ+γ₃γ₄   ×γ₃γ₄     E[relu] гаус → +γ₃γ₄   ×
  7   1.547%  0.734%   0.120%   12.94      0.220% → 0.014%     16.10
 15   1.799%  0.868%   0.223%    8.07      0.167% → 0.023%      7.30
 23   1.781%  1.024%   0.308%    5.78      0.150% → 0.032%      4.73
 30   1.802%  0.995%   0.369%    4.88      0.139% → 0.030%      4.57
```
`[ЗМІРЯНО там само]` 79.5% похибки σ на шарі 30 — у квартилі з найменшим |t|
`[ЗМІРЯНО newalg95c]` вхід СФЕРА не змінює нічого ⟹ це НЕ радіальна дисперсія
`[ЗМІРЯНО newalg95b]` умовна густина по R дає ×1.02 ⟹ ГЛОБАЛЬНИЙ масштаб не той
   об'єкт; потрібні ПОКООРДИНАТНІ γ₃,γ₄ (вектори 256, НЕ тензори)

📐 ЗАМКНЕНА ФОРМА (виведено через ∫_{−t}^∞ He_n(y)φ(y)dy = He_{n−1}(−t)φ(t)):
```
E[ReLU]/σ   = ψ(t) − (γ₃/6)·t·φ  + (γ₄/24)·(t²−1)·φ ,   ψ(t)=φ(t)+tΦ(t)
E[ReLU²]/σ² = (1+t²)Φ + t·φ + γ₃·φ/3 − γ₄·t·φ/12
```
🔒 член γ₃ першого моменту = −γ₃σtφ/6 — ЗБІГАЄТЬСЯ зі старою Δm=−κ₃tφ/(6s²) ✅
Позадіагональ: лишаємо кореляцію Мелера, перемасштабовуємо на нові σ.

════════════════ ВАРІАНТИ ════════════════
 A  чистий Мелер K=24 (база)
 B  Еджворт з ОРАКУЛЬНИМИ γ₃,γ₄ (з МК 2¹⁸) — СТЕЛЯ гілки
 C  Еджворт з γ₃,γ₄ із РОБОЧОЇ вибірки 12288 — реалістичний метод
 D  ⛔ЧУЖІ γ₃,γ₄ (з іншої мережі) — контроль «чужа поправка»
 E  Еджворт лише γ₃ (оракульні) — чи потрібен γ₄
 λ-скан по B: оптимум мусить бути при λ>0 (контроль С89)

Ціль — `final_means` з датасету (істина 1e9 семплів).

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) 🔑 B (оракульна) дасть ×2.5…×6 у накопиченій похибці       шанс 55%
 (б) C (з 12288) дасть >×1.8                                    шанс 45%
 (в) D (чужа) дасть <×1.1                                       шанс 80%
 (г) E (лише γ₃) буде помітно слабша за B                       шанс 70%
 (д) λ-оптимум B буде в 0.6…1.0                                 шанс 55%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NSAMP = 1 << LOGN
NWORK = 12288
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks()
                        .flatten()).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i]
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def cumulants(W, seed, n, dev):
    """γ₃,γ₄ передактивації КОЖНОГО шару з МК на n семплах."""
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 256, generator=g, device=dev)
    G3, G4 = [], []
    for L in range(32):
        s1 = torch.zeros(256, dtype=torch.float64, device=dev)
        s2 = torch.zeros(256, dtype=torch.float64, device=dev)
        Wd = W[L]
        for o in range(0, n, CH):
            pb = (X[o:o + CH] @ Wd).double()
            s1 += pb.sum(0); s2 += (pb * pb).sum(0)
        mu = s1 / n
        sd = (s2 / n - mu * mu).clamp_min(1e-300).sqrt()
        s3 = torch.zeros(256, dtype=torch.float64, device=dev)
        s4 = torch.zeros(256, dtype=torch.float64, device=dev)
        for o in range(0, n, CH):
            c = ((X[o:o + CH] @ Wd).double() - mu) / sd
            s3 += (c ** 3).sum(0); s4 += (c ** 4).sum(0)
        G3.append(s3 / n)
        G4.append(s4 / n - 3.0)
        X = torch.relu(X @ Wd)
    return G3, G4


def step(m, C, g3, g4, lam):
    """Крок замикання: Мелер + одновимірна Еджвортова поправка на діагональ."""
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    # ── перший момент
    ma = s * (P * t + f)
    if g3 is not None:
        ma = ma + lam * s * (-(g3 / 6.0) * t * f + (g4 / 24.0) * (t * t - 1) * f)
    # ── ряд Мелера для повної Cov
    rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
    a1 = s * P
    cov = torch.outer(a1, a1) * rho
    Rn = rho.clone()
    Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
    fact, sf = 2.0, s * f
    for nn in range(2, MK + 1):
        Rn = Rn * rho
        an = sf * Hm / fact
        cov = cov + torch.outer(an, an) * Rn * fact
        Hpre, Hm = Hm, (-t) * Hm - float(nn - 2) * Hpre
        fact *= (nn + 1)
    if g3 is not None:
        # нова діагональ з Еджворта; кореляції Мелера зберігаємо
        m2 = (1.0 + t * t) * P + t * f + lam * (g3 * f / 3.0
                                                - g4 * t * f / 12.0)
        v_new = (s * s * m2 - ma * ma).clamp_min(1e-30)
        v_old = cov.diagonal().clamp_min(1e-30)
        sc = (v_new / v_old).sqrt()
        cov = cov * torch.outer(sc, sc)
    return ma, cov


def run(W, G3, G4, lam=1.0, only3=False, last_only=False):
    m = torch.zeros(256, dtype=torch.float64, device=W[0].device)
    C = W[0].double().T @ W[0].double()
    for L in range(32):
        g3 = g4 = None
        if G3 is not None and (not last_only or L == 31):
            g3 = G3[L]
            g4 = torch.zeros_like(g3) if only3 else G4[L]
        ma, cov = step(m, C, g3, g4, lam)
        if L == 31:
            return ma
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


print(f"accum95 · {DEV} · {NNET} мереж · оракул N=2^{LOGN} · робоча {NWORK}",
      flush=True)
t0 = time.time()
KEYS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] + [
    f'L{k}' for k in (25, 50, 75, 125)]
ERR = {k: [] for k in KEYS}
prev = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        G3o, G4o = cumulants(W, 954000 + 157 * q, NSAMP, DEV)
        G3w, G4w = cumulants(W, 977000 + 163 * q, NWORK, DEV)

        def er(x):
            return float((x - tru).norm() / tru.norm())
        ERR['A'].append(er(run(W, None, None)))
        ERR['B'].append(er(run(W, G3o, G4o)))
        ERR['C'].append(er(run(W, G3w, G4w)))
        ERR['E'].append(er(run(W, G3o, G4o, only3=True)))
        ERR['F'].append(er(run(W, G3o, G4o, last_only=True)))
        ERR['G'].append(er(run(W, G3w, G4w, last_only=True)))
        if prev is not None:
            ERR['H'].append(er(run(W, prev[0], prev[1], last_only=True)))
        if prev is not None:
            ERR['D'].append(er(run(W, prev[0], prev[1])))
        for k in (25, 50, 75, 125):
            ERR[f'L{k}'].append(er(run(W, G3o, G4o, lam=k / 100.0)))
        prev = (G3o, G4o)
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: A={ERR['A'][-1]*100:.4f}% B={ERR['B'][-1]*100:.4f}%"
                  f" C={ERR['C'][-1]*100:.4f}% ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 76)
print("НАКОПИЧЕНА ПОХИБКА ПРОТИ `final_means` (істина 1e9)")
print("=" * 76)
base = float(np.mean(ERR['A']))
mse = {k: float(np.mean(np.array(v) ** 2)) for k, v in ERR.items() if v}
names = {'A': 'чистий Мелер K=24 (база)', 'B': '✅Еджворт ОРАКУЛЬНІ γ₃,γ₄',
         'C': '🟢Еджворт з робочої вибірки 12288', 'D': '⛔ЧУЖІ γ₃,γ₄',
         'E': 'Еджворт лише γ₃ (оракул)',
         'F': '🔥Еджворт ЛИШЕ НА ОСТАННЬОМУ кроці (оракул)',
         'G': '🔥Едж ЛИШЕ ОСТАННІЙ, γ з вибірки 12288',
         'H': '⛔ЧУЖІ γ, лише останній крок'}
print(f"{'вар':>4s}{'віднос.похибка':>16s}{'MSE ×':>9s}{'RMS ×':>8s}  опис")
for k in ['A', 'F', 'G', 'H', 'B', 'C', 'E', 'D']:
    if not ERR[k]:
        continue
    r = float(np.mean(ERR[k]))
    print(f"{k:>4s}{r*100:>15.4f}%{mse['A']/mse[k]:>9.2f}"
          f"{base/r:>8.3f}  {names[k]}")
print("\nλ-скан по B (контроль С89: оптимум мусить бути при λ>0)")
for k in (25, 50, 75, 100, 125):
    key = 'B' if k == 100 else f'L{k}'
    print(f"  λ={k/100:.2f}: {np.mean(ERR[key])*100:.4f}%  "
          f"MSE ×{mse['A']/mse[key]:.2f}")
print(f"\n🔑 ЧИТАТИ: B — стеля гілки, C — те, що реально можна подати.")
print(f"   D мусить бути ≈×1.0, інакше виграш = ефект калібровки, не інформація.")
