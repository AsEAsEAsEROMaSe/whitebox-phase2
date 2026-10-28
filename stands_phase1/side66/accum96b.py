"""accum96b — РОЗЧЕПИТИ ЗВ'ЯЗКУ: Еджворт на `m` ОКРЕМО від Еджворта на дисперсію

════════════════ ЧОМУ САМЕ ЦЕ ════════════════
`[ЗМІРЯНО ladder96/ladderm96, С96]` канал `m` несе **×22.6**, уся `Cov`
поверх точного `m` — лише **×1.16**. Курс обміну: зняти 25% похибки `m`
щошару = **×11.03** у MSE.
`[ЗМІРЯНО accum95, С95]` варіант B (Еджворт) дав **×0.83**, тобто ГІРШЕ бази.
🔑 АЛЕ B ПРАВИВ ОБИДВА МОМЕНТИ ОДНОЧАСНО:
```
   ma  += s·(−(γ₃/6)tφ + (γ₄/24)(t²−1)φ)          ← канал `m`,  вага ×22.6
   v_new = s²·[(1+t²)Φ + tφ + γ₃φ/3 − γ₄tφ/12] − ma²
   cov  *= outer(√(v_new/v_old))                   ← канал `σ`,  вага ×1.16
```
⟹ вирок «Еджворт мертвий» стосувався ЗВ'ЯЗКИ. Канал, що несе 95% ваги,
   ОКРЕМО НІКОЛИ НЕ МІРЯВСЯ. Це і є розкоп.

════════════════ ВАРІАНТИ ════════════════
 A   чистий Мелер                          🔒ПЛОМБА 0.847121% на 24 мережах
 E   🔥Еджворт ЛИШЕ на `m`, `Cov` НЕ ЧІПАЄМО        ← ГОЛОВНИЙ
 V   Еджворт лише на дисперсію (дзеркало E)
 EV  обидва разом = варіант B із accum95    🔒ПЛОМБА звірки: має дати ≈×0.83
 S   оракульна σ щошару (`C *= outer(sc,sc)`, ρ ЗБЕРЕЖЕНО) — стеля каналу σ
 ES  Еджворт на `m` + оракульна σ
 EW  🟢E, але γ з РОБОЧОЇ вибірки 12288 — те, що можна подати
 F   ⛔E з ЧУЖИМИ γ (попередня мережа) — контроль

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) 🔑 E дасть ×3…×20 (чіпає рівно канал вагою ×22.6)          шанс 45%
 (б) EV відтворить ×0.8…×0.9 (звірка з accum95 B)               шанс 70%
 (в) V сам по собі дасть <×1.0 (тобто це ВІН псував зв'язку)    шанс 60%
 (г) S (оракульна σ) дасть <×3 — бо σ важить мало               шанс 55%
 (д) F (чужі γ) дасть <×1.1                                     шанс 70%
 (е) EW ≈ E (шум вибірки 12288 не вузьке місце, С95)            шанс 75%
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

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
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


def stats(W, seed, n, dev):
    """(σ, γ₃, γ₄) передактивації КОЖНОГО шару — понейронні вектори."""
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 256, generator=g, device=dev)
    SD, G3, G4 = [], [], []
    for L in range(32):
        Wd = W[L]
        s1 = torch.zeros(256, dtype=torch.float64, device=dev)
        s2 = torch.zeros(256, dtype=torch.float64, device=dev)
        for o in range(0, n, CH):
            pb = (X[o:o + CH] @ Wd).double()
            s1 += pb.sum(0)
            s2 += (pb * pb).sum(0)
        mu = s1 / n
        sd = (s2 / n - mu * mu).clamp_min(1e-300).sqrt()
        s3 = torch.zeros(256, dtype=torch.float64, device=dev)
        s4 = torch.zeros(256, dtype=torch.float64, device=dev)
        for o in range(0, n, CH):
            c = ((X[o:o + CH] @ Wd).double() - mu) / sd
            s3 += (c ** 3).sum(0)
            s4 += (c ** 4).sum(0)
        SD.append(sd)
        G3.append(s3 / n)
        G4.append(s4 / n - 3.0)
        X = torch.relu(X @ Wd)
    return SD, G3, G4


def run(W, SD, G3, G4, em=False, ev=False, osig=False, lam=1.0,
        ksig=1.0):
    """em — Еджворт на `m`;  ev — Еджворт на дисперсію;  osig — оракульна σ."""
    dev = W[0].device
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = W[0].double().T @ W[0].double()
    for L in range(32):
        if osig:                       # ρ ЗБЕРІГАЄМО: масштабуємо симетрично
            s0 = C.diagonal().clamp_min(1e-30).sqrt()
            sc = ((1.0 - ksig) * s0 + ksig * SD[L]) / s0
            C = C * torch.outer(sc, sc)
        s = C.diagonal().clamp_min(1e-30).sqrt()
        t = m / s
        P, f = Phi(t), phi(t)
        ma = s * (P * t + f)
        g3, g4 = G3[L], G4[L]
        if em:
            ma = ma + lam * s * (-(g3 / 6.0) * t * f
                                 + (g4 / 24.0) * (t * t - 1) * f)
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
        if ev:
            m2 = ((1.0 + t * t) * P + t * f
                  + lam * (g3 * f / 3.0 - g4 * t * f / 12.0))
            v_new = (s * s * m2 - ma * ma).clamp_min(1e-30)
            sc2 = (v_new / cov.diagonal().clamp_min(1e-30)).sqrt()
            cov = cov * torch.outer(sc2, sc2)
        if L == 31:
            return ma
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


print(f"accum96b · {DEV} · {NNET} мереж · оракул N=2^{LOGN}", flush=True)
t0 = time.time()
KEYS = ['A', 'E', 'V', 'EV', 'S', 'ES', 'EW', 'F', 'E50', 'E150'] + \
    [f'K{k}' for k in (25, 50, 75, 90, 95)] + ['SF', 'ESW', 'ESW2', 'ESWV']
ERR = {k: [] for k in KEYS}
prev = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        SDo, G3o, G4o = stats(W, 962000 + 151 * q, NSAMP, DEV)
        SDw, G3w, G4w = stats(W, 985000 + 169 * q, NWORK, DEV)

        def er(x):
            return float((x - tru).norm() / tru.norm())
        ERR['A'].append(er(run(W, SDo, G3o, G4o)))
        ERR['E'].append(er(run(W, SDo, G3o, G4o, em=True)))
        ERR['V'].append(er(run(W, SDo, G3o, G4o, ev=True)))
        ERR['EV'].append(er(run(W, SDo, G3o, G4o, em=True, ev=True)))
        ERR['S'].append(er(run(W, SDo, G3o, G4o, osig=True)))
        ERR['ES'].append(er(run(W, SDo, G3o, G4o, em=True, osig=True)))
        ERR['EW'].append(er(run(W, SDw, G3w, G4w, em=True)))
        ERR['ESW'].append(er(run(W, SDw, G3w, G4w, em=True, osig=True)))
        ERR['ESW2'].append(er(run(W, SDw, G3w, G4w, em=True, osig=True,
                                  lam=0.5)))
        ERR['ESWV'].append(er(run(W, SDw, G3w, G4w, em=True, ev=True,
                                  osig=True)))
        ERR['E50'].append(er(run(W, SDo, G3o, G4o, em=True, lam=0.5)))
        ERR['E150'].append(er(run(W, SDo, G3o, G4o, em=True, lam=1.5)))
        for k in (25, 50, 75, 90, 95):
            ERR[f'K{k}'].append(er(run(W, SDo, G3o, G4o, em=True, osig=True,
                                       ksig=k / 100.0)))
        if prev is not None:
            ERR['F'].append(er(run(W, SDo, prev[0], prev[1], em=True)))
            ERR['SF'].append(er(run(W, prev[2], G3o, G4o, em=True, osig=True)))
        prev = (G3o, G4o, SDo)
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: A={ERR['A'][-1]*100:.4f}% E={ERR['E'][-1]*100:.4f}%"
                  f" V={ERR['V'][-1]*100:.4f}% EV={ERR['EV'][-1]*100:.4f}%"
                  f" S={ERR['S'][-1]*100:.4f}% ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 80)
print("РОЗЧЕПЛЕНА ЗВ'ЯЗКА: ЩО САМЕ РОБИТЬ ЕДЖВОРТ У КОЖНОМУ КАНАЛІ")
print("=" * 80)
mse = {k: float(np.mean(np.array(v) ** 2)) for k, v in ERR.items() if v}
V0 = 2.05e-6
names = {'A': '🔒чистий Мелер (пломба 0.847121%)',
         'E': '🔥Еджворт ЛИШЕ на `m` — ГОЛОВНИЙ',
         'V': 'Еджворт лише на дисперсію',
         'EV': '🔒зв’язка = accum95 B (звірка: має бути ≈×0.83)',
         'S': 'ОРАКУЛЬНА σ щошару (ρ збережено)',
         'ES': 'Еджворт на `m` + оракульна σ',
         'EW': '🟢E, γ з робочої вибірки 12288',
         'F': '⛔E з ЧУЖИМИ γ — контроль',
         'E50': 'E, λ=0.50', 'E150': 'E, λ=1.50',
         'SF': '⛔ES із ЧУЖОЮ σ — контроль на канал σ',
         'ESW': '🔥🔥σ І γ З ВИБІРКИ 12288 — ЦЕ МОЖНА ПОДАТИ',
         'ESW2': '🔥ESW, λ=0.50',
         'ESWV': 'ESW + Еджворт і на дисперсію'}
for _k in (25, 50, 75, 90, 95):
    names[f'K{_k}'] = f'драбина σ: κ={_k/100:.2f} (похибка σ {(1-_k/100)*1.80:.2f}%)'
print(f"{'вар':>5s}{'віднос.похибка':>16s}{'MSE ×A':>9s}{'бал (оцінка)':>15s}"
      f"  опис")
for k in KEYS:
    if not ERR[k]:
        continue
    r = float(np.mean(ERR[k]))
    e2 = 0.9097 * r * r
    sc = 0.1 * V0 * e2 / (V0 + e2)
    print(f"{k:>5s}{r*100:>15.4f}%{mse['A']/mse[k]:>9.2f}{sc:>15.3e}  "
          f"{names[k]}")
print("\n🔑 ЧИТАТИ: якщо E ≫ EV — вирок С95 «Еджворт мертвий» стосувався")
print("   ЗВ'ЯЗКИ, а канал `m` живий. Рекорд для порівняння: 1.86762e-07.")
