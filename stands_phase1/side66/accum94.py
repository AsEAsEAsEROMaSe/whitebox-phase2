"""accum94 — ЧИ ВИЖИВАЄ κ₃-ПОПРАВКА ДО `C` У НАКОПИЧЕНОМУ ПРОГОНІ.

════════════════ ЧОМУ ЦЕ ЄДИНИЙ ЧЕСНИЙ ВИМІР ════════════════
`[ЦИТАТА logs/r538.log]` maxent давав **×31 на ОДНОМУ кроці** і **×1.37** після
32 шарів (M≈B до 4-го знака). ⟹ виграш кроку НЕ Є виграшем методу.
`[ЗМІРЯНО sigstep94, пломба масштабування 2¹⁷/2¹⁹]` κ₃-поправка до кроку `C`
знімає **×7.27** розкиду `σ` (1.0119% → 0.1391%), на глибині ×6.73, і залишок
СТОЇТЬ при ×4 семплів ⟹ не шум.
🔑 ПИТАННЯ: ×7 на кроці — це ×7 на виході чи ×1.0, як у maxent?

════════════════ ЯК ЗРОБЛЕНО ЧЕСНО ════════════════
🚨 Оракульний `T` прив'язаний до ІСТИННОГО стану, рекурсія ж несе СВІЙ. Пряма
підстановка `T` була б «істина в канал» — те, що пам'ять називає мертвим завжди.
⟹ `T` несеться у СТАНДАРТИЗОВАНІЙ формі `T̃_jk = T_jk/(s_j²·s_k)` (безрозмірна
ФОРМА), і на кожному кроці домножається на `s` ПОТОЧНОГО стану. Так поправка
переносить лише те, що ми потім реально зможемо нести CP-формою.

════════════════ ВАРІАНТИ ════════════════
```
A  чиста рекурсія Мелера (K=24)                        ← база
B  A + κ₃-поправка до Cov, форма T̃ оракульна           ← ЦІЛЬ
C  A, але `C` ІН'ЄКТУЄТЬСЯ істинна щошару              ← ✅поз.контроль (r538-D)
X  A + κ₃-поправка з ЧУЖОЮ формою T̃ (інша мережа)     ← ⛔контроль «чужа»
```

════════════════ ПЛОМБИ ════════════════
P1 🔒 `C` мусить дати різке падіння (r538: ×18.7 у MSE) — інакше стенд не той.
P2 ⛔ `X` мусить бути ≈ `A` або гірше. Якщо чужа форма теж дає ×7 — весь виграш
   є ефектом калібровки, а не інформації (С88: `F(t,t,ρ)` ×1.026 при чужій ×1.023).
P3 🔒 істина — `final_means` з офіційного датасету (1e9 семплів), НЕ наша МК.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) B дасть ×3…×10 у MSE проти A                    шанс 50%
 (б) B дасть <×1.5 (з'їдається, як maxent)            шанс 35%
 (в) чужа форма X дасть <×1.3                         шанс 75%
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
KMEL = 24
NSAMP = 1 << LOGN
CH = 1 << 15
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i].astype(np.float64)
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def relu_cov(m, C, K=KMEL):
    """(m_a, Cov_a) активацій при гаусовій h — ПОВНИЙ ряд Мелера."""
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    ma = s * (f + t * P)
    rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
    a1 = s * P
    cov = torch.outer(a1, a1) * rho
    Rn = rho.clone()
    Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
    fact = 2.0
    sf = s * f
    for n in range(2, K + 1):
        Rn = Rn * rho
        an = sf * Hm / fact
        cov = cov + torch.outer(an, an) * Rn * fact
        Hpre, Hm = Hm, (-t) * Hm - float(n - 2) * Hpre
        fact *= (n + 1)
    return ma, cov, s, P, f


def dcov3(s, P, f, Tt):
    """κ₃-поправка. `Tt` — БЕЗРОЗМІРНА форма T̃_jk = T_jk/(s_j²s_k);
    масштабується `s` ПОТОЧНОГО стану ⟹ переноситься форма, не істина."""
    T = Tt * (s * s).unsqueeze(1) * s.unsqueeze(0)
    A = (f / s)
    B = P
    return 0.5 * (torch.outer(A, B) * T + torch.outer(B, A) * T.T)


def truth_state(W, seed):
    """Істинні (m,C) щошару + БЕЗРОЗМІРНА форма T̃."""
    n = W[0].shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    mh, Ch, Tt = [], [], []
    X = None
    for L in range(len(W)):
        s1 = torch.zeros(n, dtype=torch.float64, device=DEV)
        buf = []
        for off in range(0, NSAMP, CH):
            b = min(CH, NSAMP - off)
            h = (torch.randn(b, n, generator=g, device=DEV) @ W[0]
                 if L == 0 else X[off:off + b] @ W[L])
            s1 += h.double().sum(0)
            buf.append(h)
        m = s1 / NSAMP
        mf = m.float()
        X = None
        s2 = torch.zeros(n, n, dtype=torch.float64, device=DEV)
        s3 = torch.zeros(n, n, dtype=torch.float64, device=DEV)
        for i, h in enumerate(buf):
            d = (h - mf).double()
            s2 += d.T @ d
            s3 += (d * d).T @ d
            buf[i] = torch.relu(h)
        C = s2 / NSAMP
        sd = C.diagonal().clamp_min(1e-30).sqrt()
        mh.append(m)
        Ch.append(C)
        Tt.append((s3 / NSAMP) / ((sd * sd).unsqueeze(1) * sd.unsqueeze(0)))
        X = torch.cat(buf, 0)
        del buf
    return mh, Ch, Tt


def dmean3(m, s, Tt, lam):
    """Еджворт-поправка до E[relu]: Δ = −κ₃·t·φ(t)/(6s²), κ₃ = T_jj·s³."""
    t = m / s
    k3 = Tt.diagonal() * s ** 3
    return -lam * k3 * t * phi(t) / (6.0 * s * s)


def run(W, Ch, Tt, mode, lam=1.0):
    """Рекурсія від шару 0. mode: 'A' | 'B' | 'C' | сторонній список T̃ для 'X'."""
    n = W[0].shape[0]
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = (W[0].double().T @ W[0].double())
    for L in range(1, 32):
        if mode == 'C':
            m, C = mh_true[L - 1], Ch[L - 1]
        ma, cov, s, P, f = relu_cov(m, C)
        if mode == 'B':
            cov = cov + lam * dcov3(s, P, f, Tt[L - 1])
        elif mode == 'D':
            cov = cov + lam * dcov3(s, P, f, Tt[L - 1])
            ma = ma + dmean3(m, s, Tt[L - 1], lam)
        elif mode == 'Dm':                      # ЛИШЕ поправка до m
            ma = ma + dmean3(m, s, Tt[L - 1], lam)
        elif isinstance(mode, list):
            cov = cov + lam * dcov3(s, P, f, mode[L - 1])
            ma = ma + dmean3(m, s, mode[L - 1], lam)
        Wd = W[L].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    return s * (phi(t) + t * Phi(t))


print(f"accum94 · {DEV} · {NNET} мереж · N=2^{LOGN}", flush=True)
t0 = time.time()
LAMS = (0.5, 0.625, 0.75, 0.875)
RES = {k: [] for k in ('A', 'B', 'C', 'D', 'Dm', 'X')}
RES.update({f'L{l}': [] for l in LAMS})
prevT = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        mh_true, Ch, Tt = truth_state(W, 970000 + 191 * q)
        tru = torch.as_tensor(fm, device=DEV)
        den = float(tru.abs().mean())
        for k in ('A', 'B', 'C', 'D', 'Dm'):
            out = run(W, Ch, Tt, k)
            RES[k].append(float((out - tru).abs().mean()) / den)
        for lam in LAMS:
            out = run(W, Ch, Tt, 'D', lam)
            RES[f'L{lam}'].append(float((out - tru).abs().mean()) / den)
        if prevT is not None:
            RES['X'].append(float((run(W, Ch, Tt, prevT, 0.75) - tru
                                   ).abs().mean()) / den)
        prevT = Tt
        print(f"  net{q}: A={RES['A'][-1]*100:.4f}%  B={RES['B'][-1]*100:.4f}%"
              f"  C={RES['C'][-1]*100:.4f}%  D={RES['D'][-1]*100:.4f}%"
              f" ({time.time()-t0:.0f} с)",
              flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 74)
print("НАКОПИЧЕНИЙ ПРОГІН · відносна похибка ВИХОДУ проти final_means (1e9)")
print("=" * 74)
a = np.array(RES['A']); b = np.array(RES['B']); c = np.array(RES['C'])
x = np.array(RES['X']) if RES['X'] else None
d = np.array(RES['D']); dm = np.array(RES['Dm'])
rows_ = [('A чиста рекурсія Мелера', a), ('B κ₃ лише в C', b),
         ('Dm κ₃ лише в m', dm), ('D κ₃ УЗГОДЖЕНО (m+C)', d),
         ('C ✅іст.C щошару (поз.контр)', c), ('X ⛔чужа форма T̃', x)]
rows_ += [(f'λ-скан D λ={l}', np.array(RES[f'L{l}'])) for l in LAMS]
for k, v in rows_:
    if v is None or not len(v):
        continue
    print(f"  {k:<30s} {v.mean()*100:>9.4f}%   MSE ×{(a.mean()/v.mean())**2:>7.2f}"
          f"   RMS ×{a.mean()/v.mean():>6.3f}")
print(f"\n🔒P1 поз.контроль C: ×{(a.mean()/c.mean())**2:.1f} у MSE "
      f"{'✅ (r538-D дав ×18.7)' if (a.mean()/c.mean())**2 > 5 else '🚨 стенд не той об єкт'}")
if x is not None and len(x):
    print(f"⛔P2 чужа: ×{(a.mean()/x.mean())**2:.2f} проти своєї D(λ=.75) "
          f"×{(a.mean()/np.array(RES['L0.75']).mean())**2:.2f} "
          f"{'✅ виграш є ІНФОРМАЦІЄЮ' if (a.mean()/np.array(RES['L0.75']).mean()) > 1.5*(a.mean()/x.mean()) else '🚨 виграш = ефект калібровки'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/accum94.npz",
         a=a, b=b, c=c, x=x if x is not None else np.zeros(0))
