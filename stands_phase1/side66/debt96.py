"""debt96 — БОРГ: не правити рекурсію, а ПРОНЕСТИ втрачене й ВІДДАТИ в кінці

════════════════ ІДЕЯ ЮЗЕРА, ЗАПИСАНА ФОРМУЛОЮ ════════════════
«дані при пронесенні і форма розподілу можуть бути як борг, який віддати варто»

Хай `ρ_ℓ` — істинний закон шару, `q_ℓ` — гаусів сурогат замикання,
`e_ℓ = ρ_ℓ − q_ℓ` — НЕПОГАШЕНЕ. Замикання робить `q_{ℓ+1} = M(T(q_ℓ))`, де
`T` — прохід крізь шар, `M` — проєкція на два моменти. Тоді ТОЧНО:
```
   e_{ℓ+1} = (I − M)·T(q_ℓ)   +   A_ℓ·e_ℓ
             ‾‾‾‾‾‾‾‾‾‾‾‾‾‾       ‾‾‾‾‾‾‾‾
             НОВА позика шару:    перенесення СТАРОГО боргу
             те, що проєкція викидає
   відповідь = ⟨ReLU, q_32⟩ + ⟨ReLU, e_32⟩
```
🔑 ДРУГИЙ ДОДАНОК ЛІНІЙНИЙ ЗА `e` ⟹ борг МОЖНА НЕ ВНОСИТИ В РЕКУРСІЮ.
   Несемо його окремим вектором і додаємо на виході. **Рівновага замикання
   не ламається** — а саме вона вбила сьогодні дві поправки:
   `[ЗМІРЯНО accum96b]` оракульна σ СОЛО в рекурсію = ×0.52 (ГІРШЕ бази),
   `[ЗМІРЯНО whest run]` ін'єкція в рекурсію сабміту = 2.10e-06 проти 1.93e-06.

🔑 ОПЕРАТОР ПЕРЕНЕСЕННЯ ВЖЕ ЗМІРЯНИЙ: `∂E[ReLU]/∂m = Φ(t)` ⟹ `A_ℓ` це рівно
   ВАЛ `diag(Φ_ℓ)·W_ℓ`, для якого `[ЦИТАТА finding-turbine-shaft]` R²=0.943.
   Ціна: ОДИН матвектор на шар = 1.31e5·32 = **4.2e6 флопів = 0.0015% B**.

════════════════ ДВА ДЖЕРЕЛА ПОЗИКИ (обидва — поправки до E[ReLU]) ═════════
```
 форма:  δγ_ℓ = s·(−(γ₃/6)·t·φ + (γ₄/24)·(t²−1)·φ)      ← що викидає гаусовість
 рівень: δσ_ℓ = φ(t)·(σ_іст − σ_замик)                   ← ∂E[ReLU]/∂σ = φ(t)
 несення: c ← (c @ W_{ℓ+1}) ⊙ Φ(t_{ℓ+1}) + δ_{ℓ+1}       ← вал
 віддача: відповідь = ma_31 + c
```

🔑🔑 І ГОЛОВНЕ МЕТОДИЧНО: борг АДИТИВНИЙ ⟹ контроль «ЧУЖА поправка» ВПЕРШЕ
   НЕ ПОРОЖНІЙ ЗА ПОБУДОВОЮ. Досі будь-яка невірна поправка в 31-кроковій
   рекурсії розвалювала прогін (похибка 34–150%), тож контроль «проходив»
   завжди і не казав НІЧОГО `[ЦИТАТА закон С89]`. Тут чужий борг просто додасть
   не той вектор — і різниця «своя vs чужа» стане ВИМІРНОЮ.

════════════════ ВАРІАНТИ ════════════════
 A    чистий Мелер                        🔒ПЛОМБА: 0.847121% на 24 мережах
 Dg   борг ФОРМИ (γ₃,γ₄ оракульні), лінійне несення валом
 Ds   борг РІВНЯ (σ оракульна)
 Dgs  обидва борги
 DW   борг обох, γ і σ з РОБОЧОЇ вибірки 12288  ← подавальний варіант
 F    ⛔ЧУЖИЙ борг (з попередньої мережі) — КОНТРОЛЬ, тепер НЕ порожній
 R    ⛔борг із ПЕРЕВЕРНУТИМ знаком — другий контроль (теж тепер живий)
 λ-скан по Dgs: своя поправка мусить мати оптимум при λ>0

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ, протокол п.6) ════════════════
 (а) Dg поб'є ін'єкцію в рекурсію (E ×1.34 з accum96b)         шанс 65%
 (б) Dgs дасть ×2…×10                                          шанс 40%
 (в) 🔑контроль F буде ВИМІРНО гіршим за Dgs, але НЕ розвалить
     прогін (похибка лишиться <2%)                             шанс 80%
 (г) λ-оптимум Dgs буде в 0.5…1.5                              шанс 60%
 (д) DW (з вибірки) втратить не більше 40% виграшу Dgs          шанс 50%
 (е) борг РІВНЯ (Ds) сильніший за борг ФОРМИ (Dg)              шанс 55%
     [бо ∂m/∂σ=φ(t) б'є в той самий канал `m`, вага ×22.6]
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


def run(W, SD, G3, G4, use_g=False, use_s=False, lam=1.0, sign=1.0):
    """Рекурсія НЕДОТОРКАНА. Борг несеться окремим вектором `c`."""
    dev = W[0].device
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = W[0].double().T @ W[0].double()
    c = torch.zeros(256, dtype=torch.float64, device=dev)
    for L in range(32):
        s = C.diagonal().clamp_min(1e-30).sqrt()
        t = m / s
        P, f = Phi(t), phi(t)
        ma = s * (P * t + f)
        # ── НОВА ПОЗИКА цього шару (поправка до E[ReLU], НЕ входить у стан) ──
        d = torch.zeros_like(ma)
        if use_g:
            d = d + s * (-(G3[L] / 6.0) * t * f
                         + (G4[L] / 24.0) * (t * t - 1.0) * f)
        if use_s:
            d = d + f * (SD[L] - s)
        # ── ПЕРЕНЕСЕННЯ старого боргу валом + додавання нової позики ──
        c = c * P + sign * lam * d
        # ── рекурсія замикання: БЕЗ ЖОДНИХ ЗМІН ──
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
        if L == 31:
            return ma + c                     # ← ВІДДАЧА БОРГУ
        Wd = W[L + 1].double()
        m = ma @ Wd
        c = c @ Wd                            # ← вал: борг через ваги…
        C = Wd.T @ (cov @ Wd)                 #   …а ⊙Φ додається на початку


print(f"debt96 · {DEV} · {NNET} мереж · оракул N=2^{LOGN}", flush=True)
t0 = time.time()
KEYS = ['A', 'Dg', 'Ds', 'Dgs', 'DW', 'F', 'R'] + [f'L{k}' for k in
                                                   (25, 50, 150, 200)]
ERR = {k: [] for k in KEYS}
prev = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        SDo, G3o, G4o = stats(W, 963000 + 157 * q, NSAMP, DEV)
        SDw, G3w, G4w = stats(W, 987000 + 173 * q, NWORK, DEV)

        def er(x):
            return float((x - tru).norm() / tru.norm())
        ERR['A'].append(er(run(W, SDo, G3o, G4o)))
        ERR['Dg'].append(er(run(W, SDo, G3o, G4o, use_g=True)))
        ERR['Ds'].append(er(run(W, SDo, G3o, G4o, use_s=True)))
        ERR['Dgs'].append(er(run(W, SDo, G3o, G4o, use_g=True, use_s=True)))
        ERR['DW'].append(er(run(W, SDw, G3w, G4w, use_g=True, use_s=True)))
        ERR['R'].append(er(run(W, SDo, G3o, G4o, use_g=True, use_s=True,
                               sign=-1.0)))
        for k in (25, 50, 150, 200):
            ERR[f'L{k}'].append(er(run(W, SDo, G3o, G4o, use_g=True,
                                       use_s=True, lam=k / 100.0)))
        if prev is not None:
            ERR['F'].append(er(run(W, prev[0], prev[1], prev[2],
                                   use_g=True, use_s=True)))
        prev = (SDo, G3o, G4o)
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: A={ERR['A'][-1]*100:.4f}% Dg={ERR['Dg'][-1]*100:.4f}%"
                  f" Ds={ERR['Ds'][-1]*100:.4f}% Dgs={ERR['Dgs'][-1]*100:.4f}%"
                  f" DW={ERR['DW'][-1]*100:.4f}% ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 80)
print("БОРГ: ПРОНЕСТИ ВТРАЧЕНЕ Й ВІДДАТИ НА ВИХОДІ (рекурсія НЕДОТОРКАНА)")
print("=" * 80)
mse = {k: float(np.mean(np.array(v) ** 2)) for k, v in ERR.items() if v}
V0 = 2.05e-6
names = {'A': '🔒чистий Мелер (пломба 0.847121%)',
         'Dg': 'борг ФОРМИ (γ₃,γ₄ оракульні)',
         'Ds': 'борг РІВНЯ (σ оракульна)',
         'Dgs': '🔥обидва борги — ГОЛОВНИЙ',
         'DW': '🟢обидва, з РОБОЧОЇ вибірки 12288 (подавальний)',
         'F': '⛔ЧУЖИЙ борг — контроль (тепер НЕ порожній!)',
         'R': '⛔ПЕРЕВЕРНУТИЙ знак — другий контроль',
         'L25': 'λ=0.25', 'L50': 'λ=0.50', 'L150': 'λ=1.50', 'L200': 'λ=2.00'}
print(f"{'вар':>5s}{'віднос.похибка':>16s}{'MSE ×A':>9s}{'бал (оцінка)':>15s}  опис")
for k in KEYS:
    if not ERR[k]:
        continue
    r = float(np.mean(ERR[k]))
    e2 = 0.9097 * r * r
    print(f"{k:>5s}{r*100:>15.4f}%{mse['A']/mse[k]:>9.2f}"
          f"{0.1 * V0 * e2 / (V0 + e2):>15.3e}  {names[k]}")
print("\n🔑 ЧИТАТИ: ключове — Dgs проти F. Це ПЕРШИЙ раз, коли контроль «чужа»")
print("   не порожній за побудовою, бо борг адитивний і нічого не розвалює.")
print("   Порівняння з ін'єкцією в рекурсію (accum96b): Dg проти E ×1.34,")
print("   Dgs проти ES ×113.94 (оракул) і ESW ×13.58 (вибірка).")
