"""ladder96 — ДРАБИНА ПО ПОЗАДІАГОНАЛІ: скільки точності КУПУЄ скільки балу

════════════════ ЧОМУ ЦЕ, А НЕ ЧЕРГОВА ПОПРАВКА ════════════════
За 10 сесій усі поправки дали ×1.02…×1.39, а потрібно ×34 у MSE. Замість
шукати ще одну, МІРЯЄМО ФУНКЦІЮ ПЕРЕДАЧІ: якщо позадіагональ `Cov` знати з
точністю ε, яка вийде НАКОПИЧЕНА похибка? Тоді БУДЬ-ЯКА майбутня ідея
оцінюється ОДНИМ числом (її ε) без окремого прогону.

`[ЗМІРЯНО newalg95c]` крок із ТОЧНИМИ (μ,σ,ρ): позадіаг **4.08%**, діаг 1.03%
`[ЦИТАТА finding-closure-channels-cancel]` драбина оракулів С88, 40–60 мереж:
    тільки m ×0.912 · тільки діаг ×0.243 · тільки поза ×0.634 · m+C ×20.97
🔑 ⟹ ОКРЕМІ канали НЕ ІНФОРМАТИВНІ (взаємно скорочуються). Тому тут `m` і
   діагональ ТРИМАЄМО ОРАКУЛЬНИМИ, а свіпаємо ЛИШЕ позадіагональ:
```
   C_off ← (1−κ)·C_off^замик + κ·C_off^істина      κ = 0 … 1
   похибка позадіагоналі = (1−κ)·4.08%
```
κ=1 мусить відтворити ВІДОМЕ число `m+C ×20.97` ⟹ **0.185%** — це ПЛОМБА
приладу на першоджерелі, а не на власній алгебрі.

════════════════ ЩО САМЕ ЦЕ ВИРІШУЄ ════════════════
Якщо E(κ) виходить на полицю вже при κ=0.5 — позадіагональ ДЕШЕВА, і вузьке
місце деінде. Якщо E(κ) падає аж до κ=1 — маємо ПРЯМИЙ курс обміну
«точність позадіагоналі → бал», і одразу видно, чи існує ідея, здатна дати
потрібне (ціль: 0.145% щоб МК стала зайвою, 0.0064% щоб бути лідером).

🚨 ШУМОВА ПІДЛОГА: істинна `Cov` береться з МК на N семплів ⟹ сама має шум
~1/√N на елемент. Тому стенд рахує ДВІ НЕЗАЛЕЖНІ ПОЛОВИНИ вибірки і друкує
розбіжність між ними: якщо вона ПОРІВНЯННА з (1−κ)·4.08%, точка κ НЕ ЧИННА.
*(С92: три опорні числа проєкту виявились артефактами шумової підлоги.)*

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) κ=1 відтворить 0.15…0.25% (пломба проти ×20.97)            шанс 60%
 (б) E(κ) буде опуклою: половина виграшу вже при κ=0.5          шанс 50%
 (в) шумова підлога зробить κ≥0.9 НЕЧИННИМИ при N=2^15          шанс 65%
 (г) з нахилу вийде, що для 0.145% треба позадіаг ≤0.5%         шанс 45%
     (тобто ×8 від нинішніх 4.08% — а суміш С96 дала лише ×2.73)
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 15
NSAMP = 1 << LOGN
CH = 1 << 13
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24
KAPPA = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]


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


def true_state(W, seed, n, dev):
    """ІСТИННІ (m, Cov) передактивації кожного шару + оцінка шуму по半."""
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 256, generator=g, device=dev)
    half = n // 2
    MS, CS, NOISE = [], [], []
    for L in range(32):
        Wd = W[L]
        s1 = torch.zeros(256, dtype=torch.float64, device=dev)
        G = torch.zeros(256, 256, dtype=torch.float64, device=dev)
        s1h = torch.zeros(256, dtype=torch.float64, device=dev)
        Gh = torch.zeros(256, 256, dtype=torch.float64, device=dev)
        for o in range(0, n, CH):
            pb = X[o:o + CH] @ Wd
            a = pb.double().sum(0)
            b = (pb.T @ pb).double()
            s1 += a
            G += b
            if o < half:
                s1h += a
                Gh += b
        m = s1 / n
        C = G / n - torch.outer(m, m)
        mh = s1h / half
        Ch = Gh / half - torch.outer(mh, mh)
        off = ~torch.eye(256, dtype=torch.bool, device=dev)
        # розбіжність половини від цілого ×1/√3 ≈ шум ПОВНОЇ оцінки
        # 🚨 ДРУГИЙ рядок — шум САМОГО `m`: якщо він порівнянний з виміряною
        # накопиченою похибкою, уся точка κ — артефакт (пастка С92).
        NOISE.append((float((Ch - C)[off].norm() / C[off].norm()) / 3 ** 0.5,
                      float((mh - m).norm() / m.norm()) / 3 ** 0.5))
        MS.append(m)
        CS.append(C)
        X = torch.relu(X @ Wd)
    return MS, CS, NOISE


def mehler(m, C):
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    ma = s * (P * t + f)
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
    return ma, cov


def run(W, MS, CS, kap, oracle_m=True, oracle_diag=False):
    """kap=None ⟹ чисте замикання без жодних ін'єкцій.

    🚨 ДІАГОНАЛЬ НЕ ІН'ЄКТУЄМО (oracle_diag=False) — інакше позадіагональ
    ВЗАГАЛІ НЕ МАЄ ШЛЯХУ ДО ВИХОДУ: вихід залежить лише від `m` і `σ`, а
    `σ` наступного шару = diag(WᵀCW) і саме ТУДИ входить позадіагональ.
    Перша версія стенду ін'єктувала обидва — і всі κ дали ОДНЕ число.
    """
    dev = W[0].device
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = W[0].double().T @ W[0].double()
    eye = torch.eye(256, dtype=torch.bool, device=dev)
    for L in range(32):
        if kap is not None:
            Ct = CS[L]
            if oracle_m:
                m = MS[L]
            C = torch.where(eye, C, (1.0 - kap) * C + kap * Ct)
            if oracle_diag:
                C = C.clone()
                C.diagonal().copy_(Ct.diagonal())
        ma, cov = mehler(m, C)
        if L == 31:
            return ma
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


print(f"ladder96 · {DEV} · {NNET} мереж · МК N=2^{LOGN}", flush=True)
t0 = time.time()
ERR = {k: [] for k in ['base'] + [f"k{int(k*100)}" for k in KAPPA]
       + ['mo', 'do']}
NF = []
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        MS, CS, noise = true_state(W, 996000 + 173 * q, NSAMP, DEV)
        NF.append((float(np.mean([x[0] for x in noise[8:]])),
                   float(np.mean([x[1] for x in noise[8:]]))))

        def er(x):
            return float((x - tru).norm() / tru.norm())
        ERR['base'].append(er(run(W, MS, CS, None)))
        for k in KAPPA:
            ERR[f"k{int(k*100)}"].append(er(run(W, MS, CS, k)))
        ERR['mo'].append(er(run(W, MS, CS, 1.0, True, True)))
        ERR['do'].append(er(run(W, MS, CS, 0.0, False, False)))
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: база={ERR['base'][-1]*100:.4f}% "
                  f"κ=0 {ERR['k0'][-1]*100:.4f}% κ=1 {ERR['k100'][-1]*100:.4f}%"
                  f" шум: поза={NF[-1][0]*100:.3f}% m={NF[-1][1]*100:.4f}%"
                  f" ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("ДРАБИНА ПО ПОЗАДІАГОНАЛІ (m і діагональ — ОРАКУЛЬНІ)")
print("=" * 78)
nf = float(np.mean([x[0] for x in NF]))
nfm = float(np.mean([x[1] for x in NF]))
base = float(np.mean(ERR['base']))
mb = float(np.mean(np.array(ERR['base']) ** 2))
print(f"🚨 ШУМОВА ПІДЛОГА (МК N=2^{LOGN}): позадіагональ {nf*100:.3f}%  ·  "
      f"САМ ОРАКУЛЬНИЙ m {nfm*100:.4f}%")
print("   точки з (1−κ)·4.08% нижче першої — НЕ ЧИННІ; уся таблиця НЕ ЧИННА,")
print("   якщо накопичена похибка впала до рівня другої.")
print(f"\n{'κ':>6s}{'похибка поза':>15s}{'накопичена':>13s}{'MSE ×база':>11s}")
for k in KAPPA:
    v = ERR[f"k{int(k*100)}"]
    r = float(np.mean(v))
    m2 = float(np.mean(np.array(v) ** 2))
    flag = "  ⚠️шум" if (1 - k) * 4.08 < nf * 100 else ""
    print(f"{k:>6.2f}{(1-k)*4.08:>14.3f}%{r*100:>12.4f}%{mb/m2:>11.2f}{flag}")
print(f"\n{'чисте замикання (жодних ін’єкцій)':<38s}{base*100:>9.4f}%   ×1.00")
for k, nm in (("mo", "🔒ПЛОМБА: m+C ПОВНІСТЮ оракульні (ціль ≈0.185%)"),
              ("do", "⛔лише позадіаг оракульна, m замикання")):
    r = float(np.mean(ERR[k]))
    m2 = float(np.mean(np.array(ERR[k]) ** 2))
    print(f"{nm:<38s}{r*100:>9.4f}%   ×{mb/m2:.2f}")
print("\n🔒 ПЛОМБА: κ=1 (усе оракульне) мусить дати ≈0.185% — це відоме")
print("   число `m+C ×20.97` із С88 на 60 мережах. Розбіжність ⟹ прилад бреше.")
print("🎯 ЧИТАТИ НАХИЛ: скільки % позадіагоналі треба для 0.145% (МК зайва)")
print("   і для 0.0064% (рівень лідера).")
