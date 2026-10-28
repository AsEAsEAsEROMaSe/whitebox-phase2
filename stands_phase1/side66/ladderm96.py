"""ladderm96 — ДРАБИНА ПО СЕРЕДНЬОМУ: скільки точності в `m` купує скільки балу

════════════════ ЩО ЩОЙНО ЗМІРЯВ `ladder96` (4 мережі, N=2^18) ════════════════
```
чисте замикання                          0.7917%    ×1.00
оракульний `m` щошару, `C` — замикання   0.1591%   ×22.59   ← ВЕСЬ ВИГРАШ ТУТ
оракульні `m` І `C` (діаг+поза)          0.1478%   ×26.96   ← стеля класу
оракульний `m` + оракульна ЛИШЕ поза     0.2476%    ×9.49   ← ГІРШЕ: розсинхрон
шум оракула: m 0.0345% · позадіаг 0.332%
```
🔑 ⟹ ЦІЛЬ СЕСІЇ БУЛА НЕ ТА. Коли `m` точний, УСЯ коваріація (діагональ +
   позадіагональ разом) купує лише **×1.16**. Позадіагональ 4.08% — не
   вузьке місце, а НАСЛІДОК: вона псує `m` наступного шару через σ.
🔒 ДВІ НЕЗАЛЕЖНІ ПЛОМБИ на стелю 0.148%: (1) `newalg95c` зміряв похибку
   ОДНОГО кроку `E[relu]` з ТОЧНИМИ (μ,σ,ρ) = **0.139%**; (2) тут вона
   вийшла як накопичена межа = **0.148%**. Різні стенди, різні дані.
⟹ стеля класу «(m,C) + гаусів вихід» = 0.148%, а поріг «МК стає зайвою» =
   0.145% ⟹ клас лягає РІВНО на беззбитковість і виграти не може ЗА ПОБУДОВОЮ.

════════════════ ПИТАННЯ ЦЬОГО СТЕНДУ ════════════════
`m` — єдиний канал, що несе ×22.6. Наскільки ТОЧНО його треба знати?
```
   m ← (1−κ)·m_замикання + κ·m_істина      κ = 0 … 1
   похибка `m` = (1−κ) · (власна похибка замикання по `m`, друкується)
```
Це ПРЯМИЙ курс обміну. Будь-яка майбутня ідея (зокрема навчена заміна кроку
на 8.2e6 ТОЧНИХ цілях `all_layer_means` з 1000 мереж) оцінюється ОДНИМ
числом — своєю похибкою по `m` — без окремого прогону.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) крива буде ПЛАВНОЮ, без обриву (похибка ~ лінійна за (1−κ))   шанс 70%
 (б) для 0.145% (МК зайва) вистачить κ≈0.8, тобто `m` з точністю
     ~20% ВІД НИНІШНЬОЇ похибки замикання                          шанс 45%
 (в) полиця 0.148% буде досягнута вже при κ=0.95                   шанс 60%
 (г) ⛔ЧУЖИЙ `m` (з іншої мережі) при κ=1 розвалить прогін          шанс 85%
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
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NSAMP = 1 << LOGN
CH = 1 << 13
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24
KAP = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]


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


def true_means(W, seed, n, dev):
    """ІСТИННІ середні передактивації кожного шару + шум оцінки."""
    g = torch.Generator(device=dev).manual_seed(seed)
    X = torch.randn(n, 256, generator=g, device=dev)
    half = n // 2
    MS, NZ = [], []
    for L in range(32):
        Wd = W[L]
        s1 = torch.zeros(256, dtype=torch.float64, device=dev)
        s1h = torch.zeros(256, dtype=torch.float64, device=dev)
        for o in range(0, n, CH):
            a = (X[o:o + CH] @ Wd).double().sum(0)
            s1 += a
            if o < half:
                s1h += a
        m = s1 / n
        NZ.append(float((s1h / half - m).norm() / m.norm()) / 3 ** 0.5)
        MS.append(m)
        X = torch.relu(X @ Wd)
    return MS, NZ


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


def run(W, MS, kap, track=None):
    dev = W[0].device
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = W[0].double().T @ W[0].double()
    for L in range(32):
        if kap is not None and L > 0:
            if track is not None:
                track.append(float((m - MS[L]).norm() / MS[L].norm()))
            m = (1.0 - kap) * m + kap * MS[L]
        ma, cov = mehler(m, C)
        if L == 31:
            return ma
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


print(f"ladderm96 · {DEV} · {NNET} мереж · МК N=2^{LOGN}", flush=True)
t0 = time.time()
ERR = {f"k{int(k*1000)}": [] for k in KAP}
ERR['base'] = []
ERR['foreign'] = []
NZ, MERR = [], []
prev = None
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        MS, nz = true_means(W, 991000 + 179 * q, NSAMP, DEV)
        NZ.append(float(np.mean(nz[8:])))

        def er(x):
            return float((x - tru).norm() / tru.norm())
        tr = []
        ERR['base'].append(er(run(W, MS, None)))
        run(W, MS, 0.0, tr)                      # власна похибка `m` замикання
        MERR.append(float(np.mean(tr[8:])))
        for k in KAP:
            ERR[f"k{int(k*1000)}"].append(er(run(W, MS, k)))
        if prev is not None:
            ERR['foreign'].append(er(run(W, prev, 1.0)))
        prev = MS
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: база={ERR['base'][-1]*100:.4f}% "
                  f"κ=1 {ERR['k1000'][-1]*100:.4f}% "
                  f"похибка m замикання={MERR[-1]*100:.3f}% "
                  f"шум m={NZ[-1]*100:.4f}% ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("ДРАБИНА ПО `m` — ПРЯМИЙ КУРС ОБМІНУ «ТОЧНІСТЬ СЕРЕДНЬОГО → БАЛ»")
print("=" * 78)
nz = float(np.mean(NZ))
me = float(np.mean(MERR))
mb = float(np.mean(np.array(ERR['base']) ** 2))
print(f"власна похибка `m` у замиканні (шари 8..31): {me*100:.3f}%")
print(f"🚨 шум оракульного `m` (МК N=2^{LOGN}): {nz*100:.4f}% ⟹ рядки з "
      f"похибкою `m` нижче цього НЕ ЧИННІ")
print(f"\n{'κ':>6s}{'похибка m':>12s}{'накопичена':>13s}{'MSE ×база':>11s}"
      f"{'бал (оцінка)':>14s}")
for k in KAP:
    v = ERR[f"k{int(k*1000)}"]
    r = float(np.mean(v))
    m2 = float(np.mean(np.array(v) ** 2))
    e2 = 0.9097 * r * r
    V = 2.05e-6
    sc = 0.1 * V * e2 / (V + e2)
    mrr = (1 - k) * me
    flag = "  ⚠️шум" if mrr < nz else ""
    print(f"{k:>6.2f}{mrr*100:>11.3f}%{r*100:>12.4f}%{mb/m2:>11.2f}"
          f"{sc:>14.3e}{flag}")
if ERR['foreign']:
    r = float(np.mean(ERR['foreign']))
    print(f"\n⛔КОНТРОЛЬ ЧУЖИЙ `m` (κ=1, інша мережа): {r*100:.4f}%  "
          f"MSE ×{mb/float(np.mean(np.array(ERR['foreign'])**2)):.3f}")
print("\n🎯 ЧИТАТИ: у колонці «бал» шукати, де перетинається 1.868e-07")
print("   (наш рекорд) і наскільки нижче можна впасти. Колонка «похибка m» —")
print("   ТЗ для будь-якого майбутнього методу, що цілиться в середнє.")
