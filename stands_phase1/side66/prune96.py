"""prune96 — ОБРІЗАТИ ЗА ВНЕСКОМ, А НЕ ЗА «ЧИ ХОЧ РАЗ ВІДКРИВСЯ»

════════════════ ЧОМУ ЦЕ НАЙБІЛЬШИЙ ЛИШОК ════════════════
`MSE ≈ 0.911·V`, а `V ∝ 1/N` ⟹ **бал прямо пропорційний вартості семпла**.
Вартість квадратична за шириною: `2·Σ rows_i·cols_i`.
Сабміт обрізає нейрон, лише якщо той **жодного разу** не відкрився на 256
пілотних семплах. Зміряно непрямо: `per_sample_cut ≈ 3.0e6` проти 4.19e6
без обрізання ⟹ прибирається лише **~15% ширини**.

`[ЗМІРЯНО newalg95, С95]` частка `t>0` = 0.491 на КОЖНОМУ шарі, медіана
`|t|` = 3.50 ⟹ половина нейронів «вимкнені», але з них обрізаються лише ті,
що мають `Φ(t) < ~1/256`. Нейрони з `Φ(t) ∈ [1e-3, 0.05]` відкриваються
зрідка, коштують ПОВНУ ціну і несуть майже нуль.

⟹ ПИТАННЯ ЦЬОГО СТЕНДУ: яка частка нейронів несе 99.99% відповіді?
```
   внесок_j(шар ℓ) = |E[h_j]| · ‖перенесення j → вихід‖
   перенесення = вал diag(Φ)·W (той самий, що R²=0.943)
```
Якщо 60% нейронів несуть 99.99%, то обрізання решти дає ціну ×0.36 ⟹
**×2.8 семплів ⟹ V/2.8 ⟹ бал ×2.8**. Це більше, ніж усе, що ми пробували
за 12 сесій разом.

🚨 РИЗИК, НАЗВАНИЙ ДО ВИМІРУ: обрізання вносить ЗМІЩЕННЯ, а зміщення не
   усереднюється. Тому міряємо ОБИДВА: скільки ціни економимо І яке
   зміщення вносимо у ФІНАЛЬНУ відповідь (проти `final_means` 1e9).

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) розподіл внеску буде ДУЖЕ нерівним: 50% нейронів < 1e-4 внеску шанс 55%
 (б) обрізання до 70% ширини дасть зміщення < 0.05%               шанс 40%
 (в) чиста економія (з урахуванням зміщення) дасть бал < 1.5e-07  шанс 30%
 (г) 🔑 глибокі шари обрізаються сильніше за перші                шанс 65%
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

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
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
        fm = np.asarray(t.column("final_means").combine_chunks()
                        .flatten()).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i]
            got += 1


def run_masked(W, z, keep):
    """Прогін із занулянням нейронів поза `keep[ℓ]` (None = всі)."""
    x = torch.relu(z @ W[0])
    if keep is not None and keep[0] is not None:
        x = x * keep[0]
    for L in range(1, 32):
        p = x @ W[L]
        if L == 31:
            return torch.relu(p).double().mean(0)
        x = torch.relu(p)
        if keep is not None and keep[L] is not None:
            x = x * keep[L]
    return None


print(f"prune96 · {DEV} · {NNET} мереж · {NS} семплів", flush=True)
t0 = time.time()
FRACS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
BIAS = {f: [] for f in FRACS}
COST = {f: [] for f in FRACS}
ALIVE = []
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        g = torch.Generator(device=DEV).manual_seed(880000 + 71 * q)
        z = torch.randn(NS, 256, generator=g, device=DEV)

        # ── понейронні середні та ЧАСТКА ВІДКРИТТЯ по шарах ──
        mus, opens = [], []
        x = torch.relu(z @ W[0])
        mus.append(x.double().mean(0))
        opens.append((x > 0).double().mean(0))
        for L in range(1, 31):
            p = x @ W[L]
            x = torch.relu(p)
            mus.append(x.double().mean(0))
            opens.append((p > 0).double().mean(0))

        # ── внесок = |E[h_j]| · ‖перенесення до виходу‖ (вал) ──
        contrib = []
        for L in range(31):
            v = torch.zeros(256, 256, dtype=torch.float64, device=DEV)
            v = torch.eye(256, dtype=torch.float64, device=DEV)
            for j in range(L + 1, 32):
                v = v @ W[j].double()
                if j < 31:
                    v = v * opens[j].unsqueeze(0) if j < len(opens) else v
            contrib.append(mus[L].abs() * v.abs().sum(1))

        # ── скільки нейронів несуть яку частку ──
        if q == 0:
            for L in (0, 8, 16, 24, 30):
                c = contrib[L] / contrib[L].sum()
                srt = torch.sort(c, descending=True).values
                cs = torch.cumsum(srt, 0)
                k999 = int((cs < 0.999).sum()) + 1
                k9999 = int((cs < 0.9999).sum()) + 1
                print(f"  шар {L:2d}: 99.9% внеску несуть {k999:3d}/256 "
                      f"нейронів · 99.99% — {k9999:3d}/256 · "
                      f"частка відкриття медіана {float(opens[L].median()):.3f}",
                      flush=True)

        full = run_masked(W, z, None)
        for fr in FRACS:
            if fr >= 1.0:
                BIAS[fr].append(float((full - tru).norm() / tru.norm()))
                COST[fr].append(1.0)
                continue
            keep, cw = [], []
            for L in range(31):
                k = int(256 * fr)
                thr = torch.sort(contrib[L], descending=True).values[k - 1]
                mk = (contrib[L] >= thr).to(z.dtype)
                keep.append(mk)
                cw.append(float(mk.sum()))
            keep.append(None)
            pred = run_masked(W, z, keep)
            BIAS[fr].append(float((pred - tru).norm() / tru.norm()))
            widths = [256.0] + cw
            c = sum(widths[i] * (widths[i + 1] if i + 1 < len(widths) else 256.0)
                    for i in range(32))
            COST[fr].append(c / (32 * 256 * 256))
        if q % 4 == 3 or q == 0:
            print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 74)
print("ОБРІЗАННЯ ЗА ВНЕСКОМ: скільки коштує і скільки зміщує")
print("=" * 74)
print(f"{'частка':>8s}{'ціна×':>9s}{'семплів×':>10s}{'похибка':>11s}"
      f"{'бал (оцінка)':>15s}")
V0, E2 = 2.05e-6, 0.9097 * (0.0085 ** 2)
for fr in FRACS:
    c = float(np.mean(COST[fr]))
    b = float(np.mean(BIAS[fr]))
    V = V0 * c                       # більше семплів ⟹ V менша пропорційно
    bias2 = 0.9097 * b * b
    mse = V0 * c * E2 / (V0 * c + E2) + bias2
    print(f"{fr:>8.2f}{c:>9.3f}{1/c:>10.2f}{b*100:>10.4f}%"
          f"{0.1*mse:>15.4e}")
print("\n🔑 ЧИТАТИ: «похибка» при частці 1.00 — це ЧИСТИЙ шум МК на цьому N;")
print("   різниця з нею при меншій частці і є ЗМІЩЕННЯМ обрізання.")
print("   Бал рахується як 0.1·[V·E²/(V+E²) + зміщення²]; рекорд 1.86762e-07.")
