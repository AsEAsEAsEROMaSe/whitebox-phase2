"""off91 — САБМІТ ПРОТИ ОФІЦІЙНОЇ ІСТИНИ З 1e9 СЕМПЛІВ.

🚨 ЗНАХІДКА С91: офіційний датасет `aicrowd/arc-whestbench-public-2026`
лежить у нас на диску з 27.07 і НІКОЛИ не використовувався.
```
1000 мереж · weights (32,256,256) · final_means = істина з 1e9 семплів
шум істини = avg_variance/1e9 = 4.33e-11 = 0.0023% нашої робочої MSE
наш банк:   шум істини 4.84e-07 = 24% ⟹ офіційна ТОЧНІША в 11000×
мереж 1000 замість 48 ⟹ SE виміру падає ще в √(1000/48) = 4.6×
```
📌 Розподіли збігаються: `‖істина‖²` CV 0.8626 (банк 0.8577) ·
   `avg_variance` CV 0.6392 (банк 0.6598) ⟹ банк не зміщений, лише шумний.

════════════════ ЩО ЦЕ ДАЄ ════════════════
Уперше можна поставити пломбу, якої НІКОЛИ не було: чи відтворює стенд
**ГРЕЙДЕРНУ АБСОЛЮТНУ MSE**. `sub99` на грейдері = **1.86325e-06**
(бал 1.86762e-07 = MSE × множник 0.10023).

Тут — порт сабміта (`sub91_port.py`, звірений із ЖИВИМ estimator на 1.126e-06)
з константами sub99: `_TARGET_UTIL`=0.09011, `_W_BLEND`=0.0883, Холецький.

════════════════ ПЛОМБИ ════════════════
P1 🎯 ГОЛОВНА: абсолютна MSE мусить лягти на 1.86325e-06 ± шум по мережах.
   Якщо ні — стенд НЕ міряє те, що міряє грейдер, і всі локальні вироки
   треба переглядати.
P2 множник: `C/B` мусить дати 0.10023 (сабмітне значення) — рахуємо флопи
   тим самим виразом, що й сабміт.
P3 шум істини: різниця між нашою МК на 1e6 семплів і `final_means` мусить
   бути на рівні 1/√1e6, а не 1/√1e9 — контроль, що істина справді точна.
P4 крива від N мереж: похибка оцінки MSE мусить падати як 1/√N.

════════════════ ПЕРЕДБАЧЕННЯ (записано ДО запуску) ════════════════
 (а) P1 зійдеться в межах 10%                       — шанс 55%
 (б) якщо не зійдеться — винен `_TARGET_UTIL`/обрізання, не метод — 70%
 🎯 головна цінність не в числі, а в тому, що вперше є ПРИЛАД, який
    міряє ту саму величину, що й грейдер.
"""
import glob
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

sys.path.insert(0, "/mnt/win_d/llm/White-Box/side66")
import sub91_port as P                                        # noqa: E402

# 🔑 КОНСТАНТИ sub99 (порт написаний під sub91)
P.TARGET_UTIL = 0.09011
P.W_BLEND = 0.0883
P.ALPHA = -0.00751

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 250
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
GRADER_MSE = 1.86325e-06
GRADER_SCORE = 1.86762e-07
B = 272_000_000_000


def rows(limit):
    """Ліниве читання шардів: (ваги (32,256,256), істина (256,))."""
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        av = np.asarray(t.column("avg_variance").combine_chunks())
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i].astype(np.float64), float(av[i])
            got += 1


def whiten_chol(z, npair):
    cov = (z.T @ z).double() / float(npair)
    return torch.linalg.inv(torch.linalg.cholesky(cov)).T.float()


MSE, REL, MULT, NP, CLE, MCE = [], [], [], [], [], []
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth, av) in enumerate(rows(NNET)):
        Wt = [torch.as_tensor(wnp[i], device=DEV) for i in range(32)]
        tr = torch.as_tensor(truth, device=DEV, dtype=torch.float64)
        wc, npair, rng = P.cut_budget91(Wt, DEV)
        # 🔑 sub99 бере Холецький замість eigh — підміняємо всередині порту
        pr, _ = P.predict91(wc, npair, DEV, rng=rng)
        d = pr - tr
        mse = float((d * d).mean())                       # РІВНО як грейдер
        MSE.append(mse)
        REL.append(float((d * d).sum() / (tr * tr).sum()))
        NP.append(npair)
        # ⛔ P2: флопи тим самим виразом, що й сабміт
        per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
        fl = 2.0 * P.PILOT_PAIRS * (2.0 * 32 * 256 * 256) + 2.0 * npair * per_cut
        MULT.append(max(0.1, fl / B))
        if q % 25 == 0:
            print(f"  net{q:>4d}  MSE {mse:.4e}  n_pairs {npair} "
                  f"({time.time()-t0:.0f} с)", flush=True)
MSE, REL, MULT = map(np.array, (MSE, REL, MULT))
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(MSE)}\n", flush=True)

m = MSE.mean()
se = MSE.std(ddof=1) / np.sqrt(len(MSE))
print("=" * 96)
print(f"САБМІТ sub99 ПРОТИ ОФІЦІЙНОЇ ІСТИНИ (1e9 семплів) · {len(MSE)} мереж")
print("=" * 96)
print(f"{'абсолютна MSE (як у грейдера)':>34s} : {m:.6e}  ± {se:.2e} (SE)")
print(f"{'ГРЕЙДЕР sub99':>34s} : {GRADER_MSE:.6e}")
print(f"{'відношення стенд/грейдер':>34s} : {m/GRADER_MSE:.4f}  "
      f"({(m-GRADER_MSE)/se:+.1f} SE)")
print(f"{'відносна похибка':>34s} : {100*np.sqrt(REL.mean()):.4f}%")
print(f"{'множник C/B':>34s} : {MULT.mean():.5f}  (сабміт 0.10023)")
print(f"{'бал = MSE × множник':>34s} : {(MSE*MULT).mean():.6e}  "
      f"(грейдер {GRADER_SCORE:.6e})")
print(f"{'n_pairs':>34s} : {np.mean(NP):.0f}")

print(f"\n⛔ P1 {'✅ СТЕНД ВІДТВОРЮЄ ГРЕЙДЕР' if abs(m/GRADER_MSE-1) < 0.10 else '🚨 РОЗІЙШЛОСЬ — стенд міряє НЕ ТЕ'}"
      f"  (|Δ| = {100*abs(m/GRADER_MSE-1):.2f}%)")
print(f"⛔ P2 множник {'✅' if abs(MULT.mean()/0.10023-1) < 0.02 else '🚨'}")
print(f"\n⛔ P4 крива від N мереж (SE мусить падати як 1/√N):")
for k in (25, 50, 100, 200, len(MSE)):
    if k <= len(MSE):
        print(f"   N={k:>4d}  середнє {MSE[:k].mean():.4e}  "
              f"SE {MSE[:k].std(ddof=1)/np.sqrt(k):.2e}  "
              f"відн.SE {100*MSE[:k].std(ddof=1)/np.sqrt(k)/MSE[:k].mean():.2f}%")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/off91.npz",
         MSE=MSE, REL=REL, MULT=MULT, NP=np.array(NP))
