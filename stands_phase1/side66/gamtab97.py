"""gamtab97 — ЧИ МОЖНА ЗАШИТИ `γ₄(ℓ)` ТАБЛИЧКОЮ. ЦЕ ПЕРЕВОДИТЬ ІДЕЮ З
ВИБІРКОВИХ У ДЕТЕРМІНОВАНІ, А ЗНАЧИТЬ МІНЯЄ ЇЙ ПЛАНКУ 0.1454% → 0.5070%.

════════════════ ЗВІДКИ ПИТАННЯ ════════════════
`[ЗМІРЯНО gamiso97, 6 мереж, 131072 семпли]` негаусовість активацій
**ІЗОТРОПНА**: випадковий напрямок дає ті самі `|γ₃|`, що й реальні нейрони
(0.3970 проти 0.4024 на шарі 31), а ПІДПИСАНЕ середнє ≈0 (0.0103 / 0.0050).
`[ЗМІРЯНО saddlechk96, 49152 нейрони]` понейронно: `γ₃` медіана **−0.0016**
(5–95% ±0.51), `γ₄` медіана **+0.2791**.

🔑 ⟹ РОЗЩЕПЛЕННЯ, ЯКОГО НЕ БУЛО В С96:
```
   γ₃ понейронно ЗНАКОЗМІННИЙ із середнім ≈0 ⟹ гаусове припущення по 3-му
      порядку НЕ ЗМІЩЕНЕ; вибіркова оцінка `g3` несе САМ ШУМ
   γ₄ понейронно СИСТЕМАТИЧНО ДОДАТНИЙ    ⟹ це ЗМІЩЕННЯ, і його можна
      зашити КОНСТАНТОЮ НА ШАР — детерміновано, ціна НУЛЬ
```
С96 міряла ін'єкцію `(sh, g3, g4)` РАЗОМ і всі три З ВИБІРКИ ⟹ вирок
«вибіркова, планка 0.1454%, марна». Якщо `γ₄` — табличний, вирок не діє.

════════════════ ЩО МІРЯЄМО ════════════════
```
1. γ₄(ℓ) понейронно: МЕДІАНА по нейронах, і РОЗКИД ЦІЄЇ МЕДІАНИ МІЖ МЕРЕЖАМИ
      мала дисперсія між мережами ⟹ табличка ПРАЦЮЄ
2. γ₃(ℓ) те саме — пломба: медіана мусить бути ≈0 на КОЖНОМУ шарі
3. чи залежить γ₄ від `t_j` (кореляція) ⟹ якщо так, табличка як функція t
      краща за константу; якщо ні — константи досить
4. частка нейронів з γ₄ ≤ 0 — вона ж блокер седлової (зміряно 0.96%)
```

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) медіана γ₃(ℓ) буде |·| < 0.03 на КОЖНОМУ шарі                шанс 80%
 (б) медіана γ₄(ℓ) монотонно росте з ℓ і насичується ≈0.28        шанс 65%
 (в) розкид медіани γ₄ МІЖ мережами < 5% від самої медіани        шанс 60%
     ⟹ якщо так, табличка законна і ідея стає ДЕТЕРМІНОВАНОЮ
 (г) corr(γ₄, t) по нейронах буде |·| < 0.25                      шанс 55%
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NS = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 18)
CH = 1 << 14
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")

print(f"gamtab97 · {DEV} · {NNET} мереж · {NS} семплів", flush=True)
t0 = time.time()
M3 = np.zeros((NNET, 32))          # медіана γ₃ по нейронах, на мережу/шар
M4 = np.zeros((NNET, 32))
NEG = np.zeros((NNET, 32))         # частка γ₄ ≤ 0
CT = np.zeros((NNET, 32))          # corr(γ₄, t)
ALL4 = [[] for _ in range(32)]

with torch.no_grad():
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= NNET:
            break
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        for i in range(t.num_rows):
            if got >= NNET:
                break
            W = [torch.as_tensor(np.ascontiguousarray(w[i][L]), device=DEV)
                 for L in range(32)]
            S = [torch.zeros(4, 256, dtype=torch.float64, device=DEV)
                 for _ in range(32)]
            # ── прохід 1: μ, σ передактивацій ──
            for o in range(0, NS, CH):
                zg = torch.Generator(device=DEV).manual_seed(
                    970000 + 31 * got + o)
                z = torch.randn(CH, 256, generator=zg, device=DEV)
                h = z
                for L in range(32):
                    p = (h @ W[L])
                    pd = p.double()
                    S[L][0] += pd.sum(0)
                    S[L][1] += (pd * pd).sum(0)
                    h = torch.relu(p)
            MU = [S[L][0] / NS for L in range(32)]
            SD = [((S[L][1] / NS - MU[L] * MU[L]).clamp_min(1e-300)).sqrt()
                  for L in range(32)]
            # ── прохід 2: центровані 3-й і 4-й моменти ──
            for o in range(0, NS, CH):
                zg = torch.Generator(device=DEV).manual_seed(
                    970000 + 31 * got + o)
                z = torch.randn(CH, 256, generator=zg, device=DEV)
                h = z
                for L in range(32):
                    p = (h @ W[L])
                    c = (p.double() - MU[L]) / SD[L]
                    c2 = c * c
                    S[L][2] += (c2 * c).sum(0)
                    S[L][3] += (c2 * c2).sum(0)
                    h = torch.relu(p)
            for L in range(32):
                g3 = (S[L][2] / NS).cpu().numpy()
                g4 = (S[L][3] / NS - 3.0).cpu().numpy()
                tt = (MU[L] / SD[L]).cpu().numpy()
                M3[got, L] = float(np.median(g3))
                M4[got, L] = float(np.median(g4))
                NEG[got, L] = float((g4 <= 0).mean())
                CT[got, L] = float(np.corrcoef(g4, tt)[0, 1])
                ALL4[L].append(g4)
            got += 1
            print(f"  net{got-1} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 76)
print("ПОНЕЙРОННІ γ ПЕРЕДАКТИВАЦІЙ — ЧИ ЗАКОННА ТАБЛИЧКА НА ШАР")
print("=" * 76)
print(f"{'шар':>4s}{'med γ₃':>10s}{'σ між мер.':>12s}"
      f"{'med γ₄':>10s}{'σ між мер.':>12s}{'розкид%':>10s}"
      f"{'γ₄≤0':>8s}{'corr(γ₄,t)':>12s}")
for L in range(32):
    r = 100.0 * M4[:, L].std() / max(abs(M4[:, L].mean()), 1e-12)
    print(f"{L:>4d}{M3[:, L].mean():>10.4f}{M3[:, L].std():>12.4f}"
          f"{M4[:, L].mean():>10.4f}{M4[:, L].std():>12.4f}{r:>9.2f}%"
          f"{100*NEG[:, L].mean():>7.2f}%{CT[:, L].mean():>12.4f}")

print("\n" + "=" * 76)
print("🔑 ВИРОК:")
print("  розкид% < 5 на глибоких шарах ⟹ ТАБЛИЧКА ЗАКОННА, ідея стає")
print("     ДЕТЕРМІНОВАНОЮ, планка падає 0.1454% → 0.5070%")
print("  med γ₃ ≈ 0 на кожному шарі ⟹ вибіркова `g3` несла САМ ШУМ")
print("\nТАБЛИЧКА ДЛЯ ЕСТИМАТОРА (медіана γ₄ по шарах 1..31):")
print("_G4TAB = (" + ", ".join(f"{M4[:, L].mean():.4f}"
                              for L in range(1, 32)) + ")")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/gamtab97.npz",
         M3=M3, M4=M4, NEG=NEG, CT=CT)
