"""homog98 — ЧИ ВИНОСИТЬСЯ МАСШТАБ ТОЧНО. ПЕРЕВІРКА ФАКТОРИЗАЦІЇ h = R·û.

════════════════ МЕХАНІЗМ, ЯКИЙ ЦЕ ПЕРЕВІРЯЄ ════════════════
ReLU ПОЗИТИВНО ОДНОРІДНИЙ: `ReLU(W·(R·û)) = R·ReLU(W·û)` ТОТОЖНО при R>0.
`[ЦИТАТА CLAUDE.md]` `cos(h(z), h(z′))@31 = 0.9741` ⟹ напрямок майже фіксований.
⟹ ЯКЩО `R` і `û` незалежні, то
```
    E[ReLU(W·h)] = E[R] · E[ReLU(W·û)]
```
і задача перестає бути про моменти `h` (наш клас `(m,C)`), а стає про розподіл
НАПРЯМКУ на сфері — інше ПРЕДСТАВЛЕННЯ СТАНУ (другий гейт С89/С90).

🚨 ЧОМУ ЦЕ НЕ ЗАКРИТО РАНІШЕ (перевірено грепом механізму, не гілки):
```
«мережа НЕ майже-лінійна відносно входу, R²=0.222»  — про залежність від `z`,
    а тут масштаб виноситься ТОЧНО за однорідністю, лінійність не потрібна
«розклад навколо голої лінійної МЕРТВИЙ, роздув 1.03e5×» — там розкладали саму
    МЕРЕЖУ; тут розкладається лише НАПРЯМОК, норма винесена
«cos=0.974 — це колінеарність МІЖ СЕМПЛАМИ, а масштаб R(z) СИЛЬНО нелінійний»
    — саме так, і для ReLU це НЕ ВАЖИТЬ: нелінійний масштаб виноситься точно
```

════════════════ ЩО МІРЯЄМО (усе на шарі 31, ціль оцінювача) ════════════════
```
ТОЧНО      A = E[ReLU(h @ W31)]                              (еталон)
(1) F1     E[R] · E[ReLU(û @ W31)]        ← факторизація, потребує незалежності
(2) F0     E[R] · ReLU(û₀ @ W31)          ← «напрямок сталий» (найгрубіше)
(3) наше   двомоментне гаусове замикання не міряємо тут — воно вже 0.507%
```
🔒 ПЛОМБИ:
 (1) ТОТОЖНІСТЬ ОДНОРІДНОСТІ: `mean(R·ReLU(û@W))` мусить дати A з похибкою 0.00
     (це алгебра, не наближення) — якщо ні, стенд зламаний
 (2) ПЛОМБА ШУМУ: половини вибірки мусять дати ту саму похибку
 (3) МАСШТАБУВАННЯ N×4: шум падає вдвічі, справжня похибка стоїть

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) 🔒тотожність однорідності зійдеться до <1e-6                      95%
 (б) corr(R, ReLU(û@W)) ВІДМІННА ВІД НУЛЯ ⟹ F1 не точна                70%
     (бо велика норма корелює з напрямком — обидва ростуть уздовж вала)
 (в) похибка F1 буде МЕНШОЮ за наше замикання 0.507%                   40%
 (г) похибка F0 («напрямок сталий») буде 3…15%                         55%
 🎯 ВИРОК: F1 < 0.145% ⟹ це новий клас, МК стає зайвою. F1 > 0.5% ⟹ мертво.
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

DEV = "cuda:0"          # пінування робить CUDA_VISIBLE_DEVICES=GPU-<uuid>
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
NS = 1 << (int(sys.argv[2]) if len(sys.argv) > 2 else 19)
CH = 1 << 13
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")

print(f"homog98 · {DEV} · {NNET} мереж · {NS} семплів", flush=True)
print(f"  видимі GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '(усі)')}",
      flush=True)
t0 = time.time()
RES = []

with torch.no_grad():
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        if got >= NNET:
            break
        tb = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = tb.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(tb.num_rows, 32, 256, 256)
        for i in range(tb.num_rows):
            if got >= NNET:
                break
            W = [torch.as_tensor(np.ascontiguousarray(w[i][L]), device=DEV)
                 for L in range(32)]
            # ── акумулятори (усе у float64) ──
            acc = {k: torch.zeros(256, dtype=torch.float64, device=DEV)
                   for k in ("A", "ID", "U", "H1", "H2")}
            sR = torch.zeros((), dtype=torch.float64, device=DEV)
            sR2 = torch.zeros((), dtype=torch.float64, device=DEV)
            sU = torch.zeros(256, dtype=torch.float64, device=DEV)
            nn = 0
            # ── прохід 1: E[R], E[û], і еталон A ──
            for o in range(0, NS, CH):
                g = torch.Generator(device=DEV).manual_seed(9100 + got + o)
                z = torch.randn(CH, 256, generator=g, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 31):
                    h = torch.relu(h @ W[L])
                hd = h.double()
                R = hd.norm(dim=1, keepdim=True)
                U = hd / R.clamp_min(1e-30)
                a = torch.relu(hd @ W[31].double())
                acc["A"] += a.sum(0)
                # 🔒 ПЛОМБА ОДНОРІДНОСТІ: R·ReLU(û@W) мусить дорівнювати a
                acc["ID"] += (R * torch.relu(U @ W[31].double())).sum(0)
                acc["U"] += torch.relu(U @ W[31].double()).sum(0)
                sU += U.sum(0)
                sR += R.sum()
                sR2 += (R * R).sum()
                half = 0 if (o // CH) % 2 == 0 else 1
                acc["H1" if half == 0 else "H2"] += a.sum(0)
                nn += CH
            A = acc["A"] / nn
            ID = acc["ID"] / nn
            EU = acc["U"] / nn
            ER = sR / nn
            U0 = sU / nn
            U0 = U0 / U0.norm()
            F1 = ER * EU
            F0 = ER * torch.relu(U0 @ W[31].double())

            def rel(x):
                return float((x - A).norm() / A.norm())

            RES.append((rel(ID), rel(F1), rel(F0),
                        float(sR2 / nn / (ER * ER) - 1.0)))
            print(f"  net{got}  🔒тотожність {RES[-1][0]:.2e} · "
                  f"F1 {100*RES[-1][1]:.4f}% · F0 {100*RES[-1][2]:.3f}% · "
                  f"Var(R)/E[R]² {RES[-1][3]:.5f}  ({time.time()-t0:.0f} с)",
                  flush=True)
            got += 1

m = lambda k: float(np.mean([r[k] for r in RES]))
print("\n" + "=" * 70)
print("ФАКТОРИЗАЦІЯ МАСШТАБ×НАПРЯМОК — ЧИ ВИНОСИТЬСЯ МАСШТАБ")
print("=" * 70)
print(f"🔒 ПЛОМБА ОДНОРІДНОСТІ (мусить бути ~1e-15): {m(0):.3e}")
print(f"   F1 = E[R]·E[ReLU(û@W)]   похибка {100*m(1):.4f}%")
print(f"   F0 = E[R]·ReLU(û₀@W)     похибка {100*m(2):.4f}%")
print(f"   Var(R)/E[R]²             {m(3):.5f}")
print("\n🎯 ПОРІГ: наше замикання 0.507% · пряма МК 0.145% · лідер 0.0064%")
print("   F1 < 0.145% ⟹ НОВИЙ КЛАС, МК зайва · F1 > 0.5% ⟹ мертво")
