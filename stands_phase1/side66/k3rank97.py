"""k3rank97 — ЧИ ЗБЕРІГАЄ κ₃ НИЗЬКИЙ РАНГ КРІЗЬ ReLU. ЦЕ ВИРІШУЄ ГІЛКУ ЦІЛКОМ.

════════════════ ЧОМУ САМЕ ЦЕ ЧИСЛО ════════════════
Три незалежні виміри сьогодні зійшлись в одну точку:
```
[ЗМІРЯНО pairtrunc97, d9, 2 мережі, крос-фіт]  87% негаусовості ПАРИ несуть
    крос-кумулянти 3-го порядку `λ₂₁`; 4-й порядок НЕ ДОДАЄ (85.3/89.2%)
[ЦИТАТА С74, k3diag.py]  ранг 16 дає 83% оракульної інформації про κ₃
[ЦИТАТА CLAUDE.md]  ранг-r CP коштує 1.219e7·r ⟹ r=16 це 1.95e8 = 0.07% B
```
🔑 І ВЛАСТИВІСТЬ, ЯКОЇ МИ НЕ ВИКОРИСТАЛИ: під ЛІНІЙНИМ кроком ранг НЕ РОСТЕ.
```
   κ₃(h) = Σᵢ cᵢ aᵢ^⊗3   ⟹   κ₃(hW) = Σᵢ cᵢ (Wᵀaᵢ)^⊗3
```
Це `r` МАТВЕКТОРІВ (1.311e5 кожен), а не тризгортка тензора 256³.
⟹ нести третій кумулянт крізь рекурсію ДЕТЕРМІНОВАНО може коштувати
   16·1.311e5·32 = 6.7e7 = **0.025% бюджету**.

🚨 УСЕ ТРИМАЄТЬСЯ НА ОДНОМУ НЕЗМІРЯНОМУ ПРИПУЩЕННІ: **ReLU — нелінійний, і він
може РОЗДУВАТИ ранг.** Якщо після кожного шару ранг стрибає до 256, схема
мертва: доведеться щошару перепроєктовувати, а це вже повний тензор.

════════════════ ЩО МІРЯЄМО ════════════════
Ранг вимірюємо через РОЗГОРТКУ (mode-1 unfolding) тензора третіх центральних
моментів, БЕЗ побудови самого 256³ (він 16.7e6 чисел і повільний):
```
   M1[a, (b,c)] = E[ x̃_a x̃_b x̃_c ]  ,  x̃ = h − E[h]
   ⟹ M1 = (1/N) · X̃ᵀ · KR(X̃, X̃)      KR = построковий Кронекер
   ранг κ₃ ≥ ранг M1, і для симетричного тензора вони збігаються
```
Замість повного 256×65536 беремо СКЕТЧ: проєкцію на `S` випадкових зрізів
(`b,c` замінюємо на випадкові напрямки `v`), що дає незміщену оцінку спектра.
```
   M̂[a, s] = E[ x̃_a · (x̃·v_s)² ] − E[x̃_a]·E[(x̃·v_s)²]   (друге = 0, бо центровано)
```
🔒 ПЛОМБИ:
```
 (1) ГАУСІВ КОНТРОЛЬ: замінити h на гаусів вектор із тим самим Cov
     ⟹ κ₃ ≡ 0 ⟹ спектр мусить впасти до шумової підлоги
 (2) ШУМ: половини вибірки мусять дати той самий спектр
 (3) участь (participation ratio) на ШАРІ 31 мусить відтворити С74 (≈3.1)
```

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) 🔒гаусів контроль дасть спектр у ≥10× нижчий                 шанс 85%
 (б) PR(κ₃) на шарі 31 буде 2…6 (С74 каже 3.1)                    шанс 60%
 (в) 🔑 PR НЕ зросте монотонно з глибиною (ReLU не роздуває ранг)  шанс 45%
     ⟹ ЦЕ І Є ВИРОК ГІЛЦІ. Росте до >64 ⟹ мертво.
 (г) частка спектру в топ-16 буде >0.80 на глибоких шарах         шанс 50%
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NS = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 19)
CH = 1 << 13
NSK = 512                       # скетч-напрямків
SHOW = (1, 2, 4, 8, 16, 24, 31)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def spectrum(M):
    """сингулярні числа скетчу ⟹ PR і частка топ-k."""
    s = torch.linalg.svdvals(M.float()).double()
    s2 = s * s
    pr = float(s2.sum() ** 2 / (s2 * s2).sum())
    tot = float(s2.sum())
    f16 = float(s2[:16].sum() / tot) if tot > 0 else 0.0
    return pr, f16, tot


print(f"k3rank97 · {DEV} · {NNET} мереж · {NS} семплів · скетч {NSK}",
      flush=True)
t0 = time.time()
PR = {L: [] for L in SHOW}
F16 = {L: [] for L in SHOW}
PRG = {L: [] for L in SHOW}
RAT = {L: [] for L in SHOW}

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
            gv = torch.Generator(device=DEV).manual_seed(4400 + got)
            V = torch.randn(256, NSK, generator=gv, device=DEV)
            V = V / V.norm(dim=0, keepdim=True)

            # ── прохід 1: середні активацій на потрібних шарах ──
            S1 = {L: torch.zeros(256, dtype=torch.float64, device=DEV)
                  for L in SHOW}
            S2 = {L: torch.zeros(256, 256, dtype=torch.float64, device=DEV)
                  for L in SHOW}
            for o in range(0, NS, CH):
                g = torch.Generator(device=DEV).manual_seed(4400 + got + o)
                z = torch.randn(CH, 256, generator=g, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 32):
                    if L in SHOW:
                        hd = h.double()
                        S1[L] += hd.sum(0)
                        S2[L] += hd.T @ hd
                    h = torch.relu(h @ W[L])
            MU = {L: S1[L] / NS for L in SHOW}
            CV = {L: S2[L] / NS - torch.outer(MU[L], MU[L]) for L in SHOW}

            # ── прохід 2: скетч κ₃ + ГАУСІВ КОНТРОЛЬ ──
            A = {L: torch.zeros(256, NSK, dtype=torch.float64, device=DEV)
                 for L in SHOW}
            AG = {L: torch.zeros(256, NSK, dtype=torch.float64, device=DEV)
                  for L in SHOW}
            Lc = {L: torch.linalg.cholesky(
                CV[L] + 1e-8 * torch.eye(256, dtype=torch.float64, device=DEV))
                for L in SHOW}
            for o in range(0, NS, CH):
                g = torch.Generator(device=DEV).manual_seed(4400 + got + o)
                z = torch.randn(CH, 256, generator=g, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 32):
                    if L in SHOW:
                        xt = h.double() - MU[L]
                        q = (xt @ V.double()) ** 2
                        A[L] += xt.T @ q
                        gg = torch.Generator(device=DEV).manual_seed(
                            77000 + got + o + L)
                        e = torch.randn(CH, 256, generator=gg, device=DEV,
                                        dtype=torch.float64)
                        xg = e @ Lc[L].T
                        qg = (xg @ V.double()) ** 2
                        AG[L] += xg.T @ qg
                    h = torch.relu(h @ W[L])
            for L in SHOW:
                pr, f16, tot = spectrum(A[L] / NS)
                prg, _, totg = spectrum(AG[L] / NS)
                PR[L].append(pr)
                F16[L].append(f16)
                PRG[L].append(prg)
                RAT[L].append(tot / max(totg, 1e-300))
            got += 1
            print(f"  net{got-1}  PR@31={PR[31][-1]:.2f}  "
                  f"гаус={PRG[31][-1]:.2f}  ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 70)
print("РАНГ κ₃ ПО ГЛИБИНІ — ЧИ РОЗДУВАЄ ЙОГО ReLU")
print("=" * 70)
print(f"{'шар':>5s}{'PR(κ₃)':>10s}{'частка топ-16':>16s}"
      f"{'🔒PR гаусів':>14s}{'енергія/гаус':>15s}")
for L in SHOW:
    print(f"{L:>5d}{np.mean(PR[L]):>10.2f}{np.mean(F16[L]):>15.4f}"
          f"{np.mean(PRG[L]):>14.2f}{np.mean(RAT[L]):>15.1f}")
print("\n🔑 ВИРОК:")
print("  PR НЕ росте з глибиною і лишається < 32 ⟹ ReLU НЕ роздуває ранг,")
print("     ранг-r несення κ₃ ЖИВЕ, ціна 1.219e7·r (r=16 ⟹ 0.07% бюджету)")
print("  PR росте до > 64 ⟹ щошару потрібна перепроєкція ⟹ МЕРТВО")
print("  🔒 'енергія/гаус' ≫ 1 ⟹ κ₃ справжній, не шум оцінки")
print("  🔒 С74 каже PR ≈ 3.1 на шарі 31 — відтворення пломбує прилад")
