"""k3sweep97 — СКІЛЬКИ ПОХИБКИ ЗНІМАЄ РАНГ-r `κ₃` ПІДСТАНОВКОЮ, А НЕ R².

🚨 ЧОМУ ЦЕЙ СТЕНД ІСНУЄ (АРХІВ зупинила гілку, і по суті):
```
[ЦИТАТА С86, finding-non-gaussianity-lives-in-the-spectrum-tail.md]
  κ₃(h_i) = Σ_rst M_rst·G_ir G_is G_it,  G = WᵀB   🔒ПЛОМБА: корел = 1.00000000
  ВИРОК: потрібен R ≈ 190 із 256 (числовий ранг 226), точність росте лише ×1.364
[ЦИТАТА С74, k3diag.py]  ранг 16 дає 83% — але це R² похибки `C` на ОДНОМУ шарі
```
⟹ ДВА ВИМІРИ, ДВА РІЗНІ ОБ'ЄКТИ, ПРОТИЛЕЖНІ ВИРОКИ (r=190 проти r=16).

🚨 І МОЯ ВЛАСНА ПОМИЛКА, ЯКУ ЦЕЙ СТЕНД ВИПРАВЛЯЄ:
`[ЗМІРЯНО k3rank97]` PR(κ₃)=1.12, топ-16 = 99.84% — я прочитав це як «ранг малий».
Але `[ЗАКОН КОРПУСУ]` **«ЕФЕКТИВНИЙ РАНГ МІРЯЄ, ДЕ ДИСПЕРСІЯ. Негаусовість живе
у ХВОСТІ спектру: домінантні компоненти за ЦГТ МАЙЖЕ ГАУСОВІ.»**
⟹ топ-r бере найбільшу ЕНЕРГІЮ, а вона може бути найгаусовішою частиною.
🔒 І це вже зміряно як наслідок: `[ЦИТАТА С86]` ранг-1 і ранг-3 наближення `S`
дали **×0.835 / ×0.847 — ГІРШЕ ЗА НІЧОГО**, попри еф. ранг `C` = 3.2.
📌 Це ТРЕТІЙ за вечір випадок «не той об'єкт» поспіль (розкид `|γ₃(v)|` · `util`
як residual · `PR` як ранг несення). Механізм щоразу однаковий: величина, яку
легко зміряти, не є величиною, яка вирішує.

════════ ЩО МІРЯЄМО — ПІДСТАНОВКОЮ, ЯК ПРИПИСУЄ С74 ════════
Не R², не енергію. ЧАСТКУ СПРАВЖНЬОЇ ПОХИБКИ, зняту поправкою 3-го порядку,
побудованою з РАНГ-r наближення:
```
   E    = Cov(ReLU(p)) − Cov_гаус          ← справжня негаусова похибка
   Δ_r  = поправка 3-го порядку з рангу r
   знято(r) = 1 − ‖E − Δ_r‖ / ‖E‖
```
Поправка 3-го порядку в базисі Ерміта (форма d9, запечатана квадратурою):
```
   Δ(j,k) = (λ₂₁/2)·a₂(j)a₁(k) + (λ₁₂/2)·a₁(j)a₂(k)
   a₁ = σΦ(t) · a₂ = σφ(t) · λ₂₁(j,k) = E[x̃_j² x̃_k],  x̃ = (p−m)/σ
```
Ранг-r: проєктуємо ЦЕНТРОВАНІ активації на топ-r власних напрямків **розгортки
κ₃** (не `Cov`! саме це і є носій), і рахуємо `λ` з проєкції.

🔒 ПЛОМБИ:
```
 (1) r = 256 мусить відтворити стелю d9: знято ≈ 87%  [pairtrunc97, крос-фіт]
 (2) r = 0   мусить дати РІВНО 0
 (3) базис із Cov замість κ₃ мусить бути ГІРШИМ — контроль «чужий підпростір»
```

════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════
 (а) 🔒r=256 дасть 80…92% (пломба на число d9)                     шанс 70%
 (б) крива буде ПЛОСКОЮ до r=64 ⟹ С86 права, потрібен ~190        шанс 45%
 (в) 50% знімається вже на r ≤ 16 ⟹ С74 права, гілка дешева        шанс 30%
 (г) 🔑 базис `κ₃` поб'є базис `Cov` щонайменше в 1.5×             шанс 55%
     ⟹ якщо НІ — «носій негаусовості» не має власного підпростору
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
LAY = int(sys.argv[4]) if len(sys.argv) > 4 else 31
CH = 1 << 13
NSK = 768
RS = (1, 4, 16, 32, 64, 128, 190, 256)
S2PI = (2.0 * np.pi) ** -0.5
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def run(W, seed, n):
    """активації ПЕРЕД шаром LAY."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(n, 256, generator=g, device=DEV)
    h = torch.relu(z @ W[0])
    for L in range(1, LAY):
        h = torch.relu(h @ W[L])
    return h


print(f"k3sweep97 · {DEV} · {NNET} мереж · {NS} семплів · шар {LAY}",
      flush=True)
t0 = time.time()
GOT = {("k3", r): [] for r in RS}
GOT.update({("cov", r): [] for r in RS})

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
            Wl = W[LAY].double()
            sd = 5100 + 17 * got

            # ── прохід 1: m, Cov активацій + скетч κ₃ ──
            s1 = torch.zeros(256, dtype=torch.float64, device=DEV)
            s2 = torch.zeros(256, 256, dtype=torch.float64, device=DEV)
            for o in range(0, NS, CH):
                hd = run(W, sd + o, CH).double()
                s1 += hd.sum(0)
                s2 += hd.T @ hd
            mh = s1 / NS
            Ch = s2 / NS - torch.outer(mh, mh)
            gv = torch.Generator(device=DEV).manual_seed(sd + 3)
            V = torch.randn(256, NSK, generator=gv, device=DEV,
                            dtype=torch.float64)
            V = V / V.norm(dim=0, keepdim=True)
            SK = torch.zeros(256, NSK, dtype=torch.float64, device=DEV)
            for o in range(0, NS, CH):
                xt = run(W, sd + o, CH).double() - mh
                SK += xt.T @ ((xt @ V) ** 2)
            U3 = torch.linalg.svd(SK / NS, full_matrices=False)[0]
            ev, Uc = torch.linalg.eigh(Ch)
            Uc = Uc.flip(1)

            # ── прохід 2: істинні Cov(ReLU) і λ, ПОВНІ та РАНГ-r ──
            acc = {}
            for key in ("full",) + tuple(f"{b}{r}" for b in ("k3", "cov")
                                         for r in RS):
                acc[key] = None
            n1 = torch.zeros(256, dtype=torch.float64, device=DEV)
            n2 = torch.zeros(256, 256, dtype=torch.float64, device=DEV)
            L21 = {k: torch.zeros(256, 256, dtype=torch.float64, device=DEV)
                   for k in ["full"] + [f"k3{r}" for r in RS]
                   + [f"cov{r}" for r in RS]}
            mp = mh @ Wl
            for o in range(0, NS, CH):
                xt = run(W, sd + o, CH).double() - mh
                p = xt @ Wl + mp
                r0 = torch.relu(p)
                n1 += r0.sum(0)
                n2 += r0.T @ r0
            mr = n1 / NS
            Cr = n2 / NS - torch.outer(mr, mr)
            sp = torch.sqrt(torch.clamp(
                torch.diagonal((Wl.T @ (Ch @ Wl))), min=1e-300))
            tt = mp / sp
            a1 = sp * (0.5 * torch.erfc(-tt * (2.0 ** -0.5)))
            a2 = sp * (torch.exp(-0.5 * tt * tt) * S2PI)

            # гаусів еталон: Мелер K висок. порядку не потрібен — беремо
            # ТОЧНУ гаусову Cov(ReLU) через підстановку гауса
            gg = torch.Generator(device=DEV).manual_seed(sd + 99)
            Lc = torch.linalg.cholesky(
                Wl.T @ (Ch @ Wl)
                + 1e-10 * torch.eye(256, dtype=torch.float64, device=DEV))
            g1 = torch.zeros(256, dtype=torch.float64, device=DEV)
            g2 = torch.zeros(256, 256, dtype=torch.float64, device=DEV)
            for o in range(0, NS, CH):
                e = torch.randn(CH, 256, generator=gg, device=DEV,
                                dtype=torch.float64)
                pg = e @ Lc.T + mp
                rg = torch.relu(pg)
                g1 += rg.sum(0)
                g2 += rg.T @ rg
            mg = g1 / NS
            Cg = g2 / NS - torch.outer(mg, mg)
            E = Cr - Cg

            # λ₂₁ повні і ранг-r
            for o in range(0, NS, CH):
                xt = run(W, sd + o, CH).double() - mh
                for key, Ub in (("full", None),) + tuple(
                        (f"k3{r}", U3[:, :r]) for r in RS) + tuple(
                        (f"cov{r}", Uc[:, :r]) for r in RS):
                    xx = xt if Ub is None else (xt @ Ub) @ Ub.T
                    q = (xx @ Wl) / sp
                    L21[key] += (q * q).T @ q
            for key in L21:
                lam = L21[key] / NS
                D = 0.5 * (lam * torch.outer(a2, a1)
                           + lam.T * torch.outer(a1, a2))
                off = ~torch.eye(256, dtype=torch.bool, device=DEV)
                rem = 1.0 - float(((E - D)[off]).norm() / (E[off]).norm())
                if key == "full":
                    full_rem = rem
                else:
                    b, r = key[:-len(key.lstrip('abcdefghijklmnopqrstuvwxyz'))
                                or 2], int(key[2:])
                    GOT[(key[:2], int(key[2:]))].append(rem)
            print(f"  net{got}  повний λ знімає {100*full_rem:.1f}%  "
                  f"({time.time()-t0:.0f} с)", flush=True)
            got += 1

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 62)
print("ЧАСТКА ПОХИБКИ ПОЗАДІАГОНАЛІ, ЗНЯТА РАНГ-r ПОПРАВКОЮ 3-го ПОРЯДКУ")
print("=" * 62)
print(f"{'r':>6s}{'базис κ₃':>14s}{'базис Cov (контроль)':>24s}"
      f"{'ціна % B':>12s}")
for r in RS:
    c = 100 * (1.426e8 * r / 16.0) / 2.72e11
    print(f"{r:>6d}{100*np.mean(GOT[('k3', r)]):>13.1f}%"
          f"{100*np.mean(GOT[('cov', r)]):>23.1f}%{c:>11.3f}%")
print("\n🔑 ВИРОК:")
print("  крива ПЛОСКА до r=64 ⟹ С86 права (потрібен ~190) ⟹ ціна 0.62% B ⟹ мертво")
print("  ≥50% уже на r≤16     ⟹ С74 права ⟹ ціна 0.05% B ⟹ ГІЛКА ЖИВА")
print("  базис κ₃ ≈ базис Cov ⟹ носій не має власного підпростору")
print("  🔒 r=256 мусить дати ≈87% — стеля d9 [pairtrunc97, крос-фіт]")
