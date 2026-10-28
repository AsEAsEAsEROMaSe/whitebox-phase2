"""lowdim97 — ЧИ МОЖНА ЗАМІНИТИ МК КВАДРАТУРОЮ В МАЛІЙ РОЗМІРНОСТІ.

════════════════ ЩО ЛЕЖИТЬ НА ОЧАХ І НЕ ВИКОРИСТАНЕ ════════════════
`[ЦИТАТА корпус]` еф. ранг `Cov(h)` падає **127.6 → 3.2** до виходу ·
`cos(h(z),h(z′))@31 = 0.9741`. Ми використовували це ЛИШЕ щоб ПОЯСНЮВАТИ
аномалії (розкид γ₃, концентрація модуля) — і жодного разу ЯК МЕТОД.

🔑 ПИТАННЯ: якщо `h_31` живе у ~3 вимірах, навіщо інтегрувати по ньому
**6000 ВИПАДКОВИМИ семплами у 256 вимірах**? У 3 вимірах гаусова квадратура
з 10 вузлами на вимір = 1000 точок і ТОЧНА для поліномів степеня 19.
```
   наш метод   vs sampling  2.6×
   ЛІДЕР       vs sampling  1750×      ⟹ це не «краще семплювання»,
                                          це ВІДМОВА від семплювання
```
🎯 Стрибок ×1750 не буває від поправок. Він буває від зміни ПРЕДСТАВЛЕННЯ:
випадкові вузли → детерміновані, 256 вимірів → еф. розмірність.

════════════════ ЩО САМЕ МІРЯЄМО (ДЕШЕВО, БЕЗ РЕАЛІЗАЦІЇ МЕТОДУ) ════════════
Не будуємо квадратуру. Міряємо СТЕЛЮ: скільки інформації про відповідь несе
проєкція `h` на топ-`k` головних компонент.
```
   h = m + Σ_r √λ_r u_r ξ_r        (ξ — некорельовані, НЕ гаусові)
   h^(k) = m + проєкція на топ-k ПК   (решта — ЗАМІНЕНА НА НУЛЬ)
   ціль = E[ReLU(h·W_out)]  проти  E[ReLU(h^(k)·W_out)]
```
🚨 КОНТРОЛЬ, БЕЗ ЯКОГО ВИМІР НІЧОГО НЕ КАЖЕ: «занулити хвіст» — це НЕ те саме,
що «проінтегрувати хвіст гаусово». Тому міряємо ТРИ варіанти:
```
   (A) хвіст ЗАНУЛЕНО            ← песимістична межа
   (B) хвіст ГАУСІВ, точно       ← реалістична: топ-k квадратурою, хвіст аналітично
   (C) повний МК на N семплів    ← еталон (він і є ціль)
```
Якщо (B) при `k`=3..8 дає похибку ≪ 0.145% (наша МК) — метод живий, і це
найбільший важіль, який ми бачили. Якщо (B) ≈ (A) — еф. ранг оманливий.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) (A) при k=3 дасть похибку > 5% — занулення хвоста грубе        шанс 75%
 (б) (B) при k=8 дасть похибку < 0.145% (б'є нашу МК)               шанс 30%
 (в) 🔑 частка дисперсії у топ-8 ПК буде > 0.90                     шанс 60%
 (г) еф. ранг за participation ratio відтворить 3.2 ± 1 на шарі 31  шанс 70%
     ← 🔒ПЛОМБА на прилад: не відтворить ⟹ міряю не той об'єкт
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
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 8
NS = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 19)
NW = int(sys.argv[4]) if len(sys.argv) > 4 else 6144   # робоче N сабміту
CH = 1 << 14
KS = (8, 16, 32, 64)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")

print(f"lowdim97 · {DEV} · {NNET} мереж · {NS} семплів", flush=True)
t0 = time.time()
ERR_A = {k: [] for k in KS}
ERR_B = {k: [] for k in KS}
PR, VAR8 = [], []

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
        fm = np.asarray(tb.column("final_means").combine_chunks()
                        .flatten()).reshape(tb.num_rows, 256)
        for i in range(tb.num_rows):
            if got >= NNET:
                break
            W = [torch.as_tensor(np.ascontiguousarray(w[i][L]), device=DEV)
                 for L in range(32)]
            tru = torch.as_tensor(fm[i].astype(np.float64), device=DEV)

            # ── прохід 1: m і Cov активацій ПЕРЕД останнім шаром ──
            s1 = torch.zeros(256, dtype=torch.float64, device=DEV)
            s2 = torch.zeros(256, 256, dtype=torch.float64, device=DEV)
            for o in range(0, NS, CH):
                g = torch.Generator(device=DEV).manual_seed(9700 + 7 * got + o)
                z = torch.randn(CH, 256, generator=g, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 31):
                    h = torch.relu(h @ W[L])
                hd = h.double()
                s1 += hd.sum(0)
                s2 += hd.T @ hd
            m = s1 / NS
            C = s2 / NS - torch.outer(m, m)
            ev, U = torch.linalg.eigh(C)
            ev = ev.flip(0).clamp_min(0)
            U = U.flip(1)
            pr = float(ev.sum() ** 2 / (ev * ev).sum())
            PR.append(pr)
            VAR8.append(float(ev[:8].sum() / ev.sum()))

            # ── прохід 2: похибка усічення, ТРИ варіанти ──
            acc = {("A", k): torch.zeros(256, dtype=torch.float64, device=DEV)
                   for k in KS}
            acc.update({("B", k): torch.zeros(256, dtype=torch.float64,
                                              device=DEV) for k in KS})
            Wl = W[31].double()
            for o in range(0, NS, CH):
                g = torch.Generator(device=DEV).manual_seed(9700 + 7 * got + o)
                z = torch.randn(CH, 256, generator=g, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 31):
                    h = torch.relu(h @ W[L])
                d = h.double() - m
                co = d @ U                       # координати у власному базисі
                for k in KS:
                    # (A) хвіст ЗАНУЛЕНО
                    hk = m + co[:, :k] @ U[:, :k].T
                    acc[("A", k)] += torch.relu(hk @ Wl).sum(0)
                    # (B) хвіст ГАУСІВ: p = hk·W має ще дисперсію від хвоста
                    #     ⟹ E[ReLU] береться аналітично при відомій σ_tail
                    pk = hk @ Wl
                    st = (ev[k:].unsqueeze(1) * (U[:, k:].T @ Wl) ** 2).sum(0)
                    st = st.clamp_min(1e-30).sqrt()
                    t = pk / st
                    ph = torch.exp(-0.5 * t * t) / (2 * np.pi) ** 0.5
                    Ph = 0.5 * torch.erfc(-t * (2.0 ** -0.5))
                    acc[("B", k)] += (pk * Ph + st * ph).sum(0)
            for k in KS:
                for tag in ("A", "B"):
                    pred = acc[(tag, k)] / NS
                    e = float((pred - tru).norm() / tru.norm())
                    (ERR_A if tag == "A" else ERR_B)[k].append(e)
            got += 1
            print(f"  net{got-1}  PR={pr:.2f}  var@8={VAR8[-1]:.4f}"
                  f"  ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 72)
print("ЧИ ЖИВЕ ВІДПОВІДЬ У МАЛІЙ РОЗМІРНОСТІ")
print("=" * 72)
print(f"🔒 ПЛОМБА: participation ratio = {np.mean(PR):.2f} "
      f"(корпус каже 3.2 на виході) · частка дисперсії у топ-8 = "
      f"{np.mean(VAR8):.4f}")
print(f"\n{'k':>4s}{'(A) хвіст=0':>16s}{'(B) хвіст гаусів':>20s}")
for k in KS:
    print(f"{k:>4d}{100*np.mean(ERR_A[k]):>15.4f}%"
          f"{100*np.mean(ERR_B[k]):>19.4f}%")
print("\n🎯 ПОРІГ: наша МК = 0.1454% · наше замикання = 0.5070% · ЛІДЕР = 0.0064%")
print("   (B) < 0.145% при малому k ⟹ КВАДРАТУРА В k ВИМІРАХ Б'Є МК")
print("   (B) ≈ (A) ⟹ еф. ранг оманливий, хвіст несе відповідь")
