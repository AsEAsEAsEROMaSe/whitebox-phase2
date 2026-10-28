"""rb97 — RAO-BLACKWELL НА РОБОЧОМУ `N`: ЧИ Б'Є ВІН ЧИСТУ МК ЗА ТІ САМІ СЕМПЛИ.

════════════════ ІДЕЯ, ЯКА ЛЕЖАЛА НА ОЧАХ ════════════════
`[ЗМІРЯНО lowdim97, 6 мереж]` participation ratio `Cov(h₃₁)` = **3.80**
(🔒 корпус каже 3.2 — пломба пройшла), частка дисперсії в топ-8 ПК = 0.8416.
⟹ активації виходу живуть у ~4 вимірах, а ми інтегруємо по них ВИПАДКОВИМИ
семплами у 256 вимірах.

🔑 МЕХАНІЗМ (не поправка — зміна оцінювача): розкласти `p = h·W` на
проєкцію на топ-`k` ПК і хвіст, і **проінтегрувати хвіст АНАЛІТИЧНО**:
```
   МК       Ê = (1/N) Σ ReLU(p_i)
   RB       Ê = (1/N) Σ E[ReLU(p) | топ-k координати]
            = (1/N) Σ [ pk_i·Φ(pk_i/s_t) + s_t·φ(pk_i/s_t) ]
   s_t² = Σ_{r>k} λ_r (u_r·W_j)²          ← дисперсія хвоста, ТОЧНА
```
За законом повної дисперсії `Var(E[X|Y]) ≤ Var(X)` ⟹ **дисперсія падає, і
жодного семпла на це не витрачено**. Ціна: `Cov(h)` + eigh + одна пара
Φ/φ на нейрон.

🚨 РИЗИК, НАЗВАНИЙ ДО ВИМІРУ: RB вносить ЗМІЩЕННЯ (хвіст не гаусів), а
зміщення не усереднюється. `[ЗМІРЯНО lowdim97]` стеля зміщення при N=262144:
k=8 → 0.1085% · k=16 → 0.0830% · k=64 → 0.0568%. Наша МК на робочому N має
0.1454% переважно з ДИСПЕРСІЇ ⟹ питання рівно в тому, що більше.

════════════════ ЩО МІРЯЄМО ════════════════
На ОДНИХ І ТИХ САМИХ семплах (спільний жереб ⟹ парне порівняння):
```
   MC   чиста Монте-Карло на N семплах
   RB_k Rao-Blackwell із топ-k, `Cov` і хвіст оцінені З ТИХ САМИХ N
   🔒 ПЛОМБА: RB з k = width МУСИТЬ дати РІВНО MC (хвіст порожній)
```
проти `final_means` (1e9 семплів, ПРЯМО з arrow).

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) 🔒RB при k=256 відтворить MC до 1e-12                       шанс 90%
 (б) RB при k=8 буде КРАЩИМ за MC на робочому N=6144             шанс 45%
 (в) оптимальне k буде 8…32 (компроміс зміщення/дисперсія)       шанс 55%
 (г) виграш, якщо буде, складе ×1.2…×2.0 у похибці               шанс 40%
     ⟹ ×1.4…×4 у MSE. Це НЕ ×1750, і я цього не обіцяю.
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
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 6144
REP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
KS = (4, 8, 16, 32, 64, 128, 256)
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = (2.0 * np.pi) ** -0.5

print(f"rb97 · {DEV} · {NNET} мереж · N={NW} · {REP} жеребів", flush=True)
t0 = time.time()
E_MC, E_RB = [], {k: [] for k in KS}

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
            Wl = W[31].double()
            mc_e, rb_e = [], {k: [] for k in KS}
            for r in range(REP):
                g = torch.Generator(device=DEV).manual_seed(
                    970000 + 101 * got + r)
                z = torch.randn(NW, 256, generator=g, device=DEV)
                h = torch.relu(z @ W[0])
                for L in range(1, 31):
                    h = torch.relu(h @ W[L])
                hd = h.double()
                p = hd @ Wl
                mc = torch.relu(p).mean(0)
                mc_e.append(float((mc - tru).norm() / tru.norm()))

                m = hd.mean(0)
                d = hd - m
                C = (d.T @ d) / float(NW)
                ev, U = torch.linalg.eigh(C)
                ev = ev.flip(0).clamp_min(0)
                U = U.flip(1)
                co = d @ U                       # (N, 256)
                UW = U.T @ Wl                    # (256, 256): рядок r = u_r·W
                base = m @ Wl
                for k in KS:
                    pk = base + co[:, :k] @ UW[:k]
                    if k >= 256:
                        st = torch.zeros(256, dtype=torch.float64, device=DEV)
                        rb = torch.relu(pk).mean(0)
                    else:
                        st = (ev[k:].unsqueeze(1) * UW[k:] ** 2).sum(0)
                        st = st.clamp_min(1e-300).sqrt()
                        t = pk / st
                        ph = torch.exp(-0.5 * t * t) * S2PI
                        Ph = 0.5 * torch.erfc(-t * (2.0 ** -0.5))
                        rb = (pk * Ph + st * ph).mean(0)
                    rb_e[k].append(float((rb - tru).norm() / tru.norm()))
            E_MC.append(float(np.mean(mc_e)))
            for k in KS:
                E_RB[k].append(float(np.mean(rb_e[k])))
            got += 1
            print(f"  net{got-1}  MC={100*E_MC[-1]:.4f}%  "
                  f"RB8={100*E_RB[8][-1]:.4f}%  RB32={100*E_RB[32][-1]:.4f}%"
                  f"  ({time.time()-t0:.0f} с)", flush=True)

mc = float(np.mean(E_MC))
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 66)
print(f"RAO-BLACKWELL проти ЧИСТОЇ МК на ОДНИХ СЕМПЛАХ (N={NW})")
print("=" * 66)
print(f"{'k':>6s}{'похибка':>12s}{'×MC у похибці':>16s}{'×MC у MSE':>13s}")
print(f"{'MC':>6s}{100*mc:>11.4f}%{1.0:>16.3f}{1.0:>13.3f}")
for k in KS:
    e = float(np.mean(E_RB[k]))
    print(f"{k:>6d}{100*e:>11.4f}%{mc/e:>16.3f}{(mc/e)**2:>13.3f}")
print(f"\n🔒 ПЛОМБА: k=256 мусить дати РІВНО MC — "
      f"різниця {abs(np.mean(E_RB[256])-mc)/mc:.2e}")
print("🎯 наша МК у сабміті = 0.1454% · замикання 0.5070% · ЛІДЕР 0.0064%")
