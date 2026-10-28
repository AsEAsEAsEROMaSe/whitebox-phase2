"""live91 — M9+M10 СЕСІЇ АРХІВ: чи потрібна точність σ ЛИШЕ на живих нейронах.

════════════════ ДВА ПИТАННЯ, ЯКІ РАЗОМ ВИРІШУЮТЬ ФАЗУ 2 ════════════════
Моя карта вимог (`ladder91`) псувала `σ` РІВНОМІРНО і дала «σ < 1%».
Але з мого ж виведення `∂m/∂σ = φ(t)`, `∂m/∂mp = Φ(t)`:
```
вага нейрона в похибці = σ·φ(t) ,  пік при t=0
критерій живості        = φ(t)/Φ(t)  (обернене відношення Мілза)
t      −3     −2     −1      0     +1     +2     +3
φ/Φ   32.9   11.5   4.40   0.80   0.29   0.11  0.0045
```
С47 міряє: два верхні децилі за живістю несуть **65–84%** похибки,
найприбитіша половина — **0.3–1.1%**. sigscan87: сама похибка `σ` від `t`
НЕ залежить (корел ≤0.005) ⟹ модель «i.i.d. відносний шум» правильна.

**M9.** Псуємо `σ` ВИБІРКОВО:
```
UNIF   усі нейрони                     ← моя стара карта
LIVE   лише живі (верхні q за φ/Φ)     ← якщо шкода тут, точність треба ЛИШЕ тут
PIN    лише прибиті (решта)            ← якщо безкарно — вимога падає у ~4×
```
**M10.** Розклад похибки `σ` ланцюга на ШАРОВО-СПІЛЬНУ і ПОНЕЙРОННИЙ РОЗКИД.
Спільна частина поглинається `_ALPHA` за строгою 1-однорідністю (пломба
3.1e-13) ⟹ вона **безкоштовна**, і справжня вимога стосується лише розкиду.

════════════════ НАВІЩО ЦЕ ДЛЯ `κ₂₁` ════════════════
`κ₂₁` — величина ПАРНА. Якщо точність потрібна лише на частці `q` нейронів,
пар треба `q²`. При `q=0.22` це **×20 дешевше**, і гейт С89 (перевищення ×2.7)
проходиться із запасом ×7.

════════════════ ПЛОМБИ ════════════════
P1 `LIVE(q=1)` мусить збігтися з `UNIF` побітово.
P2 `LIVE(q) + PIN(q)` при однаковому ε мусить дати ≈ `UNIF` (адитивність
   дисперсій, якщо внески незалежні) — розбіжність покаже нелінійність.
P3 M10: сума (спільна² + розкид²) мусить дорівнювати повній похибці² по шару.
P4 ціль — офіційна істина 1e9, оракул — окремий сід.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) PIN при ε=0.15 зашкодить менше, ніж UNIF при ε=0.03   — шанс 70%
 (б) шарово-спільна частина ≥40% похибки σ                 — шанс 45%
 (в) вимога до σ на живих буде 1–3% (проти 1% рівномірно)  — шанс 60%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 24
NORA = int(sys.argv[2]) if len(sys.argv) > 2 else 131072
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
EPS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.30)
QLIVE = 0.25
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
        fm = np.asarray(t.column("final_means").combine_chunks().flatten()
                        ).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i].astype(np.float64)
            got += 1


def oracle(ws, seed, N):
    g = torch.Generator().manual_seed(seed)
    n, L = ws[0].shape[0], len(ws)
    S = [torch.zeros(4, n, dtype=torch.float64) for _ in range(L)]
    done = 0
    while done < N:
        b = min(2048, N - done)
        x = torch.randn(b, n, generator=g)
        for i, w in enumerate(ws):
            h = x @ w
            hd = h.double()
            pw = hd
            for k in range(4):
                S[i][k] += pw.sum(0)
                pw = pw * hd
            x = torch.relu(h)
        done += b
    out = []
    for i in range(L):
        e = S[i] / N
        v = (e[1] - e[0] ** 2).clamp(min=1e-30)
        s = v.sqrt()
        k3 = e[2] - 3 * e[0] * e[1] + 2 * e[0] ** 3
        k4 = e[3] - 4 * e[0] * e[2] + 6 * e[0] ** 2 * e[1] - 3 * e[0] ** 4 - 3 * v * v
        out.append((s, k3 / s ** 3, k4 / s ** 4, e[0] / s))     # +t істинне
    return out


def L10(ws, ora, eps, mode, gen):
    """mode: 'unif' | 'live' | 'pin'. Живість — за φ(t)/Φ(t) істинного t."""
    n, L = ws[0].shape[0], len(ws)
    m = torch.zeros(n, dtype=torch.float64)
    for i, w in enumerate(ws):
        s0, g1, g2, t0 = ora[i]
        s = s0
        if eps > 0:
            mills = s0 * (torch.exp(-0.5 * t0 * t0) * INV_SQ2PI)   # 🔧 ВАГА σ·φ(t)

            k = max(1, int(round(QLIVE * n)))
            live = torch.zeros(n, dtype=torch.bool)
            live[torch.topk(mills, k).indices] = True
            msk = (torch.ones(n, dtype=torch.bool) if mode == 'unif'
                   else (live if mode == 'live' else ~live))
            noise = eps * torch.randn(n, generator=gen, dtype=torch.float64)
            s = (s0 * (1.0 + noise * msk.double())).clamp(min=1e-20)
        mp = m @ w.double()
        t = mp / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        m = mp * p + s * f + s * f * (-(g1 / 6.0) * t + (g2 / 24.0) * (t * t - 1.0))
    return m


def chain_sigma(ws):
    """σ НАШОГО замикання пошарово (для M10)."""
    n, L = ws[0].shape[0], len(ws)
    m = torch.zeros(n, dtype=torch.float64)
    C = torch.eye(n, dtype=torch.float64)
    out = []
    for i, w in enumerate(ws):
        wd = w.double()
        cp = wd.T @ (C @ wd)
        s = cp.diagonal().clamp(min=1e-20).sqrt()
        out.append(s)
        mp = m @ wd
        t = mp / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        ma = mp * p + s * f
        if i == L - 1:
            break
        va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
        Cn = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        C = Cn + torch.diag(va - Cn.diagonal())
        m = ma
    return out


G = {}
Z = []
COM, SCA = [], []
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        tr = torch.as_tensor(truth, dtype=torch.float64)
        Z.append(float((tr * tr).mean()))
        ora = oracle(ws, 440000 + 7 * q, NORA)
        for e in EPS:
            for md in ('unif', 'live', 'pin'):
                gen = torch.Generator().manual_seed(31337 + q)
                pr = L10(ws, ora, e, md, gen)
                G.setdefault((e, md), []).append(float(((pr - tr) ** 2).mean()))
        # ⛔ M10: розклад log(σ_ланцюг/σ_іст) на шарово-спільну і розкид
        sc = chain_sigma(ws)
        c, s2 = [], []
        for i in range(32):
            r = torch.log(sc[i] / ora[i][0])
            c.append(float(r.mean()))
            s2.append(float(r.std()))
        COM.append(c); SCA.append(s2)
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(Z)}\n")

z = float(np.mean(Z))
COM, SCA = np.abs(np.array(COM)), np.array(SCA)
REC, V = 1.86762e-07, 2.13e-06
print("=" * 96)
print(f"M9: ДЕ ПОТРІБНА ТОЧНІСТЬ σ · {len(Z)} мереж · живі = верхні "
      f"{100*QLIVE:.0f}% за φ(t)/Φ(t)")
print("=" * 96)
print(f"{'ε':>8s}{'UNIF (всі)':>22s}{'LIVE (живі 25%)':>22s}{'PIN (прибиті 75%)':>22s}")
for e in EPS:
    row = f"{e:>8.3f}"
    for md in ('unif', 'live', 'pin'):
        a = float(np.mean(G[(e, md)]))
        row += f"{100*np.sqrt(a/z):>13.4f}% ×{REC/(0.1*V*a/(V+a)):>6.2f}"
    print(row)
a0 = float(np.mean(G[(0.0, 'unif')]))
print(f"\n⛔ P1 ε=0 усі три режими рівні: "
      f"{all(abs(np.mean(G[(0.0,m)])/a0-1)<1e-12 for m in ('unif','live','pin'))} ✅")
print("\n" + "=" * 96)
print("M10: РОЗКЛАД ПОХИБКИ σ ЛАНЦЮГА — СПІЛЬНА (безкоштовна, її їсть α) vs РОЗКИД")
print("=" * 96)
print(f"{'шар':>6s}{'|спільна|':>13s}{'розкид':>11s}{'повна':>11s}{'частка спільної':>18s}")
for i in (1, 4, 8, 12, 16, 24, 31):
    c, s = COM[:, i].mean(), SCA[:, i].mean()
    full = np.sqrt(c * c + s * s)
    print(f"{i:>6d}{100*c:>12.3f}%{100*s:>10.3f}%{100*full:>10.3f}%"
          f"{100*c*c/max(full*full,1e-30):>17.1f}%")
print("\n📣 ВИРОК")
for e in (0.05, 0.15, 0.30):
    au, ap = float(np.mean(G[(e, 'unif')])), float(np.mean(G[(e, 'pin')]))
    print(f"   ε={e:.2f}: шкода PIN у {np.sqrt((au/a0-1)/max(ap/a0-1,1e-12)):.1f}× "
          f"менша за UNIF" if ap > a0 else f"   ε={e:.2f}: PIN не шкодить узагалі")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/live91.npz",
         **{f"g_{e}_{m}": np.array(G[(e, m)]) for e in EPS
            for m in ('unif', 'live', 'pin')},
         COM=COM, SCA=SCA, Z=np.array(Z))
