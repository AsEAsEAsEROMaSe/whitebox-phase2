"""sigma91 — E1+E2 СЕСІЇ АРХІВ І СУПЕРЕЧНІСТЬ С56 ↔ orac91.

════════════════ СУПЕРЕЧНІСТЬ, ЯКУ ЦЕ ЗНІМАЄ ════════════════
```
С56 CP_RESET :  C ← істинна щошару  ⟹  m@31 = 0.140%   КРАЩЕ за базу 0.998%
orac91 O2    :  C ← істинна щошару  ⟹  m    = 1.0835%  ГІРШЕ за базу 0.7908%
```
Та сама операція, протилежний знак. Від відповіді залежить, чи живе `E1`
(підставити вибіркову `σ` у стан рекурсії — коштує НУЛЬ, бо сабміт її вже
рахує на всіх 32 шарах і викидає для шарів 9..32).

════════════════ ЩО САМЕ РОБИМО ════════════════
На кожному шарі підмінюємо ЛИШЕ масштаб, зберігаючи кореляційну структуру:
```
s_λ  = (1−λ)·s_ланцюг + λ·s_істина            λ ∈ {0, .25, .5, .75, 1}
cp   ← cp · outer(s_λ/s_ланцюг, s_λ/s_ланцюг)  ← діагональна конгруенція,
                                                 PSD зберігається ТОЧНО
```
Це РІВНО об'єкт `E1` в оракульній версії (верхня межа) і водночас крива
чутливості `E2`: `m@вихід` як функція похибки `σ`.

🔑 ВІДМІННІСТЬ ВІД `O2`: там підмінялась ВСЯ `C` (діагональ І недіагональ),
тут — лише МАСШТАБ, кореляції лишаються від ланцюга. Якщо знаки розійдуться,
винна саме недіагональ.

════════════════ ПЛОМБИ ════════════════
P1 λ=0 мусить дати ПОБІТОВО базу (0.7908% на цьому зрізі).
P2 конгруенція зберігає PSD: min власне число cp ≥ 0 (перевіряємо на 1 мережі).
P3 ⛔КОНТРОЛЬ «ЧУЖА σ» з іншої мережі при λ=1 — мусить бути ГІРШЕ за базу.
P4 сіди `σ`-оракула і цілі РОЗДІЛЕНІ (ціль — офіційна істина 1e9).
P5 масштабування: N оракула ×4 не мусить зсунути вирок.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) λ=1 дасть КРАЩЕ за базу (С56 права)         — шанс 55%
 (б) крива буде монотонною без насичення         — шанс 60%
 (в) чужа σ дасть гірше за базу                  — шанс 85%
 🎯 якщо (а) — `E1` живий і коштує НУЛЬ; шанс, що реальна (не оракульна)
    вибіркова `σ` дасть ≥×1.3 у балі: **30%** (оцінка архіву, приймаю)
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
LAM = (0.0, 0.25, 0.5, 0.75, 1.0)
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


def true_sigma(ws, seed, N):
    """ІСТИННА σ ПЕРЕДАКТИВАЦІЇ на кожному шарі (окремий сід від цілі)."""
    g = torch.Generator().manual_seed(seed)
    n, L = ws[0].shape[0], len(ws)
    S1 = [torch.zeros(n, dtype=torch.float64) for _ in range(L)]
    S2 = [torch.zeros(n, dtype=torch.float64) for _ in range(L)]
    done = 0
    while done < N:
        b = min(2048, N - done)
        x = torch.randn(b, n, generator=g)
        for i, w in enumerate(ws):
            h = x @ w
            hd = h.double()
            S1[i] += hd.sum(0)
            S2[i] += (hd * hd).sum(0)
            x = torch.relu(h)
        done += b
    return [((S2[i] / N - (S1[i] / N) ** 2).clamp(min=1e-30)).sqrt()
            for i in range(L)]


def relu_gauss(mp, s):
    t = mp / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mp * p + s * f
    va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
    return p, f, ma, va


def chain(ws, sig, lam, check_psd=False):
    """sig — список істинних σ по шарах; lam=0 ⟹ побітово база."""
    n, L = ws[0].shape[0], len(ws)
    m = torch.zeros(n, dtype=torch.float64)
    C = torch.eye(n, dtype=torch.float64)
    err = []
    for i, w in enumerate(ws):
        wd = w.double()
        cp = wd.T @ (C @ wd)
        s0 = cp.diagonal().clamp(min=1e-20).sqrt()
        if lam > 0.0:
            s = (1.0 - lam) * s0 + lam * sig[i]
            r = s / s0
            cp = cp * torch.outer(r, r)            # діагональна конгруенція
            if check_psd and i == 5:
                ev = torch.linalg.eigvalsh(cp)
                err.append(float(ev.min()))
        else:
            s = s0
        mp = m @ wd
        p, f, ma, va = relu_gauss(mp, s)
        if i == L - 1:
            return ma, err
        Cn = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        C = Cn + torch.diag(va - Cn.diagonal())
        m = ma


LAB = [f"λ={l:.2f}" for l in LAM] + ["⛔ЧУЖА σ (λ=1)"]
ERR = {k: [] for k in LAB}
SIGERR = []
Z = []
prev = None
psd = None
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        tr = torch.as_tensor(truth, dtype=torch.float64)
        Z.append(float((tr * tr).mean()))
        sig = true_sigma(ws, 660000 + 11 * q, NORA)
        # похибка σ ланцюга на шарі 31 — те, що С56 називає 16.6%
        base_s = []
        m0 = torch.zeros(256, dtype=torch.float64)
        C0 = torch.eye(256, dtype=torch.float64)
        for i, w in enumerate(ws):
            wd = w.double()
            cp = wd.T @ (C0 @ wd)
            s0 = cp.diagonal().clamp(min=1e-20).sqrt()
            base_s.append(float(((s0 - sig[i]).abs() / sig[i]).mean()))
            mp = m0 @ wd
            p, f, ma, va = relu_gauss(mp, s0)
            if i == len(ws) - 1:
                break
            Cn = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s0, f / s0)
            C0 = Cn + torch.diag(va - Cn.diagonal())
            m0 = ma
        SIGERR.append(base_s)
        for l, k in zip(LAM, LAB):
            pr, e = chain(ws, sig, l, check_psd=(q == 0 and l == 1.0))
            if e:
                psd = e[0]
            ERR[k].append(float(((pr - tr) ** 2).mean()))
        fs = prev if prev is not None else sig
        pr, _ = chain(ws, fs, 1.0)
        ERR[LAB[-1]].append(float(((pr - tr) ** 2).mean()))
        prev = sig
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(Z)}\n")

z = float(np.mean(Z))
SIGERR = np.array(SIGERR)
e0 = float(np.mean(ERR[LAB[0]]))
print("=" * 92)
print(f"σ У СТАН РЕКУРСІЇ · {len(Z)} мереж · істина 1e9 · σ-оракул N={NORA}")
print("=" * 92)
print("похибка σ ЛАНЦЮГА по шарах (С56 каже 16.6% на 31):")
print("  " + " · ".join(f"ш{i}:{100*SIGERR[:, i].mean():.1f}%"
                        for i in (1, 4, 8, 12, 16, 24, 31)))
print(f"\n{'варіант':>18s} {'абс. MSE':>13s} {'відн.':>10s} {'× до бази':>11s}")
for k in LAB:
    e = float(np.mean(ERR[k]))
    print(f"{k:>18s} {e:>13.4e} {100*np.sqrt(e/z):>9.4f}% {e0/e:>11.4f}")
print(f"{'🥇 ЛІДЕР':>18s} {0.0749**2*z/1e4:>13.4e} {0.0749:>9.4f}%")

g1 = e0 / float(np.mean(ERR[LAB[4]]))
gf = e0 / float(np.mean(ERR[LAB[-1]]))
print(f"\n⛔ P2 PSD після конгруенції: min власне = {psd:.3e} "
      f"{'✅' if psd is not None and psd > -1e-9 else '⚠️'}")
print(f"⛔ P3 своя σ ×{g1:.4f} проти ЧУЖОЇ ×{gf:.4f} "
      f"{'✅' if g1 > gf * 1.05 else '⛔ ІНФОРМАЦІЇ НЕМА'}")
print(f"\n📣 ВИРОК СУПЕРЕЧНОСТІ С56 ↔ orac91:")
print(f"   λ=1 (точна σ у стан) ×{g1:.4f}")
print(f"   {'✅ С56 ПРАВА — E1 живий' if g1 > 1.05 else '⛔ orac91 правий — підміна одного каналу шкодить, E1 мертвий'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/sigma91.npz",
         **{f"E{i}": np.array(ERR[k]) for i, k in enumerate(LAB)},
         SIGERR=SIGERR, Z=np.array(Z))
