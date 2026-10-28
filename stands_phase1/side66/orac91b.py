"""orac91b — ВІДПОВІДЬ НА M1 СЕСІЇ АРХІВ: чи РЕАЛІЗОВНИЙ Еджворт на виході.

M1: «у O3e Еджворт лише на виході чи крізь ланцюг? κ понейронні чи скалярні?»
ВІДПОВІДЬ: **лише на виході, понейронні** (див. orac91.run, гілка i==L-1).
⟹ його пропозиція `O3e_MC` перевіряється прямо. Плюс я додаю варіант, якого
не було в жодного з нас і який ЄДИНИЙ дає щось подавальне.

════════════════ ЧОТИРИ НОВІ ВАРІАНТИ ════════════════
```
O0e     НАШЕ замикання (без оракула) + Еджворт на виході, κ ІСТИННІ
        ← ЄДИНИЙ подавальний варіант: m,C наші, κ з МК-семплів сабміта
O0e_MC  те саме, але κ з МК при N=8192 (масштаб сабміта)
O3e_MC  оракульні m,C + κ з МК при N=8192   ← чи не κ є вузьким місцем
O3e     оракульні m,C + κ істинні (як у orac91, для звірки)
```

════════════════ ЧОМУ O0e — ГОЛОВНИЙ ════════════════
`O3e` вимагає точних `m,C`, яких у нас нема. `O0e` — це наше НИНІШНЄ
замикання плюс дві редукції (`Σp³`,`Σp⁴`) на останньому шарі ≈ **0.01% B**.
Якщо він дає щось — це подається завтра.
⚠️ Контраргумент архіву (C3): «форма компенсує аргумент» — наш ланцюг веде
СВОЄ `h`, і κ істинного `h` до нього не пасують. Тоді O0e ≈ O0 або гірше.
Це і перевіряємо.

════════════════ ПЛОМБИ ════════════════
P1 O3e мусить відтворити orac91 (0.0708% при N=131072) — той самий код.
P2 κ з МК при N→∞ мусить сходитись до істинних ⟹ O3e_MC → O3e.
P3 ⛔КОНТРОЛЬ «ЧУЖІ κ»: κ від ІНШОЇ мережі. Мусить дати ≤ бази.
P4 сіди κ і цілі РОЗДІЛЕНІ (ціль — офіційна істина 1e9).

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) O3e_MC ≈ O3e (κ при 8192 досить точні)      — шанс 70%
 (б) O0e ≈ O0 або гірше (аргумент не той)        — шанс 65%
 (в) якщо O0e таки допоможе — це ×1.05..1.3      — шанс 35%
 (г) чужі κ дадуть гірше за базу                 — шанс 85%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NBIG = int(sys.argv[2]) if len(sys.argv) > 2 else 131072      # оракул m,C
NSUB = int(sys.argv[3]) if len(sys.argv) > 3 else 8192        # масштаб сабміта
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
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


def moments(ws, seed, N, want_C):
    """m,C активацій (якщо want_C) + κ₃,κ₄ передактивації ВИХОДУ."""
    g = torch.Generator().manual_seed(seed)
    n, L = ws[0].shape[0], len(ws)
    M = [torch.zeros(n, dtype=torch.float64) for _ in range(L)] if want_C else None
    C = [torch.zeros(n, n, dtype=torch.float64) for _ in range(L)] if want_C else None
    S = torch.zeros(4, n, dtype=torch.float64)
    done = 0
    while done < N:
        b = min(2048, N - done)
        x = torch.randn(b, n, generator=g)
        for i, w in enumerate(ws):
            h = x @ w
            if i == L - 1:
                hd = h.double()
                for k in range(4):
                    S[k] += (hd ** (k + 1)).sum(0)
            x = torch.relu(h)
            if want_C:
                xd = x.double()
                M[i] += xd.sum(0)
                C[i] += xd.T @ xd
        done += b
    TM = None
    if want_C:
        TM = []
        for i in range(L):
            m = M[i] / N
            TM.append((m, C[i] / N - torch.outer(m, m)))
    e = S / N
    v = (e[1] - e[0] ** 2).clamp(min=1e-30)
    k3 = e[2] - 3 * e[0] * e[1] + 2 * e[0] ** 3
    k4 = e[3] - 4 * e[0] * e[2] + 6 * e[0] ** 2 * e[1] - 3 * e[0] ** 4 - 3 * v * v
    s = v.sqrt()
    return TM, (k3 / s ** 3, k4 / s ** 4)


def relu_gauss(mp, s, g1=None, g2=None):
    t = mp / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mp * p + s * f
    va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
    if g1 is not None:
        ma = ma + s * f * (-(g1 / 6.0) * t + (g2 / 24.0) * (t * t - 1.0))
    return p, f, ma, va


def run(ws, TM, gam, orac):
    """orac=True — підміняти m,C істинними. gam=None — без Еджворта."""
    n, L = ws[0].shape[0], len(ws)
    m = torch.zeros(n, dtype=torch.float64)
    C = torch.eye(n, dtype=torch.float64)
    for i, w in enumerate(ws):
        wd = w.double()
        cp = wd.T @ (C @ wd)
        s = cp.diagonal().clamp(min=1e-20).sqrt()
        mp = m @ wd
        if i == L - 1:
            return relu_gauss(mp, s, *(gam if gam else (None, None)))[2]
        p, f, ma, va = relu_gauss(mp, s)
        Cn = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        Cn = Cn + torch.diag(va - Cn.diagonal())
        m, C = (TM[i][0], TM[i][1]) if orac else (ma, Cn)


LAB = ["O0 наше замикання", "O0e НАШЕ + Еджворт (κ істинні)",
       "O0e_MC НАШЕ + Еджворт (κ з МК 8192)", "⛔O0e_ЧУЖІ κ",
       "O3 оракул m+C", "O3e оракул + κ істинні", "O3e_MC оракул + κ з МК 8192"]
ERR = {k: [] for k in LAB}
Z = []
prev_gam = None
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        tr = torch.as_tensor(truth, dtype=torch.float64)
        Z.append(float((tr * tr).mean()))
        TM, gbig = moments(ws, 770000 + 13 * q, NBIG, True)      # оракул
        _, gsub = moments(ws, 550000 + 29 * q, NSUB, False)      # масштаб сабміта
        foreign = prev_gam if prev_gam is not None else gbig
        for lab, gam, orac in (
                (LAB[0], None, False), (LAB[1], gbig, False),
                (LAB[2], gsub, False), (LAB[3], foreign, False),
                (LAB[4], None, True), (LAB[5], gbig, True),
                (LAB[6], gsub, True)):
            pr = run(ws, TM, gam, orac)
            ERR[lab].append(float(((pr - tr) ** 2).mean()))
        prev_gam = gbig
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(Z)}\n")

z = float(np.mean(Z))
e0 = float(np.mean(ERR[LAB[0]]))
print("=" * 96)
print(f"ЕДЖВОРТ НА ВИХОДІ — ЧИ РЕАЛІЗОВНИЙ · {len(Z)} мереж · "
      f"оракул N={NBIG} · κ сабміта N={NSUB}")
print("=" * 96)
print(f"{'варіант':>36s} {'абс. MSE':>13s} {'відн.':>10s} {'× до O0':>10s}")
for k in LAB:
    e = float(np.mean(ERR[k]))
    print(f"{k:>36s} {e:>13.4e} {100*np.sqrt(e/z):>9.4f}% {e0/e:>10.2f}")
print(f"{'🥇 ЛІДЕР':>36s} {0.0749**2*z/1e4:>13.4e} {0.0749:>9.4f}%")

g1 = e0 / float(np.mean(ERR[LAB[1]]))
gf = e0 / float(np.mean(ERR[LAB[3]]))
print(f"\n⛔ P3 КОНТРОЛЬ: своя ×{g1:.4f} проти ЧУЖИХ κ ×{gf:.4f} "
      f"{'✅' if g1 > gf * 1.05 else '⛔ ІНФОРМАЦІЇ НЕМА'}")
print(f"⛔ P2 O3e_MC/O3e = "
      f"{float(np.mean(ERR[LAB[6]]))/float(np.mean(ERR[LAB[5]])):.4f} "
      f"(1.00 ⟹ κ при 8192 досить)")
print(f"\n📣 ВІДПОВІДЬ НА M1:")
print(f"   O0e (ПОДАВАЛЬНИЙ: наші m,C + Еджворт) ×{g1:.4f}")
print(f"   {'✅ Є ЩО ПОДАВАТИ' if g1 > 1.05 else '⛔ форма справді компенсує аргумент — O0e мертвий'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/orac91b.npz",
         **{f"E{i}": np.array(ERR[k]) for i, k in enumerate(LAB)}, Z=np.array(Z))
