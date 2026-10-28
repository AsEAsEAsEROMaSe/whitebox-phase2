"""orac91 — ЧИ ЗДАТНА ДЕТЕРМІНОВАНА ГІЛКА ДІЙТИ ДО ЛІДЕРА. ОРАКУЛЬНІ МЕЖІ.

════════════════ ПИТАННЯ, ЯКЕ ВИРІШУЄ ВСЕ ════════════════
Лідеру потрібно **0.0749%** відносної похибки. Наше замикання — 0.928%,
офіційна `covariance_propagation` — 0.96%. Тобто лідер у **140× кращий за
повну коваріаційну пропагацію**, і це не може бути моментним методом.

С88 стверджує ОДНОЧАСНО дві речі, що суперечать одна одній:
   «негаусовість = 99.9% похибки замикання»   і   «оракул мішаних κ = ×1.314».
Друга зміряна на 24 мережах приладом із SE 5.5% ⟹ підозріла.

════════════════ ДРАБИНА ОРАКУЛІВ ════════════════
На кожному шарі підмінюємо ОДНУ наближену величину на ІСТИННУ (з МК):
```
O0  наше замикання                              (база)
O1  точне m на кожному шарі
O2  точна C на кожному шарі
O3  точні m І C  ⟹ лишається РІВНО негаусовість ОСТАННЬОГО шару
O3e O3 + Еджворт κ₃,κ₄ на виході (чи ловиться вона низькими кумулянтами)
```
🎯 **O3 — вирок двомоментному представленню.** Якщо O3 > 0.0749% — двох
моментів НЕ ВИСТАЧАЄ В ПРИНЦИПІ, хай як точно їх рахувати.

📐 Еджворт для `E[ReLU]` (виведено, не вгадано):
```
E[max(X,0)] = s[tΦ(t)+φ(t)] − (γ₁/6)·t·φ(t)·s + (γ₂/24)·(t²−1)·φ(t)·s
```
з `∫_{−t}^∞ He_n φ = He_{n−1}(−t)φ(t)` і `zHe_n = He_{n+1}+nHe_{n−1}`.

════════════════ ПЛОМБИ ════════════════
P1 ⛔СІДИ ОРАКУЛА І ЦІЛІ РОЗДІЛЕНІ: ціль — офіційна істина з 1e9 семплів,
   оракул — власна МК з іншим сідом ⟹ «оракул ділить жереб» неможливий.
P2 O0 мусить відтворити 7.83e-05 (зміряно `cal91`).
P3 МАСШТАБУВАННЯ: при N×4 шум оракула падає вчетверо. Якщо O3 покращується —
   він шумо-обмежений, і справжня межа НИЖЧА.
P4 Еджворт при γ=0 мусить дати рівно гаусову формулу (тотожність).
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = "cpu"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
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


def truth_moments(ws, seed, N):
    """ІСТИННІ m, C активацій на кожному шарі + κ₃,κ₄ передактивації виходу.
    ⛔P1: сід ІНШИЙ, ніж у цілі (ціль — офіційна істина з 1e9)."""
    g = torch.Generator().manual_seed(seed)
    n = ws[0].shape[0]
    L = len(ws)
    M = [torch.zeros(n, dtype=torch.float64) for _ in range(L)]
    C = [torch.zeros(n, n, dtype=torch.float64) for _ in range(L)]
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
            xd = x.double()
            M[i] += xd.sum(0)
            C[i] += xd.T @ xd
        done += b
    out = []
    for i in range(L):
        m = M[i] / N
        c = C[i] / N - torch.outer(m, m)
        out.append((m, c))
    e = S / N
    v = (e[1] - e[0] ** 2).clamp(min=1e-30)
    k3 = e[2] - 3 * e[0] * e[1] + 2 * e[0] ** 3
    k4 = e[3] - 4 * e[0] * e[2] + 6 * e[0] ** 2 * e[1] - 3 * e[0] ** 4 - 3 * v * v
    return out, (e[0], v.sqrt(), k3, k4)


def relu_gauss(mp, s, g1=None, g2=None):
    t = mp / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mp * p + s * f
    va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
    if g1 is not None:                      # ⛔P4: при g=0 додається нуль
        ma = ma + s * f * (-(g1 / 6.0) * t + (g2 / 24.0) * (t * t - 1.0))
    return p, f, ma, va


def run(ws, TM, out_cum, om, oc, edge=False):
    """om/oc — підміняти m / C істинними."""
    n = ws[0].shape[0]
    L = len(ws)
    m = torch.zeros(n, dtype=torch.float64)
    C = torch.eye(n, dtype=torch.float64)
    for i, w in enumerate(ws):
        wd = w.double()
        cp = wd.T @ (C @ wd)
        s = cp.diagonal().clamp(min=1e-20).sqrt()
        mp = m @ wd
        if i == L - 1:
            if edge:
                _, _, mh, _ = out_cum[0], None, None, None
                g1 = out_cum[2] / out_cum[1] ** 3
                g2 = out_cum[3] / out_cum[1] ** 4
                return relu_gauss(mp, s, g1, g2)[2]
            return relu_gauss(mp, s)[2]
        p, f, ma, va = relu_gauss(mp, s)
        Cn = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        Cn = Cn + torch.diag(va - Cn.diagonal())
        m = TM[i][0] if om else ma
        C = TM[i][1] if oc else Cn


LAB = ["O0 наше замикання", "O1 точне m", "O2 точна C", "O3 точні m+C",
       "O3e O3 + Еджворт κ₃κ₄"]
CFG = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1)]
ERR = {k: [] for k in LAB}
Z = []
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        tr = torch.as_tensor(truth, dtype=torch.float64)
        TM, oc4 = truth_moments(ws, 770000 + 13 * q, NS)
        Z.append(float((tr * tr).mean()))
        for lab, (om, ocv, ed) in zip(LAB, CFG):
            pr = run(ws, TM, oc4, om, ocv, edge=bool(ed))
            ERR[lab].append(float(((pr - tr) ** 2).mean()))
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(Z)} · оракул N={NS}\n")

z = float(np.mean(Z))
e0 = float(np.mean(ERR[LAB[0]]))
print("=" * 96)
print(f"ОРАКУЛЬНІ МЕЖІ ДЕТЕРМІНОВАНОЇ ГІЛКИ · {len(Z)} мереж · "
      f"істина 1e9 · оракул N={NS}")
print("=" * 96)
print(f"{'рівень':>26s} {'абс. MSE':>13s} {'відн. похибка':>15s} "
      f"{'× до O0':>10s}")
for k in LAB:
    e = float(np.mean(ERR[k]))
    print(f"{k:>26s} {e:>13.4e} {100*np.sqrt(e/z):>14.4f}% {e0/e:>10.2f}")
print(f"{'🥇 ЛІДЕР (ціль)':>26s} {0.0749**2*z/1e4:>13.4e} {0.0749:>14.4f}% "
      f"{e0/(0.0749**2*z/1e4):>10.2f}")
print(f"\n⛔ P2 O0 = {e0:.4e} (cal91 дала 7.8349e-05) "
      f"{'✅' if abs(e0/7.8349e-05-1) < 0.15 else '🚨'}")
e3 = float(np.mean(ERR[LAB[3]]))
x3 = 100 * np.sqrt(e3 / z)
print(f"\n📣 ВИРОК ДВОМОМЕНТНОМУ ПРЕДСТАВЛЕННЮ:")
print(f"   O3 (точні m і C на КОЖНОМУ шарі) = {x3:.4f}%  проти лідера 0.0749%")
print(f"   {'⛔ ДВОХ МОМЕНТІВ НЕ ВИСТАЧАЄ В ПРИНЦИПІ' if x3 > 0.0749 else '✅ ДВОХ МОМЕНТІВ ДОСИТЬ — питання лише в точності'}")
print(f"   ⚠️ P3: перезапустити з N×4 — якщо O3 покращиться, межа ще нижча")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/orac91.npz",
         **{f"E{i}": np.array(ERR[k]) for i, k in enumerate(LAB)},
         Z=np.array(Z))
