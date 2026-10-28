"""ladder91 — L10 НА ГРАДУЙОВАНОМУ ПРИЛАДІ + КАРТА ВИМОГ ДО ТОЧНОСТІ σ,γ.

════════════════ ЩО ТАКЕ L10 (з `ladder87`, С87, 14 мереж) ════════════════
```
m_{ℓ+1} = Еджворт-ReLU( m_ℓ @ W_ℓ ,  σ_оракул(ℓ) ,  γ₃,γ₄ оракул(ℓ) )
```
🔑 ЖОДНОЇ МАТРИЦІ. `C` існує рівно щоб видати вектор `σ` (`L8 ≡ L6` до 4-го
знака) ⟹ якщо `σ` є з іншого джерела, `C` не потрібна взагалі.
Ціна: 32 матвектори = **4.2e6 флопів = 0.0015% B** (проти 0.61% B за `C`).
Це пояснює профіль лідера: у нього нема `C`, бо вона й не потрібна.

С87 дає `L10 = 0.0605%` ⟹ ×5.788 до рекорду, ЛІДЕРА Б'Є (0.0749%).

════════════════ ЧОГО НІХТО НЕ МІРЯВ ════════════════
`L10` бере `σ,γ` ОРАКУЛЬНІ. Звідси вивели «ТЗ: σ до 1%, γ до 3%» — але це
**виведено, а не зміряно**. Питання, що вирішує долю всієї фази 2:
**наскільки L10 чутливий до похибки `σ` і `γ`?**
```
седлова дає покроково 2–5× по позадіагоналі ⟹ σ 14.8% → 3–7%
κ₂₁ дає ~1.6×
разом ~×8 при «потрібних ×15» — АЛЕ ×15 не зміряне
```
Якщо крива полога — седлова+maxent вистачає. Якщо крута — клас мертвий.

════════════════ СХЕМА ════════════════
`σ_ужито = σ_іст·(1+ε·ξ)`, `γ_ужито = γ_іст·(1+δ·ζ)`, `ξ,ζ ~ N(0,1)` понейронно
(мультиплікативний шум — саме такий дає будь-який наближений метод).
Свіп `ε ∈ {0,.01,.03,.05,.10,.15}` × `δ ∈ {0,.10,.30,1.0}`.

════════════════ ПЛОМБИ ════════════════
P1 `ε=0, δ=0` мусить відтворити L10 ≈ 0.0605% (С87, інший прилад ⟹ ±20%).
P2 `ε=0.148, δ=1` мусить впасти до рівня нашого замикання (~0.79–0.94%):
   це режим «σ як у ланцюга, γ нема».
P3 сіди оракула і цілі РОЗДІЛЕНІ (ціль — офіційна істина 1e9).
P4 ⛔КОНТРОЛЬ: `σ,γ` з ЧУЖОЇ мережі при ε=δ=0 — мусить бути катастрофа.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) P1 зійдеться в межах 20%                       — шанс 55%
 (б) крива по ε буде КРУТОЮ (σ до 3% вже втрачає половину виграшу) — 60%
 (в) чутливість до γ НИЖЧА, ніж до σ                — шанс 70%
 🎯 шанс, що при ε=5% лишається ≥×2 до рекорду: **35%**
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
EPS = (0.0, 0.01, 0.03, 0.05, 0.10, 0.15)
DEL = (0.0, 0.10, 0.30, 1.0)
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
    """σ, γ₃, γ₄ ПЕРЕДАКТИВАЦІЇ на кожному шарі (окремий сід від цілі)."""
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
        out.append((s, k3 / s ** 3, k4 / s ** 4))
    return out


def L10(ws, ora, eps, dlt, gen):
    """m рекурсується САМ; σ,γ — з оракула, зіпсовані відносним шумом."""
    n, L = ws[0].shape[0], len(ws)
    m = torch.zeros(n, dtype=torch.float64)
    for i, w in enumerate(ws):
        s0, g1, g2 = ora[i]
        s = s0 * (1.0 + eps * torch.randn(n, generator=gen, dtype=torch.float64)) \
            if eps > 0 else s0
        s = s.clamp(min=1e-20)
        if dlt > 0:
            g1 = g1 * (1.0 + dlt * torch.randn(n, generator=gen, dtype=torch.float64))
            g2 = g2 * (1.0 + dlt * torch.randn(n, generator=gen, dtype=torch.float64))
        mp = m @ w.double()
        t = mp / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        m = mp * p + s * f + s * f * (-(g1 / 6.0) * t + (g2 / 24.0) * (t * t - 1.0))
    return m


G = {}
Z = []
prev = None
FOR = []
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        tr = torch.as_tensor(truth, dtype=torch.float64)
        Z.append(float((tr * tr).mean()))
        ora = oracle(ws, 440000 + 7 * q, NORA)
        for e in EPS:
            for d in DEL:
                gen = torch.Generator().manual_seed(31337 + q)
                pr = L10(ws, ora, e, d, gen)
                G.setdefault((e, d), []).append(float(((pr - tr) ** 2).mean()))
        if prev is not None:                      # ⛔ P4 чужий оракул
            gen = torch.Generator().manual_seed(1)
            FOR.append(float(((L10(ws, prev, 0.0, 0.0, gen) - tr) ** 2).mean()))
        prev = ora
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(Z)}\n")

z = float(np.mean(Z))
REC = 1.86762e-07
V = 2.13e-06                     # MSE чистої МК на цьому зрізі


def score(e_abs):
    """бал через оптимальний бленд із МК (незалежні джерела)."""
    return 0.1 * (V * e_abs) / (V + e_abs)


print("=" * 100)
print(f"L10 (БЕЗ МАТРИЦЬ) І КАРТА ВИМОГ ДО ТОЧНОСТІ · {len(Z)} мереж · "
      f"істина 1e9 · оракул N={NORA}")
print("=" * 100)
print(f"{'ε(σ) \\ δ(γ)':>12s}" + "".join(f"{d:>16.2f}" for d in DEL))
for e in EPS:
    row = f"{e:>12.3f}"
    for d in DEL:
        a = float(np.mean(G[(e, d)]))
        row += f"{100*np.sqrt(a/z):>10.4f}% ×{REC/score(a):>4.2f}"
    print(row)
print(f"\n🥇 ЛІДЕР 0.0749%   ·   наше замикання ~0.79%   ·   рекорд {REC:.4e}")
a00 = float(np.mean(G[(0.0, 0.0)]))
print(f"\n⛔ P1 L10 при ε=δ=0: {100*np.sqrt(a00/z):.4f}%  (С87 дала 0.0605%) "
      f"{'✅' if abs(np.sqrt(a00/z)*100/0.0605-1) < 0.35 else '🚨'}")
print(f"⛔ P2 ε=0.15, δ=1.0: {100*np.sqrt(float(np.mean(G[(0.15,1.0)]))/z):.4f}% "
      f"(мусить впасти до рівня замикання ~0.8%)")
if FOR:
    print(f"⛔ P4 ЧУЖИЙ оракул: {100*np.sqrt(float(np.mean(FOR))/z):.4f}% "
          f"{'✅ катастрофа, як і має бути' if np.mean(FOR) > 10*a00 else '🚨'}")
print(f"\n📣 КАРТА ВИМОГ: щоб лишитись нижче лідера (0.0749%), потрібно")
for d in DEL:
    ok = [e for e in EPS if np.sqrt(float(np.mean(G[(e, d)])) / z) * 100 < 0.0749]
    print(f"   при δ(γ)={d:.2f} ⟹ ε(σ) ≤ "
          f"{max(ok)*100 if ok else 0:.1f}%" + ("" if ok else "  (недосяжно)"))
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/ladder91.npz",
         **{f"g_{e}_{d}": np.array(G[(e, d)]) for e in EPS for d in DEL},
         Z=np.array(Z), FOR=np.array(FOR))
