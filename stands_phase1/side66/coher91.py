"""coher91 — ДВА ЧИСЛА, ЯКІ ВИРІШУЮТЬ ГІЛКУ `κ₂₁` (D1.2 і D4.4 сесії АРХІВ).

════════════════ ЧИСЛО 1: МІЖШАРОВА КОГЕРЕНТНІСТЬ ІН'ЄКЦІЇ ════════════════
Архів вивів (і я перевірив):
```
когерентні ін'єкції:  σ_∞/inj = 1/(1−G)   = 41.67
некогерентні:                   1/√(1−G²) =  4.59      G = 0.976
зміряно нині:                   42.29                 ⟹ КОГЕРЕНТНІ ✅
```
Після `κ₂₁` лишається те, чого регресія не передбачає. Якщо залишок
некогерентний — `σ_∞` падає у 9.07× сильніше, ніж рахував я.
**Міра:** `Ψ = (Σ_ℓ r_ℓ)² / (L·Σ_ℓ r_ℓ²)`, `Ψ→1` когерентно, `Ψ→1/L` — ні.

Тут міряю `Ψ` для НИНІШНЬОЇ покрокової ін'єкції, розклавши її на
шарово-СПІЛЬНУ і понейронний РОЗКИД (M10 показав 56.9% / 43.1% на шарі 31).
🔑 Якщо обидві компоненти когерентні — арифметика `1/(1−G)` чинна для обох,
і мій розрахунок «κ₂₁ ⟹ σ_∞ 2.44%» правильний.
🔑 Якщо розкид некогерентний — він накопичується як `1/√(1−G²)`, і тоді
ефективна (шкідлива) частина вже зараз мала, а κ₂₁ дає значно більше.

════════════════ ЧИСЛО 2: ЕФЕКТИВНИЙ РАНГ `Ã²` ════════════════
Архів пропонує рахувати `Ã²W` через ескіз рангу `r`: `(Ã²U)(UᵀW)`,
`2Nnr + 2n²r` замість `2Nn²` ⟹ при `r=8` у 28× дешевше (0.11% B).
Спирається на здогад «стовпці `Ã²` майже колінеарні».
⚠️ С89 уже провалив ВИПАДКОВИЙ ескіз `κ₂₁` (k=1.098 замість 2.066), але там
ескізувався сам `κ₂₁` — мала різниця великих. Тут — додатна матриця.
**Міряю еф. ранг `Ã²` по шарах прямо.**

════════════════ ПЛОМБИ ════════════════
P1 еф.ранг `Ã` (не квадрата) мусить відтворити відоме: спад до ~3 на глибині
   (С88 `tprof88`: 127.6 → 3.1) — це контроль, що рахую те саме.
P2 сума `спільна² + розкид²` = повна² (квадратура, як у M10).
P3 `Ψ` для КОНСТАНТНОГО ряду мусить дати рівно 1.0 (тотожність).
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 16
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
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
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def erank(M):
    """Ентропійний еф. ранг спектру грам-матриці стовпців M."""
    G = (M.T @ M).double()
    ev = torch.linalg.eigvalsh(G).clamp(min=0)
    p = ev / ev.sum().clamp(min=1e-300)
    p = p[p > 1e-300]
    return float(torch.exp(-(p * p.log()).sum()))


def step_inject(ws, N, seed):
    """Покрокова ін'єкція в σ: C скидається до ІСТИННОЇ щошару (CP_RESET).

    Повертає (спільна_по_шарах, розкид_по_шарах, ранг_A, ранг_A²).
    """
    g = torch.Generator().manual_seed(seed)
    n, L = ws[0].shape[0], len(ws)
    z = torch.randn(N, n, generator=g)
    com, sca, rA, rA2 = [], [], [], []
    x = z
    Cprev = torch.eye(n, dtype=torch.float64)
    for i, w in enumerate(ws):
        wd = w.double()
        h = x @ w
        # істинна σ передактивації
        hd = h.double()
        s_true = (hd.var(0, unbiased=True)).clamp(min=1e-30).sqrt()
        # σ, яку дав би ОДИН крок формули з ІСТИННОЇ Cov попереднього шару
        cp = wd.T @ (Cprev @ wd)
        s_form = cp.diagonal().clamp(min=1e-20).sqrt()
        r = torch.log(s_form / s_true)
        com.append(float(r.mean()))
        sca.append(float(r.std()))
        x = torch.relu(h)
        xd = x.double()
        xc = xd - xd.mean(0)
        Cprev = (xc.T @ xc) / (N - 1)
        if i in (1, 4, 8, 12, 16, 24, 30):
            rA.append((i, erank(xc.float())))
            q = xc * xc
            rA2.append((i, erank((q - q.mean(0)).float())))
    return np.array(com), np.array(sca), rA, rA2


COM, SCA = [], []
RA, RA2 = {}, {}
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])) for i in range(32)]
        c, s, ra, ra2 = step_inject(ws, NS, 313000 + 5 * q)
        COM.append(c); SCA.append(s)
        for i, v in ra:
            RA.setdefault(i, []).append(v)
        for i, v in ra2:
            RA2.setdefault(i, []).append(v)
        if q % 4 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)
COM, SCA = np.array(COM), np.array(SCA)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(COM)} · семплів {NS}\n")


def psi(r):
    """Ψ = (Σr)²/(L·Σr²).  1 = когерентно, 1/L = некогерентно."""
    r = np.asarray(r, dtype=np.float64)
    return float(r.sum() ** 2 / (len(r) * (r * r).sum()))


L = COM.shape[1]
print("=" * 92)
print(f"ЧИСЛО 1: КОГЕРЕНТНІСТЬ ПОКРОКОВОЇ ІН'ЄКЦІЇ · {len(COM)} мереж · N={NS}")
print("=" * 92)
print(f"  Ψ = 1 когерентно · Ψ = 1/{L} = {1/L:.4f} некогерентно")
pc = np.mean([psi(COM[q]) for q in range(len(COM))])
print(f"\n  Ψ(спільна по шару) = {pc:.4f}")
print(f"  ⛔P3 Ψ(константа)  = {psi(np.ones(L)):.4f} (мусить бути 1.0000) "
      f"{'✅' if abs(psi(np.ones(L))-1) < 1e-12 else '🚨'}")
print(f"\n  покрокова ін'єкція: |спільна| сер. {100*np.abs(COM).mean():.4f}% · "
      f"розкид сер. {100*SCA.mean():.4f}%")
print(f"  знак спільної: додатних шарів {100*(COM > 0).mean():.1f}% "
      f"⟹ {'ОДНОБІЧНА (когерентна)' if (COM > 0).mean() > 0.8 or (COM > 0).mean() < 0.2 else 'ЗНАКОЗМІННА'}")
G = 0.976
print(f"\n  теорія: когерентно ×{1/(1-G):.2f} · некогерентно ×{1/np.sqrt(1-G*G):.2f}")
print(f"  ⟹ спільна ін'єкція {100*np.abs(COM).mean():.4f}% × 41.67 = "
      f"{100*np.abs(COM).mean()*41.67:.2f}%  (M10 дала 10.19% спільної на ш.31)")
print(f"  ⟹ розкид {100*SCA.mean():.4f}% × 41.67 = {100*SCA.mean()*41.67:.2f}% "
      f"· × 4.59 = {100*SCA.mean()*4.59:.2f}%  (M10 дала 8.86% розкиду)")

print("\n" + "=" * 92)
print("ЧИСЛО 2: ЕФЕКТИВНИЙ РАНГ — ЧИ ПРАЦЮЄ ЕСКІЗ `Ã²W`")
print("=" * 92)
print(f"{'шар':>6s}{'еф.ранг Ã':>14s}{'еф.ранг Ã²':>14s}{'ціна ескізу r=eff':>20s}")
for i in sorted(RA):
    a, a2 = np.mean(RA[i]), np.mean(RA2[i])
    cost = 32 * (2 * NS * 256 * a2 + 2 * 256 * 256 * a2) / 2.72e11
    print(f"{i:>6d}{a:>14.2f}{a2:>14.2f}{100*cost:>19.3f}%")
print(f"\n⛔P1 еф.ранг Ã на глибині ≈ {np.mean(RA[30]):.1f} "
      f"(С88 tprof88 дала 3.1 для ПЕРЕДактивацій) ")
print(f"\n📣 ВИРОК ПО ЕСКІЗУ: {'✅ Ã² низькорангова — ескіз можливий' if np.mean(RA2[30]) < 20 else '⛔ Ã² ПОВНОРАНГОВА — ескіз не працює'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/coher91.npz",
         COM=COM, SCA=SCA,
         RA=np.array([[i, np.mean(RA[i])] for i in sorted(RA)]),
         RA2=np.array([[i, np.mean(RA2[i])] for i in sorted(RA2)]))
