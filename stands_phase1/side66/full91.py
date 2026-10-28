"""full91 — НАСКРІЗНИЙ ВИМІР: замикання + κ₂₁ проти ОФІЦІЙНОЇ ІСТИНИ 1e9.

════════════════ ЧОМУ НЕ МОЖНА БУЛО ЗУПИНИТИСЬ НА ІН'ЄКЦІЇ ════════════════
`k21cost91` дав виграш у РОЗКИДІ `σ`. Щоб назвати бал, я переклав його в
похибку замикання ЧЕРЕЗ карту `ladder91` (i.i.d. мультиплікативний шум).
🚨 Це переклад ДІЛЕННЯМ — рівно те, на чому проєкт горів чотири рази
(×4.61, «fnp 30.6 мкс», «1/r=15.66», «λ=3.7×»). Закон: **КАЛІБРУВАЛЬНУ
КОНСТАНТУ МІРЯТИ, А НЕ ВИВОДИТИ.** Тому — прямий наскрізний прогін.

Крім того карта `ladder91` псувала `σ` i.i.d., а справжня похибка `σ`
СТРУКТУРОВАНА (57% шарово-спільна, яку з'їдає `α`) ⟹ переклад міг бути
хибним в обидва боки.

════════════════ ЩО МІРЯЄТЬСЯ ════════════════
Повне замикання (як у сабміті), але на кожному шарі `Cov(a) += λ·ΔCov³`,
де `T` береться з НЕЗАЛЕЖНОГО потоку МК розміру `N_T`. Вихід — `m` шару 31,
ціль — `final_means` офіційного датасету (1e9 семплів, шум 0.0023%).

🔒 ГРАДУЮВАННЯ: `λ=0` мусить відтворити `cal91` = **7.8349e-05** абсолютної
MSE і **0.928%** відносної. Без цього жодне число нижче не варте нічого.

════════════════ ПЛОМБИ ════════════════
P1 `λ=0` ≡ зміряне замикання `cal91`: 7.8349e-05 / 0.928%.
P2 ⛔ЧУЖА `T` (перемішані нейрони) — оптимум мусить бути при `λ→0`.
P3 сіди `T` і істини роздільні за побудовою (істина — офіційна, не МК).
P4 масштабування `N_T`: монотонність виграшу.
P5 ЦІНА рахується явно і зіставляється з гейтом окупності.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) λ=0 відтворить cal91 у межах 2%                        шанс 85%
 (б) оракул (N_T=16384) дасть замикання 0.35–0.60%          шанс 55%
 (в) при N_T=512 виграш на балі буде k < 1.2 ⟹ мертво       шанс 75%
 (г) ⛔ЧУЖА покаже оптимум λ=0                              шанс 90%
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NTS = tuple(int(v) for v in (sys.argv[2].split(',') if len(sys.argv) > 2
                             else ('512', '2048', '16384')))
LAM = (0.0, 0.25, 0.5, 0.7, 0.85, 1.0)
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
B = 2.72e11
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "mini"   # cal91 градуйовано на mini
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/" + SPLIT)


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


def closure(ws, T3s, lam, perm=None):
    """Замикання сабміту + λ·ΔCov³. T3s — список (31,) або None."""
    n = 256
    m = torch.zeros(n, dtype=torch.float64)
    C = torch.eye(n, dtype=torch.float64)
    for i, w in enumerate(ws):
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp(min=1e-20).sqrt()
        mp = m @ w
        t = mp / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        ma = mp * p + s * f
        if i == 31:
            return ma
        va = ((mp * mp + s * s) * p + mp * s * f - ma * ma).clamp(min=0.0)
        Cn = cp * torch.outer(p, p) + (0.5 * (cp * cp)) * torch.outer(f / s, f / s)
        if lam > 0.0 and T3s is not None:
            T = T3s[i]
            if perm is not None:
                T = T[perm[i]][:, perm[i]]
            A = torch.outer(f / s, p) * T
            Cn = Cn + lam * 0.5 * (A + A.T)
        C = Cn + torch.diag(va - Cn.diagonal())
        m = ma


G = {}
ZT = []
t0 = time.time()
with torch.no_grad():
    for q, (wnp, truth) in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])).double()
              for i in range(32)]
        tr = torch.as_tensor(truth, dtype=torch.float64)
        ZT.append(float((tr * tr).mean()))
        gO = torch.Generator().manual_seed(310000 + 31 * q)
        gp = torch.Generator().manual_seed(5500 + q)
        x = torch.randn(max(NTS), 256, generator=gO)
        T3 = {nt: [] for nt in NTS}
        for k in range(31):
            h = (x @ ws[k].float()).double()
            for nt in NTS:
                hs = h[:nt]
                hc = hs - hs.mean(0)
                T3[nt].append(((hc * hc).T @ hc) / nt)
            x = h.clamp(min=0.0).float()
        perm = [torch.randperm(256, generator=gp) for _ in range(31)]
        for nt in NTS:
            for lam in LAM:
                pr = closure(ws, T3[nt], lam)
                G.setdefault((nt, lam), []).append(float(((pr - tr) ** 2).mean()))
                if lam > 0.0:
                    pf = closure(ws, T3[nt], lam, perm=perm)
                    G.setdefault((f"ч{nt}", lam), []).append(
                        float(((pf - tr) ** 2).mean()))
        if q % 8 == 0:
            print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

z = float(np.mean(ZT))
print(f"\n  час {time.time()-t0:.0f} с · мереж {NNET}\n")
a0 = float(np.mean(G[(NTS[0], 0.0)]))
print("=" * 96)
print(f"ЗАМИКАННЯ + κ₂₁ ПРОТИ ОФІЦІЙНОЇ ІСТИНИ 1e9 · {NNET} мереж")
print("=" * 96)
print(f"🔒 P1 ГРАДУЮВАННЯ λ=0: абс MSE {a0:.6e} проти cal91 7.8349e-05 "
      f"({100*(a0/7.8349e-05-1):+.2f}%) · відн {100*np.sqrt(a0/z):.4f}% "
      f"проти 0.928% {'✅' if abs(a0/7.8349e-05-1) < 0.05 else '🚨'}\n")
print(f"{'N_T':>7s}{'ціна':>9s}" + "".join(f"{'λ='+str(l):>15s}" for l in LAM))
for nt in NTS:
    cost = 100 * 31 * 2 * nt * 65536 / B
    row = f"{nt:>7d}{cost:>8.3f}%"
    for lam in LAM:
        a = float(np.mean(G[(nt, lam)]))
        row += f"{100*np.sqrt(a/z):>9.4f}% ×{a0/a:>3.2f}"
    print(row)
print(f"\n{'⛔ЧУЖА':>7s}{'':>9s}" + "".join(f"{'λ='+str(l):>15s}" for l in LAM))
for nt in NTS:
    row = f"{nt:>7d}{'':>9s}"
    for lam in LAM:
        if lam == 0.0:
            row += f"{100*np.sqrt(a0/z):>9.4f}% ×1.00"
            continue
        a = float(np.mean(G[(f"ч{nt}", lam)]))
        row += f"{100*np.sqrt(a/z):>9.4f}% ×{a0/a:>3.2f}"
    print(row)

print("\n" + "=" * 96)
print("📣 ВИРОК — ГЕЙТ ОКУПНОСТІ")
print("=" * 96)
GATE = ((1.2, 0.162), (1.35, 0.284), (2.0, 0.811), (31.2, 7.542))
for nt in NTS:
    cost = 100 * 31 * 2 * nt * 65536 / B
    bl, bk = min(((float(np.mean(G[(nt, l)])), l) for l in LAM))
    k = a0 / bl
    need = [g for g, allw in GATE if cost <= allw]
    req = min(need) if need else 99.9
    print(f"  N_T={nt:>6d}: замикання {100*np.sqrt(bl/z):.4f}% при λ={bk:.2f} "
          f"⟹ k(замикання)={k:.3f} · ціна {cost:.3f}% ⟹ треба k≥{req:.2f} "
          f"{'✅ ОКУПАЄТЬСЯ' if k >= req else '⛔ МЕРТВО'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/full91.npz",
         **{f"g_{k0}_{k1}": np.array(v) for (k0, k1), v in G.items()},
         Z=np.array(ZT))
