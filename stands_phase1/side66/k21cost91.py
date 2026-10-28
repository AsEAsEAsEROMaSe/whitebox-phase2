"""k21cost91 — КРИВА «ВИГРАШ κ₂₁ проти ЧИСЛА СЕМПЛІВ `N_T`». ГЕЙТ ОКУПНОСТІ.

════════════════ ЧОМУ САМЕ ЦЕ ЧИСЛО ВИРІШУЄ ════════════════
`k21_91` (оракул, N=8192): κ₂₁ знижує ін'єкцію у **4.95×**, чесний SPLIT
**3.82×**, з κ₄ **5.99×**. Потрібно (psi91) **8.20×**.
Але `T_jk = E[hc_j²hc_k] = (hc∘hc)ᵀhc/N` коштує матмул `2·N·256²` НА ШАР:
```
N_T = 8192 → 32·2·8192·65536 = 3.44e10 = 12.6% B     ⛔
N_T = 2048 →                    8.59e9 =  3.2% B     ⛔
N_T =  512 →                    2.15e9 = 0.79% B     ≈ гейт k=2.0 (0.811%)
N_T =  256 →                    1.07e9 = 0.40% B     ≈ гейт k=1.4
```
🔑 Пропагацію семплів НЕ рахую: сабміт і так жене МК ⟹ `h` уже є.
⟹ **питання рівно одне: скільки κ₂₁ дає при `N_T` = 256…2048.**
Шумна `T` ⟹ оптимальне `λ < 1` (усадка). Тому скан `λ` обов'язковий.

Додатково: `psi91` показав, що **77% внеску дають шари 16–30** ⟹ варіант
`DEEP` — поправка лише на другій половині, ціна вдвічі менша.

════════════════ ПЛОМБИ ════════════════
P1 `N_T → ∞` мусить відтворити `k21_91`: ×4.9±0.3 при λ=1.
P2 `λ=0` ≡ база побітово для всіх `N_T`.
P3 ЧУЖА `T` (перемішані нейрони) при кожному `N_T`: оптимум мусить бути λ=0.
P4 СІДИ РОЗДІЛЕНІ ЖОРСТКО: істина (`Ch`, `Ĉov_a`) — один потік семплів,
   `T` — ІНШИЙ, незалежний. Інакше виграш завищено (у k21_91 було 4.95 проти
   чесних 3.82 саме через спільний сід).
P5 усадка: оптимальне `λ(N_T)` мусить МОНОТОННО падати з ростом шуму.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) при N_T=512 виграш ≤ ×1.8                       шанс 65%
 (б) λ*(512) ≤ 0.5                                    шанс 70%
 (в) DEEP при половинній ціні дасть ≥70% виграшу FULL шанс 55%
 (г) жодна точка кривої не пройде гейт окупності      шанс 75%
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

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
NTRUE = int(sys.argv[2]) if len(sys.argv) > 2 else 32768
NTS = (256, 512, 1024, 2048, 4096, 16384)
LAM = (0.0, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0)
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
B = 2.72e11
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


def base_cov(mh, Ch):
    s = Ch.diagonal().clamp(min=1e-20).sqrt()
    t = mh / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mh * p + s * f
    va = ((mh * mh + s * s) * p + mh * s * f - ma * ma).clamp(min=0.0)
    Cn = Ch * torch.outer(p, p) + (0.5 * (Ch * Ch)) * torch.outer(f / s, f / s)
    return ma, Cn, va, s, p, f


R = {}
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])).double()
              for i in range(32)]
        gT = torch.Generator().manual_seed(120000 + 23 * q)      # істина
        gO = torch.Generator().manual_seed(950000 + 29 * q)      # ⛔інший потік
        gp = torch.Generator().manual_seed(7700 + q)
        xt = torch.randn(NTRUE, 256, generator=gT)
        xo = torch.randn(max(NTS), 256, generator=gO)
        for k in range(32):
            ht = (xt @ ws[k].float()).double()
            ho = (xo @ ws[k].float()).double()
            at = ht.clamp(min=0.0)
            if k < 31:
                mh = ht.mean(0)
                hc = ht - mh
                Ch = (hc.T @ hc) / NTRUE
                ac = at - at.mean(0)
                CAt = (ac.T @ ac) / NTRUE
                ma, Cn, va, s, p, f = base_cov(mh, Ch)
                wd = ws[k + 1]
                dt = (wd.T @ (CAt @ wd)).diagonal().clamp(min=1e-30)
                W = torch.outer(f / s, p)

                def rho(D):
                    C = Cn + D
                    C = C + torch.diag(va - C.diagonal())
                    dp = (wd.T @ (C @ wd)).diagonal().clamp(min=1e-30)
                    return 0.5 * torch.log(dp / dt)

                pm = torch.randperm(256, generator=gp)
                for nt in NTS:
                    hs = ho[:nt]
                    hcs = hs - hs.mean(0)
                    T3 = ((hcs * hcs).T @ hcs) / nt
                    A = W * T3
                    D3 = 0.5 * (A + A.T)
                    Af = W * T3[pm][:, pm]
                    D3f = 0.5 * (Af + Af.T)
                    for lam in LAM:
                        R.setdefault(('своя', nt, lam), []).append(rho(lam * D3))
                        R.setdefault(('чужа', nt, lam), []).append(rho(lam * D3f))
                        if k >= 16:
                            R.setdefault(('deep', nt, lam), []).append(rho(lam * D3))
                        else:
                            R.setdefault(('deep', nt, lam), []).append(rho(D3 * 0))
            xt = at.float()
            xo = ho.clamp(min=0.0).float()
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с · мереж {NNET} · істина N={NTRUE}")


def sc(key):
    return float(torch.stack(R[key]).std(1).mean())


base = sc(('своя', NTS[0], 0.0))
p2 = max(abs(sc((nm, nt, 0.0)) - base) for nm in ('своя', 'чужа') for nt in NTS)
print(f"⛔P2 λ=0 ≡ база: max|Δ| = {p2:.3e} {'✅' if p2 < 1e-14 else '🚨'}")
print(f"\n  база (λ=0) розкид ін'єкції = {100*base:.4f}% · "
      f"потрібно ×8.20 ⟹ {100*base/8.20:.4f}%\n")

print("=" * 100)
print("ВИГРАШ κ₂₁ ПРОТИ ЧИСЛА СЕМПЛІВ ДЛЯ `T` · оптимум по λ · роздільні сіди")
print("=" * 100)
print(f"{'N_T':>7s}{'ціна FULL':>12s}{'своя ×':>11s}{'λ*':>7s}"
      f"{'⛔чужа ×':>11s}{'λ*ч':>7s}{'DEEP ×':>10s}{'ціна DEEP':>12s}")
LOG = {}
for nt in NTS:
    cost = 32 * 2 * nt * 65536 / B
    bo = min((sc(('своя', nt, l)), l) for l in LAM)
    bf = min((sc(('чужа', nt, l)), l) for l in LAM)
    bd = min((sc(('deep', nt, l)), l) for l in LAM)
    LOG[nt] = (base / bo[0], bo[1], base / bd[0], cost)
    print(f"{nt:>7d}{100*cost:>11.3f}%{base/bo[0]:>11.3f}{bo[1]:>7.2f}"
          f"{base/bf[0]:>11.3f}{bf[1]:>7.2f}{base/bd[0]:>10.3f}"
          f"{100*cost/2:>11.3f}%")
print(f"\n⛔P1 N_T=16384 дає ×{LOG[16384][0]:.2f} проти k21_91 ×3.82 (SPLIT) "
      f"{'✅' if abs(LOG[16384][0]-3.82) < 0.9 else '⚠️ розбіжність'}")
lams = [LOG[nt][1] for nt in NTS]
print(f"⛔P5 усадка λ*(N_T) = {lams} "
      f"{'✅ монотонна' if all(lams[i] <= lams[i+1]+1e-9 for i in range(len(lams)-1)) else '⚠️ немонотонна'}")

print("\n" + "=" * 100)
print("ГЕЙТ ОКУПНОСТІ: σ(h_31) розкид 8.20% → X% · дозволена ціна з M63")
print("=" * 100)
print(f"{'N_T':>7s}{'σ розкид':>12s}{'ціна':>10s}{'дозволено при k':>18s}{'вирок':>10s}")
GATE = ((1.2, 0.162), (1.35, 0.284), (2.0, 0.811), (31.2, 7.542))
for nt in NTS:
    g, _, gd, cost = LOG[nt]
    sig = 8.20 / g
    ok = [k for k, allw in GATE if cost * 100 <= allw]
    print(f"{nt:>7d}{sig:>11.3f}%{100*cost:>9.3f}%"
          f"{(f'k≥{min(ok):.2f}' if ok else 'жодного'):>18s}"
          f"{('⚠️' if ok else '⛔'):>10s}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/k21cost91.npz",
         **{f"{nm}_{nt}_{lam}": torch.stack(R[(nm, nt, lam)]).numpy()
            for nm in ('своя', 'чужа', 'deep') for nt in NTS for lam in LAM})
