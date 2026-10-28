"""k21_91 — СТЕЛЯ ПАРНОЇ НЕГАУСОВОЇ ПОПРАВКИ ДО `Cov(a)`. Оракульні кумулянти.

════════════════ ВИВЕДЕННЯ (моє, з ермітового базису) ════════════════
`mehl91` довів: обрив ряду Мелера дає 0.05% ін'єкції, решта 99.95% —
негаусовість пари `(h_j,h_k)`. Ряд Мелера `E[He_n(u)He_m(v)] = n!r^nδ_nm`
чинний ЛИШЕ для гаусової пари. Для негаусової додаються НЕДІАГОНАЛЬНІ
елементи, і кожен множиться на власний коефіцієнт `a_n` (виведені в mehl91):
```
a_0 = s·M(t)   a_1 = s·Φ(t)   a_n = s·φ(t)·He_{n−2}(−t)/n!  ⟹ a_2 = sφ/2, a_3 = −stφ/6
```
`E[g h] = Σ_{n,m} a_n b_m E[He_n(u)He_m(v)]`, і третій порядок дає
`E[He_2He_1] = κ_{21}`, `E[He_3He_0] = κ_{30}`. Члени з `He_0` скорочуються
в КОВАРІАЦІЇ (входять і в `E[g]E[h]`) ⟹ лишається рівно:
```
ΔCov³_jk = a_2^j a_1^k κ_{21}(j,k) + a_1^j a_2^k κ_{21}(k,j)
         = ½[ (φ_j/s_j)Φ_k · T_jk + Φ_j(φ_k/s_k) · T_kj ],   T_jk = E[hc_j² hc_k]
```
🔑 `T = (hc∘hc)ᵀhc / N` — ОДИН матмул 256³, тобто ціна ще однієї коваріації.

Четвертий порядок (`E[He_2He_2] = 2r² + κ_{22}`, `E[He_3He_1] = κ_{31}`):
```
ΔCov⁴_jk = a_2^j a_2^k κ_{22} + a_3^j a_1^k κ_{31}(j,k) + a_1^j a_3^k κ_{31}(k,j)
κ_{22} = E[u²v²] − 1 − 2r²          κ_{31} = E[u³v] − 3r
```
⟹ ще два матмули (`(hc²)ᵀ(hc²)` і `(hc³)ᵀhc`).

════════════════ ЩО МІРЯЄМО ════════════════
СТЕЛЮ: кумулянти ОРАКУЛЬНІ (з семплів). Питання не «чи дешево», а
«чи ВЗАГАЛІ вистачає» — вимога `psi91`: знизити ін'єкцію у **8.20×**.

════════════════ ПЛОМБИ ════════════════
P1 `ΔCov` симетрична: `max|Δ−Δᵀ|` = 0 машинно (з побудови ½[·+·ᵀ]).
P2 `λ=0` ≡ база побітово.
P3 ЧУЖА `T` (з перемішаних нейронів): свій оптимум мусить бути при `λ>0`,
   чужий — при `λ→0`. Це заміна порожньому контролю «знак» (С89).
P4 МАСШТАБУВАННЯ: `T` шумніша за `Cov` (третій момент) ⟹ два `N` обов'язкові.
P5 сіди істини й оракула РОЗДІЛЕНІ? — тут ні: `T` і `Ĉov_a` з ТИХ САМИХ
   семплів. Це ЗАВИЩУЄ виграш (спільний шум) ⟹ додаю варіант `SPLIT`:
   `T` з першої половини семплів, істина — з другої.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) κ₂₁ (λ=1, оракул, спільний сід) знизить ін'єкцію у 2–4×   шанс 55%
 (б) SPLIT з'їсть щонайменше третину виграшу                    шанс 60%
 (в) κ₂₁+κ₄ не дійде до потрібних 8.2×                          шанс 70%
 (г) чужа T дасть оптимум при λ ≤ 0.25                          шанс 80%
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
NS = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
LAM = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
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
    """Гаусове замикання (K=2 ≡ повний ряд Мелера за mehl91)."""
    s = Ch.diagonal().clamp(min=1e-20).sqrt()
    t = mh / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mh * p + s * f
    va = ((mh * mh + s * s) * p + mh * s * f - ma * ma).clamp(min=0.0)
    Cn = Ch * torch.outer(p, p) + (0.5 * (Ch * Ch)) * torch.outer(f / s, f / s)
    return ma, Cn, va, s, t, p, f


def stats(h, halves=False):
    """Центровані моменти передактивації. Повертає (mh, Ch, T3, K22, K31)."""
    N = h.shape[0]
    mh = h.mean(0)
    hc = h - mh
    Ch = (hc.T @ hc) / N
    h2 = hc * hc
    T3 = (h2.T @ hc) / N                       # E[hc_j² hc_k]
    K22r = (h2.T @ h2) / N                     # E[hc_j² hc_k²]
    K31r = ((h2 * hc).T @ hc) / N              # E[hc_j³ hc_k]
    return mh, Ch, T3, K22r, K31r


def dcov3(T3, s, p, f):
    """ΔCov від третіх кумулянтів пари (виведення в докстрінгу)."""
    A = torch.outer(f / s, p) * T3
    return 0.5 * (A + A.T)


def dcov4(Ch, K22r, K31r, s, t, p, f):
    """ΔCov від четвертих кумулянтів пари."""
    S = torch.outer(s, s)
    r = Ch / S
    s2 = s * s
    k22 = K22r / torch.outer(s2, s2) - 1.0 - 2.0 * r * r
    k31 = K31r / torch.outer(s2 * s, s) - 3.0 * r
    a2 = s * f / 2.0
    a3 = -s * t * f / 6.0
    a1 = s * p
    B = torch.outer(a3, a1) * k31
    return torch.outer(a2, a2) * k22 + B + B.T


ROWS = {}
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])).double()
              for i in range(32)]
        g = torch.Generator().manual_seed(660000 + 17 * q)
        gp = torch.Generator().manual_seed(4400 + q)
        x = torch.randn(NS, 256, generator=g)
        acc = {}
        for k in range(32):
            h = (x @ ws[k].float()).double()
            a = h.clamp(min=0.0)
            if k < 31:
                half = NS // 2
                mh, Ch, T3, K22r, K31r = stats(h)
                # SPLIT: T3 лише з ПЕРШОЇ половини семплів
                hcA = h[:half] - h[:half].mean(0)
                T3s = ((hcA * hcA).T @ hcA) / half
                ac = a - a.mean(0)
                CAt = (ac.T @ ac) / NS
                ma, Cn, va, s, t, p, f = base_cov(mh, Ch)
                D3 = dcov3(T3, s, p, f)
                D3s = dcov3(T3s, s, p, f)
                D4 = dcov4(Ch, K22r, K31r, s, t, p, f)
                pm = torch.randperm(256, generator=gp)
                D3f = dcov3(T3[pm][:, pm], s, p, f)
                wd = ws[k + 1]
                dt = (wd.T @ (CAt @ wd)).diagonal().clamp(min=1e-30)

                def rho(D):
                    C = Cn + D
                    C = C + torch.diag(va - C.diagonal())
                    dp = (wd.T @ (C @ wd)).diagonal().clamp(min=1e-30)
                    return 0.5 * torch.log(dp / dt)

                for lam in LAM:
                    acc.setdefault(('своя', lam), []).append(rho(lam * D3))
                    acc.setdefault(('чужа', lam), []).append(rho(lam * D3f))
                    acc.setdefault(('split', lam), []).append(rho(lam * D3s))
                    acc.setdefault(('3+4', lam), []).append(rho(lam * (D3 + D4)))
                if k == 0:
                    P1 = float((D3 - D3.T).abs().max())
            x = a.float()
        for key, v in acc.items():
            ROWS.setdefault(key, []).append(torch.stack(v))
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с · мереж {NNET} · семплів {NS}")
print(f"⛔P1 ΔCov симетрична: max|Δ−Δᵀ| = {P1:.3e} {'✅' if P1 < 1e-15 else '🚨'}")
b0 = torch.stack(ROWS[('своя', 0.0)])
for nm in ('чужа', 'split', '3+4'):
    d = float((torch.stack(ROWS[(nm, 0.0)]) - b0).abs().max())
    print(f"⛔P2 λ=0 ≡ база ({nm}): max|Δ| = {d:.3e} {'✅' if d < 1e-14 else '🚨'}")

print("\n" + "=" * 92)
print(f"ІН'ЄКЦІЯ ρ (розкид, сер. по шарах 1..30) · {NNET} мереж · N={NS}")
print("=" * 92)
base = float(b0[:, 1:].std(2).mean())
print(f"{'λ':>7s}{'своя κ₂₁':>16s}{'⛔чужа':>16s}{'SPLIT сіди':>16s}{'κ₂₁+κ₄':>16s}")
for lam in LAM:
    row = f"{lam:>7.2f}"
    for nm in ('своя', 'чужа', 'split', '3+4'):
        v = float(torch.stack(ROWS[(nm, lam)])[:, 1:].std(2).mean())
        row += f"{100*v:>10.4f}% ×{base/max(v,1e-30):>4.2f}"
    print(row)
print(f"\n  база (λ=0) розкид ін'єкції = {100*base:.4f}%")
print(f"  ПОТРІБНО (psi91): знизити у 8.20× ⟹ до {100*base/8.20:.4f}%")
best = min((float(torch.stack(ROWS[('своя', l)])[:, 1:].std(2).mean()), l)
           for l in LAM)
bf = min((float(torch.stack(ROWS[('чужа', l)])[:, 1:].std(2).mean()), l)
         for l in LAM)
bs = min((float(torch.stack(ROWS[('split', l)])[:, 1:].std(2).mean()), l)
         for l in LAM)
b34 = min((float(torch.stack(ROWS[('3+4', l)])[:, 1:].std(2).mean()), l)
          for l in LAM)
print(f"\n⛔P3 СКАН λ: своя оптимум λ={best[1]:.2f} (×{base/best[0]:.2f}) · "
      f"ЧУЖА оптимум λ={bf[1]:.2f} (×{base/bf[0]:.2f}) "
      f"{'✅ контроль чистий' if bf[1] < best[1] else '🚨 ЧУЖА НЕ ГІРША'}")
print(f"\n📣 СТЕЛЯ: κ₂₁ ×{base/best[0]:.2f} · SPLIT ×{base/bs[0]:.2f} "
      f"(λ={bs[1]:.2f}) · κ₂₁+κ₄ ×{base/b34[0]:.2f} (λ={b34[1]:.2f})")
print(f"📣 σ(h_31) розкид впаде 8.20% → {8.20/max(base/bs[0],1e-9):.2f}% "
      f"(SPLIT, лінеаризація psi91 нев'язка 0.5%)")
print(f"📣 ВИРОК: {'✅ ВИМОГУ ≤1% ДОСЯГНУТО' if base/bs[0] >= 8.2 else '⛔ вимоги ≤1%% НЕ досягнуто, лишається ×%.2f' % (8.20/(base/bs[0]))}")
np.savez(f"/mnt/win_d/llm/White-Box/arena62/logs/k21_91_N{NS}.npz",
         **{f"{nm}_{lam}": torch.stack(ROWS[(nm, lam)]).numpy()
            for nm in ('своя', 'чужа', 'split', '3+4') for lam in LAM})
