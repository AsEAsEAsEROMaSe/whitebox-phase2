"""psi91 — ВИРІШАЛЬНЕ ЧИСЛО `Ψ`: чи КОГЕРЕНТНО накопичується ін'єкція в `σ`.

════════════════ ЧОМУ ПОПЕРЕДНІЙ СТЕНД (coher91) БУВ ТАВТОЛОГІЄЮ ════════════
Я міряв `√diag(WᵀĈov(a)W)` проти вибіркової `σ(h)`. Але
`Var̂(h_b) = Σ_jk w_jb w_kb Ĉov(a)_jk = diag(WᵀĈov(a)W)_b` **ТОТОЖНО** на тих
самих семплах ⟹ різниця 0 за побудовою. Лінійний крок замикання ТОЧНИЙ.
🔑 **Ін'єкція народжується ВИКЛЮЧНО у гаусовій формулі `(m_h, Cov_h) → Cov_a`.**

════════════════ ЩО МІРЯЄ ЦЕЙ СТЕНД ════════════════
Крок ін'єкції на шарі ℓ (чесний, не тавтологія):
```
вхід   ІСТИННІ  m̂_h(ℓ), Ĉov_h(ℓ)          ← з семплів, точні
формула        (ma, Ca) = relu_moments(...)  ← ЄДИНА апроксимація
істина         m̂_a(ℓ), Ĉov_a(ℓ)           ← з ТИХ САМИХ семплів
ρ_ℓ = ½·log diag(WᵀCa_pred W) / diag(WᵀĈov_a W)   = δσ/σ, понейронно
```
Далі — розклад накопичення. Для КОЖНОГО ℓ проганяю ДВІ копії замикання від
шару ℓ до 31: одну з `Ca_pred(ℓ)`, другу з `Ĉov_a(ℓ)`. Різниця на виході
```
e_ℓ = log σ(h_31)_збурена / σ(h_31)_опорна
```
— це внесок ОДНІЄЇ ін'єкції в кінцеву похибку `σ`, пропущений через власний
якобіан замикання (без жодних припущень про `G`).

**МІРА КОГЕРЕНТНОСТІ**
```
Ψ = (Σ_ℓ e_ℓ)² / (L · Σ_ℓ e_ℓ²)      Ψ→1 когерентно · Ψ→1/L некогерентно
```
Окремо для шарово-СПІЛЬНОЇ компоненти (середнє по нейронах) і для РОЗКИДУ
(понейронний вектор із знятим середнім — чи збігаються самі ВІЗЕРУНКИ).

════════════════ НАВІЩО ЦЕ ВИРІШУЄ ГІЛКУ `κ₂₁` ════════════════
`κ₂₁` прибирає частку `f` ін'єкції на КОЖНОМУ шарі. Під лінеаризацією
`σ_31` падає рівно в `(1−f)`, незалежно від Ψ. АЛЕ Ψ каже, ЯКА ЧАСТИНА
нинішньої похибки взагалі народжена ін'єкцією, а яка — насиченням:
```
Ψ висока ⟹ σ_∞ = 41.67·inj = 2.44%,  κ₂₁ мусить прибрати 72%, щоб дійти 0.68%
Ψ низька ⟹ σ_∞ = 4.59·inj = 0.268%, тоді ВИМОГА <1% УЖЕ ВИКОНАНА
```
Різниця ×9.07 — і вона вирішує, чи детермінована гілка жива без `κ₂₁`.

════════════════ ПЛОМБИ ════════════════
P1 ЛІНЕАРИЗАЦІЯ: `Σ_ℓ e_ℓ` мусить відтворити повну похибку замикання на 31
   (окремий прогін від шару 0). Розбіжність >30% ⟹ накопичення нелінійне.
P2 ТОТОЖНІСТЬ: `e_30` мусить ТОЧНО дорівнювати сирій ін'єкції `ρ_30`
   (між 30 і 31 лише лінійний крок, пропагувати нічого).
P3 Ψ(константний ряд) = рівно 1.0 · Ψ(знакозмінний ±1) = 1/L при парному L.
P4 МАСШТАБУВАННЯ: два `N` (8192 і 32768). Шум ін'єкції падає в 2×,
   справжня стоїть. Без цього не відрізнити формулу від шуму Ĉov.

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) Ψ(спільна) > 0.7  — ін'єкція однобічна                     шанс 65%
 (б) Ψ(розкид)  < 0.3  — понейронні візерунки НЕ вирівняні      шанс 55%
 (в) P1 зійдеться в межах 30%                                   шанс 55%
 (г) сира ін'єкція ρ на глибині 0.2–0.5% (з stepcov87 0.35%)    шанс 50%
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
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i]
            got += 1


def relu_moments(mh, Ch):
    """ЄДИНА апроксимація замикання: (m,Cov) передактивації → (m,Cov) активації."""
    s = Ch.diagonal().clamp(min=1e-20).sqrt()
    t = mh / s
    p = 0.5 * (1.0 + torch.erf(t / SQ2))
    f = torch.exp(-0.5 * t * t) * INV_SQ2PI
    ma = mh * p + s * f
    va = ((mh * mh + s * s) * p + mh * s * f - ma * ma).clamp(min=0.0)
    Cn = Ch * torch.outer(p, p) + (0.5 * (Ch * Ch)) * torch.outer(f / s, f / s)
    Ca = Cn + torch.diag(va - Cn.diagonal())
    return ma, Ca


def run_from(ws, ell, ma, Ca):
    """Замикання від активації шару ell до σ(h_31). Повертає σ(h_31)."""
    for k in range(ell + 1, 32):
        wd = ws[k]
        mh = ma @ wd
        Ch = wd.T @ (Ca @ wd)
        if k == 31:
            return Ch.diagonal().clamp(min=1e-30).sqrt()
        ma, Ca = relu_moments(mh, Ch)


def empirical(ws, N, seed):
    """Істинні (m_h, Cov_h, m_a, Cov_a) по всіх шарах з N семплів."""
    g = torch.Generator().manual_seed(seed)
    n = ws[0].shape[0]
    MH, CH, MA, CA = [], [], [], []
    x = torch.randn(N, n, generator=g)
    for k in range(32):
        h = (x @ ws[k].float()).double()
        mh = h.mean(0)
        hc = h - mh
        MH.append(mh); CH.append((hc.T @ hc) / (N - 1))
        a = h.clamp(min=0.0)
        ma = a.mean(0)
        ac = a - ma
        MA.append(ma); CA.append((ac.T @ ac) / (N - 1))
        x = a.float()
    return MH, CH, MA, CA


def psi(E):
    """E: (L, n). Ψ по сумі проти суми квадратів."""
    L = E.shape[0]
    num = float((E.sum(0) ** 2).sum())
    den = L * float((E * E).sum())
    return num / max(den, 1e-300)


RHO, EV, SEAL, FULL = [], [], [], []
t0 = time.time()
with torch.no_grad():
    for q, wnp in enumerate(rows(NNET)):
        ws = [torch.as_tensor(np.ascontiguousarray(wnp[i])).double()
              for i in range(32)]
        MH, CH, MA, CA = empirical(ws, NS, 770000 + 11 * q)
        n = 256
        rho = torch.zeros(31, n, dtype=torch.float64)
        ev = torch.zeros(31, n, dtype=torch.float64)
        for ell in range(31):
            map_, Cap = relu_moments(MH[ell], CH[ell])
            wd = ws[ell + 1]
            dp = (wd.T @ (Cap @ wd)).diagonal().clamp(min=1e-30)
            dt = (wd.T @ (CA[ell] @ wd)).diagonal().clamp(min=1e-30)
            rho[ell] = 0.5 * torch.log(dp / dt)
            sp = run_from(ws, ell, map_, Cap)
            sr = run_from(ws, ell, MA[ell], CA[ell])
            ev[ell] = torch.log(sp / sr)
        # ⛔P1: повний прогін замикання від шару 0 (C_h0 = W0ᵀW0 точно)
        C0 = ws[0].T @ ws[0]
        m0 = torch.zeros(n, dtype=torch.float64)
        ma, Ca = relu_moments(m0, C0)
        sfull = run_from(ws, 0, ma, Ca)
        strue = CH[31].diagonal().clamp(min=1e-30).sqrt()
        FULL.append(torch.log(sfull / strue))
        SEAL.append(float(torch.abs(ev.sum(0) - torch.log(sfull / strue)).mean()))
        RHO.append(rho); EV.append(ev)
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

RHO = torch.stack(RHO); EV = torch.stack(EV); FULL = torch.stack(FULL)
print(f"\n  час {time.time()-t0:.0f} с · мереж {len(EV)} · семплів {NS}\n")

L = 31
print("=" * 96)
print(f"СИРА ІН'ЄКЦІЯ ρ_ℓ = δσ/σ від формули (m,Cov)→Cov_a · {len(EV)} мереж · N={NS}")
print("=" * 96)
print(f"{'шар':>6s}{'|спільна|':>13s}{'розкид':>12s}{'|e_ℓ| на 31':>15s}"
      f"{'e спільна@31':>16s}")
for ell in (0, 4, 8, 12, 16, 20, 24, 28, 30):
    c = float(RHO[:, ell].mean(1).abs().mean())
    s = float(RHO[:, ell].std(1).mean())
    ec = float(EV[:, ell].mean(1).abs().mean())
    es = float(EV[:, ell].abs().mean())
    print(f"{ell:>6d}{100*c:>12.4f}%{100*s:>11.4f}%{100*es:>14.4f}%{100*ec:>15.4f}%")

print(f"\n⛔P2 e_30 ≡ ρ_30: max |Δ| = "
      f"{float((EV[:, 30] - RHO[:, 30]).abs().max()):.3e} "
      f"{'✅' if float((EV[:,30]-RHO[:,30]).abs().max()) < 1e-12 else '🚨'}")
print(f"⛔P3 Ψ(конст) = {psi(torch.ones(L, 1, dtype=torch.float64)):.6f} (треба 1.0) · "
      f"Ψ(±1) = {psi(torch.tensor([[(-1.0)**i] for i in range(L)], dtype=torch.float64)):.6f} "
      f"(треба ≈{1/L if L%2==0 else 1/L**2:.5f})")
sl = float(np.mean(SEAL))
fm = float(FULL.abs().mean())
print(f"⛔P1 ЛІНЕАРИЗАЦІЯ: |Σe_ℓ − повна| сер. {100*sl:.4f}% проти повної "
      f"{100*fm:.4f}% ⟹ нев'язка {100*sl/max(fm,1e-30):.1f}% "
      f"{'✅' if sl/max(fm,1e-30) < 0.30 else '🚨 НЕЛІНІЙНО'}")

print("\n" + "=" * 96)
print("Ψ — КОГЕРЕНТНІСТЬ НАКОПИЧЕННЯ")
print("=" * 96)
pc = np.mean([psi(EV[q].mean(1, keepdim=True)) for q in range(len(EV))])
ps = np.mean([psi(EV[q] - EV[q].mean(1, keepdim=True)) for q in range(len(EV))])
pa = np.mean([psi(EV[q]) for q in range(len(EV))])
print(f"  Ψ = 1 когерентно · Ψ = 1/{L} = {1/L:.4f} некогерентно\n")
print(f"  Ψ(спільна по нейронах) = {pc:.4f}")
print(f"  Ψ(розкид, центрований) = {ps:.4f}")
print(f"  Ψ(повний вектор)       = {pa:.4f}")
print(f"\n  повна похибка σ(h_31) замикання: |спільна| "
      f"{100*float(FULL.mean(1).abs().mean()):.3f}% · розкид "
      f"{100*float(FULL.std(1).mean()):.3f}%   (live91 M10: 10.19% / 8.86%)")
amp_c = float(FULL.mean(1).abs().mean()) / max(float(RHO.mean(2).abs().mean()), 1e-30)
print(f"  ПІДСИЛЕННЯ спільної = {amp_c:.2f}×   "
      f"(теорія: когерентно 41.67 · некогерентно 4.59)")
print(f"\n📣 ВИРОК: {'КОГЕРЕНТНО ⟹ σ_∞ = 41.67·inj, κ₂₁ ПОТРІБЕН' if pc > 0.5 else 'НЕКОГЕРЕНТНО ⟹ σ_∞ = 4.59·inj, вимога <1% ближча'}")
np.savez(f"/mnt/win_d/llm/White-Box/arena62/logs/psi91_N{NS}.npz",
         RHO=RHO.numpy(), EV=EV.numpy(), FULL=FULL.numpy(),
         SEAL=np.array(SEAL))
