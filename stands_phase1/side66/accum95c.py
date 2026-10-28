"""accum95c — СУМІШ K ГАУСІВ У ПОВНІЙ 32-ШАРОВІЙ РЕКУРСІЇ (аналітично, БЕЗ семплів)

════════════════ ЗВІДКИ ════════════════
`[ЗМІРЯНО mix95.py, 4 мережі × 2¹⁸, ОДИН крок]` суміш за проєкцією на середнє:
```
шар 30:  K=1 поза 0.04082 → K=4 0.01581 (×2.58) → K=8 0.01508 (×2.71) → насичення
         σ ×1.27 · E[relu] ×1.34 · ⛔чужий поділ ×0.90 (ГІРШЕ за базу) ✅контроль
поділ за нейроном з мін|t| — слабший (×1.20) ⟹ проєкція на СЕРЕДНЄ правильна
```
🔑 ПОЗАДІАГОНАЛЬ ×2.73 — саме той канал, що несе 4.08% проти діагоналі 1.03%.
`[ЗМІРЯНО accum95]` виправлення діагоналі БЕЗ позадіагоналі дало ×0.83 ⟹
   суміш чіпає ОБА канали ОДНОЧАСНО ⟹ проходить новий гейт узгодженості.
`[ЦИТАТА сторінка лідера]` MEAN EFFECTIVE COMPUTE 2.61e10 = ×15.7 замикання
   ⟹ ціна K=4…8 (×4…8 = 6.6e9…1.3e10) ВМІЩУЄТЬСЯ з запасом.

════════════════ ЯК ЦЕ РОБИТЬСЯ АНАЛІТИЧНО (без жодних семплів) ════════════════
Розділяємо ВХІДНИЙ гаус z~N(0,I) на K рівноімовірних шматків за проєкцією z·v.
Умовний розподіл — усічений нормальний, і його моменти ТОЧНІ:
```
b_k — квантилі N(0,1);  w_k = 1/K
μ_k = (φ(b_{k−1}) − φ(b_k))·K
σ²_k = 1 + (b_{k−1}φ(b_{k−1}) − b_kφ(b_k))·K − μ_k²
компонента k:  z|k ~ N(μ_k·v , I + (σ²_k − 1)·v vᵀ)   (гаусове наближення шматка)
⟹ p_1|k:  m = μ_k·(Wᵀv) ,  C = WᵀW + (σ²_k − 1)·(Wᵀv)(Wᵀv)ᵀ   ← ТОЧНО, rank-1
```
Далі K НЕЗАЛЕЖНИХ рекурсій, на виході зважена сума перших моментів.
🔒 ПЛОМБА: K=1 мусить дати РІВНО базу (μ=0, σ²=1 ⟹ поправка нульова).

════════════════ ЯКИЙ НАПРЯМОК v ════════════════
 (shaft) домінантний правий синг. вектор ВАЛА `A = Π diag(Φ_ℓ)·W_ℓᵀ`
         — усереднений якобіан ∂h_31/∂z, тобто напрямок, до якого вихід
         НАЙЧУТЛИВІШИЙ. Вал зміряно в С94: R²=0.943 на переносі похибки.
 (w1)    домінантний синг. вектор W₁ (дешева альтернатива)
 (rand)  ⛔КОНТРОЛЬ: випадковий напрямок — виграш мусить зникнути

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) 🔑 K=4 із shaft-напрямком дасть MSE ×1.5…×3                 шанс 40%
 (б) rand дасть <×1.1                                           шанс 70%
 (в) насичення на K=4…8, як і на одному кроці                    шанс 65%
 (г) 🚨 РИЗИК: розділення на ВХОДІ розмивається за 31 шар ⟹ ×1.0 шанс 45%
     (це головна причина, чому гілка може померти)
"""
import glob
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import numpy as np                                            # noqa: E402
import pyarrow as pa                                          # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:2"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
MK = 24
KLIST = [1, 2, 3, 4, 6, 8]


def rows(limit):
    got = 0
    for f in sorted(glob.glob(SRC + "/data-*.arrow")):
        t = pa.ipc.open_stream(pa.memory_map(f, "rb")).read_all()
        x = t.column("weights").combine_chunks()
        while not pa.types.is_primitive(x.type):
            x = x.flatten()
        w = np.asarray(x).reshape(t.num_rows, 32, 256, 256)
        fm = np.asarray(t.column("final_means").combine_chunks()
                        .flatten()).reshape(t.num_rows, 256)
        for i in range(t.num_rows):
            if got >= limit:
                return
            yield w[i], fm[i]
            got += 1


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def norm_ppf(p):
    """Обернена функція розподілу N(0,1) — бісекція (потрібно лише K−1 чисел)."""
    lo = torch.full_like(p, -12.0)
    hi = torch.full_like(p, 12.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c = Phi(mid) < p
        lo = torch.where(c, mid, lo)
        hi = torch.where(c, hi, mid)
    return 0.5 * (lo + hi)


def trunc_moments(K, dev):
    """(w, μ, σ²) для K рівноімовірних шматків N(0,1) — ТОЧНІ формули."""
    z64 = dict(dtype=torch.float64, device=dev)
    if K == 1:
        return (torch.ones(1, **z64), torch.zeros(1, **z64),
                torch.ones(1, **z64))
    p = torch.arange(1, K, **z64) / K
    b = norm_ppf(p)
    inf = torch.tensor([float('inf')], **z64)
    bl = torch.cat([-inf, b])
    br = torch.cat([b, inf])
    fl = torch.where(torch.isinf(bl), torch.zeros_like(bl), phi(bl))
    fr = torch.where(torch.isinf(br), torch.zeros_like(br), phi(br))
    xl = torch.where(torch.isinf(bl), torch.zeros_like(bl), bl * fl)
    xr = torch.where(torch.isinf(br), torch.zeros_like(br), br * fr)
    w = torch.full((K,), 1.0 / K, **z64)
    mu = (fl - fr) * K
    v = 1.0 + (xl - xr) * K - mu * mu
    return w, mu, v.clamp_min(1e-12)


def mstep(m, C):
    """Мелер: (E[relu], Cov(relu))."""
    s = C.diagonal().clamp_min(1e-30).sqrt()
    t = m / s
    P, f = Phi(t), phi(t)
    ma = s * (f + t * P)
    rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
    a1 = s * P
    cov = torch.outer(a1, a1) * rho
    Rn = rho.clone()
    Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
    fact, sf = 2.0, s * f
    for nn in range(2, MK + 1):
        Rn = Rn * rho
        an = sf * Hm / fact
        cov = cov + torch.outer(an, an) * Rn * fact
        Hpre, Hm = Hm, (-t) * Hm - float(nn - 2) * Hpre
        fact *= (nn + 1)
    return ma, cov, P


def shaft_dir(W):
    """Домінантний правий синг. вектор вала A = Π diag(Φ_ℓ)·W_ℓᵀ (усередн. якобіан)."""
    m = torch.zeros(256, dtype=torch.float64, device=W[0].device)
    C = W[0].double().T @ W[0].double()
    A = W[0].double()                      # ∂p_1/∂z = W₁ᵀ (діємо як z@W)
    for L in range(32):
        ma, cov, P = mstep(m, C)
        if L == 31:
            break
        Wd = W[L + 1].double()
        A = (A * P) @ Wd                   # A ← A·diag(Φ)·W
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)
    # домінантний ПРАВИЙ синг. вектор A (у просторі z): степеневий метод на AAᵀ
    v = torch.randn(256, dtype=torch.float64, device=W[0].device)
    for _ in range(40):
        v = A @ (A.T @ v)
        v = v / v.norm().clamp_min(1e-300)
    return v


def run_mix(W, v, K):
    """K незалежних рекурсій із аналітично розділеного входу."""
    dev = W[0].device
    w, mu, vv = trunc_moments(K, dev)
    W0 = W[0].double()
    C0 = W0.T @ W0
    a = W0.T @ v                            # Wᵀv
    out = torch.zeros(256, dtype=torch.float64, device=dev)
    for k in range(K):
        m = mu[k] * a
        C = C0 + (vv[k] - 1.0) * torch.outer(a, a)
        for L in range(32):
            ma, cov, _ = mstep(m, C)
            if L == 31:
                out = out + w[k] * ma
                break
            Wd = W[L + 1].double()
            m = ma @ Wd
            C = Wd.T @ (cov @ Wd)
    return out


print(f"accum95c · {DEV} · {NNET} мереж · K={KLIST}", flush=True)
t0 = time.time()
TAGS = ['shaft', 'w1', 'rand']
ERR = {(tg, K): [] for tg in TAGS for K in KLIST}
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        vs = shaft_dir(W)
        W0 = W[0].double()
        u1 = torch.randn(256, dtype=torch.float64, device=DEV)
        for _ in range(30):
            u1 = W0 @ (W0.T @ u1)
            u1 = u1 / u1.norm()
        g = torch.Generator(device=DEV).manual_seed(958000 + 173 * q)
        vr = torch.randn(256, generator=g, device=DEV, dtype=torch.float64)
        vr /= vr.norm()
        for tg, v in (('shaft', vs), ('w1', u1), ('rand', vr)):
            for K in KLIST:
                o = run_mix(W, v, K)
                ERR[(tg, K)].append(float((o - tru).norm() / tru.norm()))
        if q % 4 == 3 or q == 0:
            print(f"  net{q}: K1={ERR[('shaft',1)][-1]*100:.4f}% "
                  f"shaft K4={ERR[('shaft',4)][-1]*100:.4f}% "
                  f"K8={ERR[('shaft',8)][-1]*100:.4f}% ({time.time()-t0:.0f} с)",
                  flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 78)
print("СУМІШ У ПОВНІЙ РЕКУРСІЇ — проти `final_means` (істина 1e9)")
print("=" * 78)
b = float(np.mean(np.array(ERR[('shaft', 1)]) ** 2))
print(f"{'напрям':>8s}{'K':>4s}{'віднос.похибка':>16s}{'MSE ×':>9s}{'RMS ×':>8s}")
for tg in TAGS:
    for K in KLIST:
        e = np.array(ERR[(tg, K)])
        print(f"{tg if K == KLIST[0] else '':>8s}{K:>4d}"
              f"{e.mean()*100:>15.4f}%{b/float((e**2).mean()):>9.2f}"
              f"{float(np.mean(ERR[('shaft',1)]))/e.mean():>8.3f}")
print("\n🔒 ПЛОМБА: K=1 однаковий для всіх трьох напрямків (мусить бути)")
for tg in TAGS:
    print(f"   {tg}: {np.mean(ERR[(tg,1)])*100:.6f}%")
print("\n🔑 ЧИТАТИ: shaft має бити rand. Якщо обидва ×1.0 — розділення на ВХОДІ")
print("   розмивається за 31 шар, і суміш треба будувати ПОШАРОВО.")
