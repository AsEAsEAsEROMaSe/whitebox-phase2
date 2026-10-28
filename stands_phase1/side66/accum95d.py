"""accum95d — РОЗДІЛЕННЯ СУМІШІ НА ПРОМІЖНОМУ ШАРІ (скан по ℓ₀), не на вході

════════════════ ЗВІДКИ ════════════════
`[ЗМІРЯНО accum95c, 24 мережі, ціль final_means]` розділення на ВХОДІ:
```
напрям  K=1      K=2      K=4      K=8     MSE×(K=8)
shaft  0.8471%  0.7506%  0.7218%  0.7122%   ×1.39
w1     0.8471%  0.8405%  0.8370%  0.8352%   ×1.03
rand   0.8471%  0.8452%  0.8445%  0.8442%   ×1.01  ⛔контроль ✅
```
🔒 пломба K=1 однакова до 6 знаків для трьох напрямків ✅
⛔ ×1.39 при ціні ×8 замикання (4.9% B) проти дозволених ~0.35% ⟹ НЕ ОКУПАЄТЬСЯ

`[ЗМІРЯНО cv95, 8 мереж]` R² лінійної апроксимації p_32(z) = **0.222** (вал),
   0.270 (ОПТИМАЛЬНА) ⟹ редукція CV лише ×1.26 ⟹ ⛔ УСІ лінеаризаційні та
   control-variate підходи мертві; мережа НЕ майже-лінійна відносно входу.

════════════════ ІДЕЯ ════════════════
Розділення на ВХОДІ розмивається за 31 шар (насичення вже на K=4). Розділяти
треба ТАМ, ДЕ НЕГАУСОВІСТЬ УЖЕ НАКОПИЛАСЬ (γ₃ росте 0.05→0.38 і насичується
близько шару 20). Це ЩЕ Й ДЕШЕВШЕ: K рекурсій лише від ℓ₀ до 31.

📐 РОЗДІЛЕННЯ ГАУСА (m,C) ПО НАПРЯМКУ u — ТОЧНІ ФОРМУЛИ:
```
σ²_y = uᵀCu ,  d = Cu/σ_y
m_k = m + μ_k·d ,  C_k = C + (v_k − 1)·d dᵀ        ← знову rank-1, дешево
(μ_k, v_k) — моменти усіченого N(0,1) на K рівноімовірних шматках
```
🔒 ПЛОМБА: K=1 ⟹ (μ,v)=(0,1) ⟹ поправка нульова ⟹ РІВНО база.

════════════════ ЩО МІРЯЄМО ════════════════
скан ℓ₀ ∈ {0,4,8,12,16,20,24,28} × K ∈ {2,4,8} × напрямок u:
  (m)   u = m/‖m‖              — найкращий на одному кроці (mix95)
  (e)   u = топ власний вектор C
  (r)   ⛔ВИПАДКОВИЙ — контроль
🔑 ЦІНА: (32−ℓ₀)/32·K замикань ⟹ ℓ₀=20, K=4 коштує ×1.5, а не ×4!

════════════════ ПЕРЕДБАЧЕННЯ (ДО запуску) ════════════════
 (а) 🔑 оптимальний ℓ₀ буде НЕ 0, а 8…20                        шанс 60%
 (б) виграш на оптимальному ℓ₀ буде >×1.39 (краще за вхід)      шанс 50%
 (в) напрямок (m) переможе (e)                                  шанс 45%
 (г) rand дасть ×1.0 на всіх ℓ₀                                 шанс 80%
 (д) 🚨 навіть найкраще не пройде гейт окупності                шанс 65%
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
L0LIST = [0, 4, 8, 12, 16, 20, 24, 28]
KLIST = [2, 4, 8]


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
    lo = torch.full_like(p, -12.0)
    hi = torch.full_like(p, 12.0)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        c = Phi(mid) < p
        lo = torch.where(c, mid, lo)
        hi = torch.where(c, hi, mid)
    return 0.5 * (lo + hi)


def trunc_moments(K, dev):
    z64 = dict(dtype=torch.float64, device=dev)
    if K == 1:
        return (torch.ones(1, **z64), torch.zeros(1, **z64),
                torch.ones(1, **z64))
    p = torch.arange(1, K, **z64) / K
    b = norm_ppf(p)
    inf = torch.tensor([float('inf')], **z64)
    bl, br = torch.cat([-inf, b]), torch.cat([b, inf])
    fl = torch.where(torch.isinf(bl), torch.zeros_like(bl), phi(bl))
    fr = torch.where(torch.isinf(br), torch.zeros_like(br), phi(br))
    xl = torch.where(torch.isinf(bl), torch.zeros_like(bl), bl * fl)
    xr = torch.where(torch.isinf(br), torch.zeros_like(br), br * fr)
    w = torch.full((K,), 1.0 / K, **z64)
    mu = (fl - fr) * K
    v = 1.0 + (xl - xr) * K - mu * mu
    return w, mu, v.clamp_min(1e-12)


def mstep(m, C):
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
    return ma, cov


def tail(W, m, C, L0):
    """Рекурсія від шару L0 до кінця, повертає E[relu(p_31)]."""
    for L in range(L0, 32):
        ma, cov = mstep(m, C)
        if L == 31:
            return ma
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


def run(W, L0, K, mode, gen=None):
    """Пряма рекурсія до L0, там розділення на K компонент, далі K хвостів."""
    dev = W[0].device
    W0 = W[0].double()
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = W0.T @ W0
    for L in range(L0):
        ma, cov = mstep(m, C)
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)
    if K == 1:
        return tail(W, m, C, L0)
    if mode == 'm':
        u = m / m.norm().clamp_min(1e-300)
        if L0 == 0:                      # на вході m=0 ⟹ беремо напрям W₀ᵀ1
            u = C.sum(1)
            u = u / u.norm()
    elif mode == 'e':
        u = torch.randn(256, dtype=torch.float64, device=dev)
        for _ in range(30):
            u = C @ u
            u = u / u.norm().clamp_min(1e-300)
    else:
        u = torch.randn(256, generator=gen, device=dev, dtype=torch.float64)
        u = u / u.norm()
    Cu = C @ u
    sy = float(u @ Cu).__abs__() ** 0.5
    d = Cu / max(sy, 1e-300)
    w, mu, vv = trunc_moments(K, dev)
    out = torch.zeros(256, dtype=torch.float64, device=dev)
    dd = torch.outer(d, d)
    for k in range(K):
        out = out + w[k] * tail(W, m + mu[k] * d,
                                C + (vv[k] - 1.0) * dd, L0)
    return out


print(f"accum95d · {DEV} · {NNET} мереж · ℓ₀={L0LIST} · K={KLIST}", flush=True)
t0 = time.time()
MODES = ['m', 'e', 'r']
ERR = {}
base = []
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        gen = torch.Generator(device=DEV).manual_seed(962000 + 181 * q)

        def er(x):
            return float((x - tru).norm() / tru.norm())
        base.append(er(run(W, 0, 1, 'm')))
        for L0 in L0LIST:
            for K in KLIST:
                for md in MODES:
                    if md == 'r' and K != 4:
                        continue
                    ERR.setdefault((L0, K, md), []).append(
                        er(run(W, L0, K, md, gen)))
        if q % 6 == 5 or q == 0:
            print(f"  net{q}: база={base[-1]*100:.4f}% "
                  f"ℓ₀=16,K=4: {ERR[(16,4,'m')][-1]*100:.4f}% "
                  f"({time.time()-t0:.0f} с)", flush=True)

bm = float(np.mean(np.array(base) ** 2))
print(f"\n  час {time.time()-t0:.0f} с\n" + "=" * 84)
print(f"РОЗДІЛЕННЯ НА ШАРІ ℓ₀ — база {np.mean(base)*100:.4f}% "
      f"(проти `final_means`, {NNET} мереж)")
print("=" * 84)
print(f"{'ℓ₀':>4s}{'ціна×':>7s}", end="")
for K in KLIST:
    print(f"{'K='+str(K)+' (m)':>13s}{'MSE×':>7s}", end="")
print(f"{'K=4 (e)':>10s}{'⛔K=4 rand':>11s}")
for L0 in L0LIST:
    cost = (32 - L0) / 32.0
    print(f"{L0:>4d}", end="")
    print(f"{cost*KLIST[0]:>7.2f}", end="")
    for K in KLIST:
        e = np.array(ERR[(L0, K, 'm')])
        print(f"{e.mean()*100:>12.4f}%{bm/float((e**2).mean()):>7.2f}", end="")
    ee = np.array(ERR[(L0, 4, 'e')])
    er_ = np.array(ERR[(L0, 4, 'r')])
    print(f"{bm/float((ee**2).mean()):>10.2f}{bm/float((er_**2).mean()):>11.2f}")
print("\n💰 ЦІНА в замиканнях = (32−ℓ₀)/32·K ⟹ гейт окупності: k=1.2→0.162% B,")
print("   k=2.0→0.811% B; одне замикання = 1.66e9 = 0.61% B")
print("🔑 ЧИТАТИ: шукаємо клітинку з найбільшим MSE× НА ОДИНИЦЮ ЦІНИ.")
