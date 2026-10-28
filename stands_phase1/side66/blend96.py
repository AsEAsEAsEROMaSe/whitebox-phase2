"""blend96 — ПОНЕЙРОННИЙ БЛЕНД. Вісь, яку за 96 сесій не чіпали жодного разу.

════════════════ ЧОМУ ЦЕ МОЖЕ БУТИ ЖИВИМ ════════════════
`_W_BLEND` — ОДИН скаляр на всі 256 нейронів: `out = mc + W·(α·cl − mc)`.
Але обидві складові сильно НЕОДНОРІДНІ по нейронах:
```
 Var(МК_j) = Var(ReLU(a_j))/N     — великий для великих σ_j
 похибка замикання E_j            — мала для великих |t_j| (там E[ReLU]≈m)
```
`[ЗМІРЯНО newalg95d, С95]` 12.3% нейронів мають |t|<1 і несуть **79.5%**
похибки σ ⟹ неоднорідність ВЕЛИЧЕЗНА. Оптимальна вага понейронно:
`W_j = V_j/(V_j + E_j²)`, і вона мусить сильно гуляти.
🔑 Ціна: НУЛЬ (вектор замість скаляра в тій самій формулі).

════════════════ ЩО МІРЯЄМО ════════════════
Спершу СТЕЛЮ (оракульні `E_j`), бо якщо стеля мала — гілка мертва без
реалізації. Тоді реалізовні версії:
```
 S0  скалярний бленд з оптимальним W        ← наша схема (стеля скаляра)
 S1  оракульний понейронний W_j             ← СТЕЛЯ гілки
 S2  W_j з МОДЕЛІ: V_j з вибірки, E_j = f(t_j) фітована ГЛОБАЛЬНО
 S3  W_j з моделі, але f(t) фітована на ІНШИХ мережах (LOO) ← чесний перенос
 ⛔F  чужа модель (з перемішаними t) — контроль
```
`[ЦИТАТА закон С86]` ВИГРАШ — ВЛАСТИВІСТЬ ПАРИ (ОЗНАКА, БАЗА) ⟹ база тут
S0 з ОПТИМАЛЬНИМ скаляром, а не з нашим 0.0883.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) стеля S1 дасть ×1.5…×4 проти S0                          шанс 55%
 (б) реалізовна S2 забере ≥половини стелі                     шанс 45%
 (в) S3 (чесний перенос) майже не втратить проти S2           шанс 60%
     (модель має лише кілька параметрів ⟹ перенавчання малоймовірне)
 (г) ⛔F дасть <×1.05                                          шанс 75%
 (д) 🔑 якщо стеля <×1.2 — гілка мертва, і це чесний ранній вихід
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

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:3"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
REP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
MK = 24
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
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


def closure(W):
    """Повертає (прогноз, t останнього шару) — `t` потрібен як ОЗНАКА."""
    dev = W[0].device
    m = torch.zeros(256, dtype=torch.float64, device=dev)
    C = W[0].double().T @ W[0].double()
    for L in range(32):
        s = C.diagonal().clamp_min(1e-30).sqrt()
        t = m / s
        P, f = Phi(t), phi(t)
        ma = s * (P * t + f)
        if L == 31:
            return ma, t
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
        Wd = W[L + 1].double()
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


def mc_estimate(W, seed, npair, dev):
    """Наша схема: антитетика + вибілювання. Повертає (оцінка, вибіркова V_j)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, 256, generator=g, device=dev)
    cov = (z.T @ z).double() / npair
    wh = torch.linalg.inv(torch.linalg.cholesky(cov)).T.float()
    y = (z @ wh) @ W[0]
    a = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    for L in range(1, 32):
        a = torch.relu(a @ W[L]) if L < 31 else (a @ W[L])
    x = torch.relu(a).double()
    n = x.shape[0]
    mu = x.mean(0)
    v = (x.var(0, unbiased=True) / n)          # дисперсія САМОЇ оцінки
    return mu, v


print(f"blend96 · {DEV} · {NNET} мереж × {REP} розіграшів · {NPAIR} пар",
      flush=True)
t0 = time.time()
REC = []
with torch.no_grad():
    for q, (wnp, fm) in enumerate(rows(NNET)):
        W = [torch.as_tensor(np.ascontiguousarray(wnp[i]), device=DEV)
             for i in range(32)]
        tru = torch.as_tensor(fm.astype(np.float64), device=DEV)
        cl, tlast = closure(W)
        for r in range(REP):
            mc, v = mc_estimate(W, 810000 + 977 * q + 13 * r, NPAIR, DEV)
            REC.append((cl, tlast, mc, v, tru))
        if q % 8 == 7 or q == 0:
            print(f"  net{q} ({time.time()-t0:.0f} с)", flush=True)

# ── глобальний оптимальний скаляр α і W (наша схема) ──────────────────
num = sum(float((c * y).sum()) for c, _, _, _, y in REC)
den = sum(float((c * c).sum()) for c, _, _, _, y in REC)
alpha = num / den
print(f"\nα (глобальний оптимальний масштаб замикання) = {alpha:.6f}")


def mse_of(w_fn):
    tot = tn = 0.0
    for c, t, mc, v, y in REC:
        w = w_fn(c, t, mc, v, y)
        p = mc + w * (alpha * c - mc)
        tot += float(((p - y) ** 2).sum())
        tn += float((y * y).sum())
    return tot / tn


best = (1e9, 0.0)
for w in np.linspace(0.0, 0.6, 61):
    m = mse_of(lambda c, t, mc, v, y, w=w: w)
    if m < best[0]:
        best = (m, w)
S0, w0 = best
print(f"S0 скалярний бленд: оптимум W={w0:.3f}, MSE={S0:.4e} "
      f"(відн. похибка {S0**0.5*100:.4f}%)")

# ── S1: ОРАКУЛЬНИЙ понейронний W_j = V_j/(V_j+E_j²) ───────────────────
S1 = mse_of(lambda c, t, mc, v, y:
            v / (v + (alpha * c - y) ** 2).clamp_min(1e-300))
print(f"S1 ОРАКУЛЬНИЙ понейронний W_j: MSE={S1:.4e} "
      f"({S1**0.5*100:.4f}%)  ⟹ СТЕЛЯ ×{S0/S1:.2f}")

# ── S2: модель E_j² = exp(поліном від t) — фіт ГЛОБАЛЬНО ──────────────
T = torch.cat([t for _, t, _, _, _ in REC])
E2 = torch.cat([(alpha * c - y) ** 2 for c, _, _, _, y in REC])
tc = T.clamp(-4.0, 6.0)
X = torch.stack([torch.ones_like(tc), tc, tc * tc, tc ** 3], 1)
lg = torch.log(E2.clamp_min(1e-30))
coef = torch.linalg.lstsq(X, lg.unsqueeze(1)).solution[:, 0]
print("модель log E² = " + " + ".join(f"{float(coef[i]):+.3f}·t^{i}"
                                      for i in range(4)))


def model_e2(t):
    tc = t.clamp(-4.0, 6.0)
    return torch.exp(coef[0] + coef[1] * tc + coef[2] * tc * tc
                     + coef[3] * tc ** 3)


S2 = mse_of(lambda c, t, mc, v, y: v / (v + model_e2(t)))
print(f"S2 модель E²=f(t), V з вибірки: MSE={S2:.4e} "
      f"({S2**0.5*100:.4f}%)  ⟹ ×{S0/S2:.2f}")

# ── S4: РЕАЛІЗОВНА без жодного передбачення похибки:
#    `W_j = V_j/(V_j + c)`, де V_j — ВИМІРЯНА вибіркова дисперсія оцінки,
#    а `c` — ОДИН фітований скаляр. Не потребує моделі `E_j` взагалі.
#    🔑 Це строго загальніше за скалярний бленд і не вимагає нічого, чого
#    в сабміті немає: `V_j` рахується з тієї самої вибірки одним `var`.
bestc = (1e9, 0.0)
for cc in np.exp(np.linspace(np.log(1e-10), np.log(1e-4), 40)):
    m = mse_of(lambda c_, t, mc, v, y, cc=cc: v / (v + cc))
    if m < bestc[0]:
        bestc = (m, cc)
S4, c4 = bestc
print(f"S4 W_j=V_j/(V_j+c), c={c4:.3e}: MSE={S4:.4e} "
      f"({S4**0.5*100:.4f}%)  ⟹ ×{S0/S4:.2f}  ← РЕАЛІЗОВНА, без моделі похибки")

# ── S5: те саме, але `c` як функція t лише ДВОМА параметрами (обережно) ──
bestd = (1e9, 0.0, 0.0)
for c0 in np.exp(np.linspace(np.log(1e-9), np.log(1e-5), 12)):
    for b in (0.0, 0.3, 0.6, 1.0, 1.5):
        m = mse_of(lambda c_, t, mc, v, y, c0=c0, b=b:
                   v / (v + c0 * torch.exp(-b * t.clamp(-3, 6))))
        if m < bestd[0]:
            bestd = (m, c0, b)
S5, c5, b5 = bestd
print(f"S5 c(t)=c0·exp(−b·t), c0={c5:.2e} b={b5:.2f}: MSE={S5:.4e} "
      f"({S5**0.5*100:.4f}%)  ⟹ ×{S0/S5:.2f}")

# ── ⛔F: контроль — ті самі ваги, але переставлені по нейронах ─────────
S3 = mse_of(lambda c, t, mc, v, y:
            (v / (v + model_e2(t)))[torch.randperm(256, device=DEV)])
print(f"⛔F перемішані ваги (контроль): MSE={S3:.4e}  ⟹ ×{S0/S3:.2f}")

V0 = 2.05e-6
print(f"\n💰 ПЕРЕКЛАД У БАЛ (масштабуванням від нашої точки 1.86762e-07):")
for nm, m in (("S0 скаляр", S0), ("S1 стеля", S1), ("S4 V_j/(V_j+c)", S4),
              ("S5 c(t)", S5), ("S2 модель", S2)):
    print(f"   {nm:12s} бал ≈ {1.86762e-07 * m / S0:.4e}")
print("\n🔑 ЧИТАТИ: якщо S1 (стеля) < ×1.2 — гілка мертва без реалізації.")
