"""learn96b — ВЧИМО ЛИШЕ ФОРМУ. МАСШТАБ ВИКЛЮЧЕНО З МЕТРИКИ ЗА ПОБУДОВОЮ.

════════════════ ЧОМУ ПОПЕРЕДНІЙ ПРОГІН НІЧОГО НЕ СКАЗАВ ════════════════
`[ЗМІРЯНО learn96, 240 мереж, val 60]` `c_n(ℓ)` (768 чисел) дали val 0.5651%,
а «ТІЛЬКИ шарові скаляри `g(ℓ)`» (32 числа) — 0.5653%. ОДНАКОВО.
🔑 ПРИЧИНА СТРУКТУРНА: замикання СТРОГО ПОЗИТИВНО-1-ОДНОРІДНЕ (пломба 3.1e-13,
С87) ⟹ будь-яке масштабне збурення в БУДЬ-ЯКОМУ шарі згортається в ОДИН
множник на виході. І `c_n` входять у `Cov` мультиплікативно ⟹ найдешевший
шлях градієнта — зсунути масштаб, а не форму. Оптимізатор його й знайшов
(`c_n(шар 16)` = 0.994…0.999). А масштаб у сабміті ВЖЕ безкоштовно бере
`_ALPHA` ⟹ виміряні ×2.43 — не нове знання.

════════════════ ЩО ВИПРАВЛЕНО ТУТ ════════════════
1. **МЕТРИКА БЕЗ МАСШТАБУ**: `err² = 1 − cos²(pred, truth)` = мінімум по α
   від `‖α·pred − truth‖²/‖truth‖²`. Тобто рівно те, що лишається ПІСЛЯ
   ідеальної калібровки — і рівно те, з чим конкурує будь-яка нова поправка.
   `[ЦИТАТА С87]` «поправка у безрозмірних (t,ρ) не може конкурувати з α ЗА
   ПОБУДОВОЮ» ⟹ прибираємо α з арени, щоб побачити саму поправку.
2. **НЕМАСШТАБНІ ПАРАМЕТРИЗАЦІЇ**: замість множників на `Cov` — зміни
   ФОРМИ як функції безрозмірного `t`:
```
   P2 Еджвортів базис на СЕРЕДНЬОМУ:  ma += s·φ(t)·(θ₀ + θ₁t + θ₂(t²−1) + θ₃(t³−3t))
      (θ₁ ≈ −γ₃/6, θ₂ ≈ γ₄/24 — але коефіцієнти НАВЧЕНІ, а не зміряні)
   P4 зсув аргументу:                 t_eff = t + d(ℓ)  у Φ і φ
   P3 порядки Мелера c_n(ℓ)           — для звірки, він масштабний
   P1 шарові скаляри g(ℓ)             — 🔒ПЛОМБА: під цією метрикою МУСИТЬ дати НУЛЬ
```
🔑 P1 — головна пломба приладу: якщо шарові скаляри дають виграш під
   масштабно-інваріантною метрикою, метрика зламана.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) 🔒P1 (скаляри) дасть ×1.00…×1.05 — пломба метрики          шанс 85%
 (б) P2 (Еджвортів базис) дасть ×1.3…×3                         шанс 45%
 (в) P4 (зсув t) дасть менше за P2                              шанс 60%
 (г) P3 (Мелер) під цією метрикою впаде до ×1.0…×1.1            шанс 70%
     ← якщо так, це ДОВОДИТЬ, що вчора виграш був чистим масштабом
 (д) найкраща комбінація перетне 0.48% (поріг сенсу)            шанс 25%
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

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:4"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 240
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 250
MODE = sys.argv[4] if len(sys.argv) > 4 else "P2"
BATCH = 4
MK = 24
S2PI = float(1.0 / np.sqrt(2.0 * np.pi))
SRC = ("/home/trim/.cache/huggingface/hub/"
       "datasets--aicrowd--arc-whestbench-public-2026/snapshots/"
       "7c3aa28ba4a6ae6b6d3e4f90f4c35efa12b832f7/prepared/full")


def load(limit):
    Ws, Ts, got = [], [], 0
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
                return Ws, Ts
            Ws.append(w[i].astype(np.float32))
            Ts.append(fm[i].astype(np.float32))
            got += 1
    return Ws, Ts


def Phi(x):
    return 0.5 * torch.erfc(-x * (2.0 ** -0.5))


def phi(x):
    return torch.exp(-0.5 * x * x) * S2PI


def closure(W, P):
    """P — словник навчених тензорів або None для кожного каналу."""
    dev = W.device
    m = torch.zeros(256, dtype=torch.float32, device=dev)
    C = W[0].T @ W[0]
    for L in range(32):
        s = C.diagonal().clamp_min(1e-20).sqrt()
        if P.get("g") is not None:
            s = s * P["g"][L]
            C = C * (P["g"][L] * P["g"][L])
        t = m / s
        te = t + P["d"][L] if P.get("d") is not None else t
        Pe, fe = Phi(te), phi(te)
        ma = s * (Pe * t + fe)
        if P.get("th") is not None:
            q = P["th"][L]
            ma = ma + s * fe * (q[0] + q[1] * t + q[2] * (t * t - 1.0)
                                + q[3] * (t * t * t - 3.0 * t))
        if L == 31:
            return ma
        rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
        a1 = s * Pe
        c = P.get("c")
        cov = torch.outer(a1, a1) * rho * (c[L, 0] if c is not None else 1.0)
        Rn = rho
        Hpre, Hm = torch.zeros_like(te), torch.ones_like(te)
        fact, sf = 2.0, s * fe
        for nn in range(2, MK + 1):
            Rn = Rn * rho
            an = sf * Hm / fact
            term = torch.outer(an, an) * Rn * fact
            cov = cov + (term * c[L, nn - 1] if c is not None else term)
            Hpre, Hm = Hm, (-te) * Hm - float(nn - 2) * Hpre
            fact *= (nn + 1)
        Wd = W[L + 1]
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


def shape_err2(p, y):
    """1 − cos² = мінімум по α від ‖αp−y‖²/‖y‖². МАСШТАБ ВИКЛЮЧЕНО."""
    num = (p * y).sum()
    return (1.0 - num * num / ((p * p).sum() * (y * y).sum())).clamp_min(0.0)


print(f"learn96b · {DEV} · {NNET} мереж · {STEPS} кроків · режим={MODE}",
      flush=True)
t0 = time.time()
Ws, Ts = load(NNET)
ntr = int(len(Ws) * 0.75)
tr_idx, va_idx = list(range(ntr)), list(range(ntr, len(Ws)))
print(f"  {len(Ws)} мереж: train={ntr} val={len(va_idx)} "
      f"({time.time()-t0:.0f} с)", flush=True)

P = {}
if "g" in MODE or MODE == "ALL":
    P["g"] = torch.ones(32, device=DEV, requires_grad=True)
if "P2" in MODE or MODE == "ALL":
    P["th"] = torch.zeros(32, 4, device=DEV, requires_grad=True)
if "P3" in MODE or MODE == "ALL":
    P["c"] = torch.ones(32, MK, device=DEV, requires_grad=True)
if "P4" in MODE or MODE == "ALL":
    P["d"] = torch.zeros(32, device=DEV, requires_grad=True)
if "P1" in MODE:
    P["g"] = torch.ones(32, device=DEV, requires_grad=True)
params = [v for v in P.values() if v.requires_grad]
opt = torch.optim.Adam(params, lr=3e-3) if params else None


def evaluate(idx, PP):
    tot = 0.0
    with torch.no_grad():
        for i in idx:
            W = torch.as_tensor(Ws[i], device=DEV)
            y = torch.as_tensor(Ts[i], device=DEV)
            tot += float(shape_err2(closure(W, PP), y))
    return (tot / len(idx)) ** 0.5


base_va = evaluate(va_idx, {})
print(f"🔒 БАЗА (жодних параметрів), МЕТРИКА БЕЗ МАСШТАБУ: "
      f"val={base_va*100:.4f}%   ← з нею й конкурує будь-яка поправка",
      flush=True)

for step in range(1, STEPS + 1):
    opt.zero_grad()
    loss = 0.0
    for i in np.random.default_rng(step).choice(ntr, BATCH, replace=False):
        W = torch.as_tensor(Ws[int(i)], device=DEV)
        y = torch.as_tensor(Ts[int(i)], device=DEV)
        loss = loss + shape_err2(closure(W, P), y)
    (loss / BATCH).backward()
    opt.step()
    if step % 25 == 0 or step == 1:
        va = evaluate(va_idx, P)
        print(f"  крок {step:4d}  train={float(loss/BATCH)**0.5*100:.4f}%"
              f"  VAL={va*100:.4f}%  (MSE ×{(base_va/va)**2:.2f})"
              f"  [{time.time()-t0:.0f} с]", flush=True)

va = evaluate(va_idx, P)
print("\n" + "=" * 74)
print(f"РЕЖИМ {MODE} · МЕТРИКА БЕЗ МАСШТАБУ · val {base_va*100:.4f}% → "
      f"{va*100:.4f}%  MSE ×{(base_va/va)**2:.2f}")
print(f"🎯 ПОРІГ СЕНСУ: 0.48% (похибка замикання, якій відповідає рекорд)")
if P.get("th") is not None:
    print("θ(шар 16) =", " ".join(f"{float(x):+.4f}" for x in P["th"][16]))
    print("θ(шар 28) =", " ".join(f"{float(x):+.4f}" for x in P["th"][28]))
if P.get("d") is not None:
    print("d(ℓ) =", " ".join(f"{float(x):+.4f}" for x in P["d"][:10]))
torch.save({k: v.detach().cpu() for k, v in P.items()},
           "/mnt/win_d/llm/White-Box/arena62/logs/learn96b_%s.pt" % MODE)
