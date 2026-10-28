"""learn96 — ВЧИМО КОЕФІЦІЄНТИ МЕЛЕРА, А НЕ ПОНЕЙРОННІ ОЗНАКИ

════════════════ ЧОМУ САМЕ ЦЕЙ ОБ'ЄКТ ════════════════
`[ЦИТАТА С87, sigscan87, 107520 точок]` понейронні ознаки дають R²≈0 для
похибки замикання, бо вона є ВЛАСТИВІСТЮ ПАРИ: `σ²_a = Σ_jk w_j w_k Cov_jk`.
`[ЗМІРЯНО ladderm96, С96]` канал `m` несе ×22.6, і живить його саме `σ`.
`[ЗМІРЯНО accum96b, С96]` драбина σ: 1.35%→×2.95 · 0.90%→×7.92 · 0.45%→×31.3
   · 0.18%→×89.2 ⟹ ТЗ на цю гілку: **σ до 0.45%** = бал 1.03e-07.

⟹ Вчити треба ПАРНИЙ об'єкт. Ряд Мелера і Є машинерією пар:
```
   Cov(ReLU_j, ReLU_k) = Σ_n a_n^(j) a_n^(k) ρⁿ · n!     (точний ДЛЯ ГАУСА)
   a_1 = s·Φ(t) ,  a_n = s·φ(t)·He_{n−2}(−t)/n!
```
Його похибка — це рівно негаусовість КОПУЛИ. Вводимо `c_n(ℓ)` (мультиплікатор
на порядок `n` у шарі `ℓ`) і `g(ℓ)` (шаровий масштаб σ):
```
   Cov ← Σ_n c_n(ℓ)·a_n^(j) a_n^(k) ρⁿ n! ,     s ← g(ℓ)·s
```
**768 + 32 числа**, ціна на інференсі РІВНО НУЛЬ (інші константи в тому самому
циклі). Рекурсія диференційовна ⟹ фітимо ПРЯМО по `final_means` (істина 1e9).

🔒 ПЛОМБА: при `c≡1, g≡1` втрата мусить дати РІВНО базу 0.847121%.
🔒 ПЛОМБА ПЕРЕНОСУ: розділ train/val ПО МЕРЕЖАХ; вирок — лише на val.
🚨 `g(ℓ)` навмисно ВКЛЮЧЕНО, хоча `_ALPHA` у сабміті вже їсть шарові скаляри:
   якщо ВЕСЬ виграш сяде в `g`, значить нове знання = нуль, і це буде видно
   в окремому запуску `тільки c` проти `тільки g`.

════════════════ ПЕРЕДБАЧЕННЯ (ДО ЗАПУСКУ) ════════════════
 (а) `тільки g` (32 числа) дасть ×1.1…×1.5 — це те, що вже робить α  шанс 70%
 (б) `c+g` дасть ×2…×6 на ВАЛІДАЦІЇ                                  шанс 35%
 (в) 🔑 σ дійде до 0.45% (бал 1.03e-07)                              шанс 25%
 (г) перенавчання: train ≫ val удвічі і більше                       шанс 45%
 (д) оптимальні `c_n` спадатимуть з `n` (висока копула переоцінена)   шанс 55%
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
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 300
MODE = sys.argv[4] if len(sys.argv) > 4 else "cg"        # "g" | "c" | "cg"
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


def closure(W, cpar, gpar):
    """Замикання з навченими коефіцієнтами. W — (32,256,256) на пристрої."""
    dev = W.device
    m = torch.zeros(256, dtype=torch.float32, device=dev)
    C = W[0].T @ W[0]
    for L in range(32):
        s = C.diagonal().clamp_min(1e-20).sqrt()
        if gpar is not None:
            s = s * gpar[L]
            C = C * torch.outer(gpar[L].expand(256), gpar[L].expand(256))
        t = m / s
        P, f = Phi(t), phi(t)
        ma = s * (P * t + f)
        if L == 31:
            return ma
        rho = (C / torch.outer(s, s)).clamp(-1.0, 1.0)
        a1 = s * P
        cov = torch.outer(a1, a1) * rho * (cpar[L, 0] if cpar is not None
                                           else 1.0)
        Rn = rho
        Hpre, Hm = torch.zeros_like(t), torch.ones_like(t)
        fact, sf = 2.0, s * f
        for nn in range(2, MK + 1):
            Rn = Rn * rho
            an = sf * Hm / fact
            term = torch.outer(an, an) * Rn * fact
            cov = cov + (term * cpar[L, nn - 1] if cpar is not None else term)
            Hpre, Hm = Hm, (-t) * Hm - float(nn - 2) * Hpre
            fact *= (nn + 1)
        Wd = W[L + 1]
        m = ma @ Wd
        C = Wd.T @ (cov @ Wd)


print(f"learn96 · {DEV} · {NNET} мереж · {STEPS} кроків · режим={MODE}",
      flush=True)
t0 = time.time()
Ws, Ts = load(NNET)
ntr = int(len(Ws) * 0.75)
print(f"  завантажено {len(Ws)} мереж: train={ntr} val={len(Ws)-ntr} "
      f"({time.time()-t0:.0f} с)", flush=True)

cpar = torch.ones(32, MK, device=DEV, requires_grad=(MODE in ("c", "cg")))
gpar = torch.ones(32, 1, device=DEV, requires_grad=(MODE in ("g", "cg")))
use_c = cpar if MODE in ("c", "cg") else None
use_g = gpar if MODE in ("g", "cg") else None
params = [p for p in (cpar, gpar) if p.requires_grad]
opt = torch.optim.Adam(params, lr=3e-3)


def evaluate(idx, c, g):
    tot = 0.0
    with torch.no_grad():
        for i in idx:
            W = torch.as_tensor(Ws[i], device=DEV)
            tr = torch.as_tensor(Ts[i], device=DEV)
            p = closure(W, c, g)
            tot += float(((p - tr).norm() / tr.norm()) ** 2)
    return (tot / len(idx)) ** 0.5


tr_idx = list(range(ntr))
va_idx = list(range(ntr, len(Ws)))
base_tr = evaluate(tr_idx[:24], None, None)
base_va = evaluate(va_idx, None, None)
print(f"🔒 ПЛОМБА (c≡1,g≡1): train={base_tr*100:.6f}%  val={base_va*100:.6f}%"
      f"   ← має бути 0.847121%", flush=True)

step = 0
while step < STEPS:
    opt.zero_grad()
    loss = 0.0
    for i in np.random.default_rng(step).choice(ntr, BATCH, replace=False):
        W = torch.as_tensor(Ws[int(i)], device=DEV)
        tr = torch.as_tensor(Ts[int(i)], device=DEV)
        p = closure(W, use_c, use_g)
        loss = loss + ((p - tr).norm() / tr.norm()) ** 2
    (loss / BATCH).backward()
    opt.step()
    step += 1
    if step % 25 == 0 or step == 1:
        va = evaluate(va_idx, use_c, use_g)
        print(f"  крок {step:4d}  train_batch={float(loss/BATCH)**0.5*100:.4f}%"
              f"  VAL={va*100:.4f}%  (×{(base_va/va)**2:.2f})"
              f"  [{time.time()-t0:.0f} с]", flush=True)

va = evaluate(va_idx, use_c, use_g)
print("\n" + "=" * 72)
print(f"РЕЖИМ {MODE}: val {base_va*100:.4f}% → {va*100:.4f}%  "
      f"MSE ×{(base_va/va)**2:.2f}")
e2 = 0.9097 * va * va
V = 2.05e-6
print(f"бал (оцінка) {0.1*V*e2/(V+e2):.4e}   проти рекорду 1.86762e-07")
if use_g is not None:
    print("g(ℓ) =", " ".join(f"{float(x):.4f}" for x in gpar[:, 0][:8]), "...")
if use_c is not None:
    print("c_n(шар 16) =", " ".join(f"{float(x):.3f}" for x in cpar[16][:8]))
torch.save({"c": cpar.detach().cpu(), "g": gpar.detach().cpu()},
           "/mnt/win_d/llm/White-Box/arena62/logs/learn96_%s.pt" % MODE)
