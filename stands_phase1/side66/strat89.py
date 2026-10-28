"""strat89 — СТРАТИФІКАЦІЯ ПО ГОЛОВНИХ НАПРЯМКАХ ПЕРЕДАЧІ.

════════════════ ЗВІДКИ ЦЕ ════════════════
Вісь backward підказав ЮЗЕР. `sens89` зміряв матрицю передачі
`J = ∂вихід/∂z` і показав:
    топ-1 напрямок несе **33.6%** чутливості, топ-3 — **50.8%**,
    топ-10 — 76.9%, ефективний ранг 19.1 (при 256 вимірах входу).
Тобто МК «розмазує» семпли рівномірно по 256 вимірах, хоча половина
відповіді живе у ТРЬОХ. Стратифікація по цих напрямках прибирає їхній
внесок у дисперсію майже повністю, і коштує НУЛЬ — це просто інший спосіб
генерувати `z`.

📐 СХЕМА: `z = Σ_r c_r v_r + z⊥`, де `v_r` — топ-`R` правих сингулярних
векторів `J`. Замість випадкових `c_r` беремо СТРАТИФІКОВАНУ сітку
(рівноймовірні страти нормалі), `z⊥` — як завжди. Оцінка лишається
НЕЗМІЩЕНОЮ: кожна страта має рівну вагу.

════════════════ ЧОМУ ЦЕ НЕ «ЗАКРИТИЙ QMC» ════════════════
QMC/кубатури закрито ЗА ПОБУДОВОЮ для повного 256-вимірного простору
(`0.733³²=5e-05`). Тут інший об'єкт: стратифікується R≤3 вимірів, де
сконцентровано 50% чутливості, а решта 253 лишаються звичайною МК.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Якщо частка `p` дисперсії йде через стратифіковані напрямки, а страт `S`,
то дисперсія падає до `(1−p) + p/S²` (для гладкої по `c` функції) або
`(1−p) + p/S` (для негладкої). ReLU кусково-лінійна ⟹ ближче до `S²`.
Ставлю: R=1,S=8 дасть ×1.3…1.5 (шанс 55%); R=3,S=4 дасть ×1.6…2.0
(шанс 40%). Шанс, що сумарно вийде >×2: **25%**.
⚠️ ГОЛОВНИЙ РИЗИК, називаю наперед: `J` усереднений по семплах, а маски
`gate` залежать від семпла ⟹ «головні напрямки» самі шумні, і для
конкретного семпла можуть бути не тими. Тоді `p` фактична буде меншою
за 50.8%, і виграш просяде.

ПЛОМБА: R=0 (без стратифікації) мусить відтворити звичайну МК.
ПЛОМБА НЕЗМІЩЕНОСТІ: при великому N стратифікована оцінка мусить збігтися
з істиною так само, як звичайна (різниця ≤ МК-шум).
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 12
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NPAIR = 2716                      # як у сабміті
NTRUTH = 1 << 18
DT = torch.float32
SQ2 = 1.4142135623730951

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · повторів {MREP} · пар {NPAIR}")


def top_dirs(ws, R, iters=12, seed=0):
    """Топ-R правих сингулярних напрямків `J` — степеневим методом.

    `J = Π_k W_k diag(p_k)`, де `p_k` = ймовірність активації із ЗАМИКАННЯ
    (дешево: L матвеків на ітерацію, а не матриця n×n).
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    # ймовірності активації з дешевого замикання по діагоналі
    m = torch.zeros(n, dtype=DT, device=DEV)
    var = torch.ones(n, dtype=DT, device=DEV)
    P = []
    for i in range(L - 1):
        w = ws[i]
        v2 = var @ (w * w)
        mu = m @ w
        s = torch.sqrt(torch.clamp(v2, min=1e-30))
        t = mu / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        P.append(p)
        f = torch.exp(-0.5 * t * t) / 2.5066282746310002
        m = s * (t * p + f)
        var = torch.clamp((mu * mu + s * s) * p + mu * s * f - m * m, min=1e-30)
    V = torch.randn(R, n, generator=g, device=DEV, dtype=DT)
    for _ in range(iters):
        X = V
        for i in range(L - 1):
            X = (X @ ws[i]) * P[i]
        X = X @ ws[L - 1]                       # вперед
        for i in range(L - 1, 0, -1):
            X = X @ ws[i].T
            X = X * P[i - 1]
        X = X @ ws[0].T                          # назад
        V, _ = torch.linalg.qr(X.T)
        V = V.T
    return V                                      # (R, n), ортонормовані


def strat_normals(nrow, R, S, gen):
    """Стратифіковані нормальні координати: `S` рівноймовірних страт на вимір."""
    c = torch.empty(nrow, R, device=DEV, dtype=DT)
    for r in range(R):
        idx = torch.randperm(nrow, generator=gen, device=DEV) % S
        u = (idx.to(DT) + torch.rand(nrow, generator=gen, device=DEV, dtype=DT)) / S
        u = torch.clamp(u, 1e-6, 1 - 1e-6)
        c[:, r] = SQ2 * torch.erfinv(2 * u - 1)
    return c


def run(ws, R, S, V, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NPAIR, n, generator=g, device=DEV, dtype=DT)
    if R > 0:
        Vr = V[:R]
        z = z - (z @ Vr.T) @ Vr                   # прибрати стару компоненту
        z = z + strat_normals(NPAIR, R, S, g) @ Vr
    x = torch.cat([z, -z], 0)
    for i in range(L):
        p = x @ ws[i]
        if i == L - 1:
            return torch.clamp(p, min=0.0).mean(0)
        x = torch.clamp(p, min=0.0)


def truth(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < NTRUTH:
        b = min(1 << 13, NTRUTH - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        for i in range(L):
            h = x @ ws[i]
            if i == L - 1:
                acc += torch.clamp(h, min=0.0).sum(0).double()
                break
            x = torch.clamp(h, min=0.0)
        done += b
    return acc / NTRUTH


CFG = [(0, 1), (1, 4), (1, 8), (1, 16), (2, 4), (3, 4), (3, 8), (6, 4)]
mse = {c: [] for c in CFG}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q).to(DT)
    den = float((tr * tr).mean())
    V = top_dirs(ws, 6, seed=333 + q)
    for (R, S) in CFG:
        acc = 0.0
        for r in range(MREP):
            p = run(ws, R, S, V, 606000 + 977 * q + r)
            acc += float(((p - tr) ** 2).mean() / den)
        mse[(R, S)].append(acc / MREP)
    if q % 4 == 0:
        print(f"  net{q:>3d}  база {100 * mse[(0, 1)][-1] ** 0.5:.4f}%  "
              f"R3S4 {100 * mse[(3, 4)][-1] ** 0.5:.4f}%  ({time.time() - t0:.0f} с)")

print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 72)
print(f"СТРАТИФІКАЦІЯ ПО ТОП-R НАПРЯМКАХ ПЕРЕДАЧІ · {NNET} мереж × {MREP} повторів")
print("=" * 72)
b0 = np.array(mse[(0, 1)]).mean()
print(f"{'R':>4s} {'страт':>7s} {'віднос.похибка':>16s} {'дисперсія ×':>13s} {'бал':>12s}")
for (R, S) in CFG:
    v = np.array(mse[(R, S)]).mean()
    print(f"{R:>4d} {S:>7d} {100 * v ** 0.5:>15.4f}% {b0 / v:>12.4f}×"
          f" {1.8676e-07 / (b0 / v):>12.4e}")
bc = max(CFG, key=lambda c: b0 / np.array(mse[c]).mean())
bg = b0 / np.array(mse[bc]).mean()
print(f"\n  🎯 НАЙКРАЩЕ R={bc[0]} страт={bc[1]}: дисперсія ×{bg:.4f} ⟹ "
      f"бал {1.8676e-07 / bg:.4e}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/strat89.npz",
         **{f"R{R}_S{S}": np.array(mse[(R, S)]) for R, S in CFG})
