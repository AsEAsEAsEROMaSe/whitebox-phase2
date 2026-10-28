"""prune90b — ВИПРАВЛЕННЯ `prune90`: БАЗА МУСИТЬ БУТИ З ПІЛОТОМ.

🚨 МОЯ ПОМИЛКА В `prune90`: базою я взяв НЕОБРІЗАНУ мережу, а в сабміті вже
СТОЇТЬ ПІЛОТ (`estimator_sub99.py`, рядки 283–303): 128 пар прогону, нейрон
живий якщо `max(h, axis=0) >= 0`. Тому «×1.4256» міряло переважно те, що ми
ВЖЕ маємо. Це рівно наш закон: ВИГРАШ Є ВЛАСТИВІСТЮ ПАРИ (ОЗНАКА, БАЗА).

🚨 І гілка вже має вирок: С80 (`finding-analytic-pruning-is-dead-gaussian-tail-fails`)
— при ОДНАКОВІЙ частці обрізання пілот втрачає 0.0100% маси, аналітичне
`Φ(m/s)` — 45.93%, у 4600× гірше. Причина: гаусів хвіст замикання нікчемний
саме там, де живе мертвість.

════════════════ ТОДІ НАЩО ЦЕЙ СТЕНД ════════════════
Бо С80 міряла ВТРАТУ МАСИ при заміні пілота, а тут інше питання:
чи є щось ПОВЕРХ пілота, і чи вартий сам пілот своїх флопів.
Три режими на одній таблиці (єдине надійне порівняння):
  A) ПІЛОТ — рівно як у сабміті (база; його ціна 2·128·повний_семпл врахована)
  B) ПІЛОТ + аналітичне довідрізання за `p_j < ε` із замикання
  C) ЛИШЕ аналітичне (пілота нема ⟹ його 1.07e9 флопів звільняються)

════════════════ ПЛОМБИ ════════════════
P1 A мусить відтворити сабмітний рівень: ~0.24% похибки, дисперсія ~91–93%.
P2 ЦІНА ПІЛОТА рахується явно і входить у бюджет (інакше C виглядатиме
   безпідставно кращим).
P3 ε=0 у режимі B мусить дати РІВНО A.

════════════════ ПЕРЕДБАЧЕННЯ ════════════════
 (а) B дасть ×1.00…1.08 (шанс 65%) — пілот уже зрізав головне;
 (б) C програє A (шанс 70%) — за С80 аналітичне ріже не ті нейрони;
 (в) шанс, що хоч щось дасть >×1.10: **20%** (було 35% до знахідки про пілот).
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 28
MREP = int(sys.argv[3]) if len(sys.argv) > 3 else 16
NMC = 5432
PILOT = 128
NTRUTH = 1 << 19
DT = torch.float32
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
FULL = 2.0 * L * n * n                      # ціна повного семпла
PILOT_COST = 2.0 * PILOT * FULL             # пілот: 128 пар = 256 семплів
print(f"{DEV} · мереж {NNET} · повторів {MREP} · повний семпл {FULL:.3e}"
      f" · пілот {PILOT_COST:.3e}\n")


def closure_p(ws):
    m = torch.zeros(n, dtype=torch.float64, device=DEV)
    C = torch.eye(n, dtype=torch.float64, device=DEV)
    P = []
    for i in range(L):
        w = ws[i].double()
        mh = m @ w
        Ch = w.T @ (C @ w)
        d = torch.diagonal(Ch).clamp(min=1e-30)
        s = d.sqrt()
        t = mh / s
        p = 0.5 * (1.0 + torch.erf(t / SQ2))
        f = torch.exp(-0.5 * t * t) * INV_SQ2PI
        ma = s * (t * p + f)
        P.append(p)
        if i == L - 1:
            break
        va = ((mh * mh + d) * p + mh * s * f - ma * ma).clamp(min=1e-30)
        Ca = Ch * torch.outer(p, p)
        C = Ca + torch.diag(va - torch.diagonal(Ca))
        m = ma
    return P


def pilot_alive(ws, seed):
    """РІВНО як у сабміті: 128 пар, нейрон живий якщо max(h) >= 0."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    zp = torch.randn(PILOT, n, generator=g, device=DEV, dtype=DT)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(L):
        h = a @ ws[i]
        if i < L - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    return alive


def cut(ws, alive):
    out, fl = [], 0.0
    prev = torch.ones(n, dtype=torch.bool, device=DEV)
    for i in range(L):
        k = alive[i] if i < L - 1 else torch.ones(n, dtype=torch.bool, device=DEV)
        out.append(ws[i][prev][:, k].contiguous())
        fl += 2.0 * int(prev.sum()) * int(k.sum())
        prev = k
    return out, fl


def fwd(ws, z):
    x = z
    for i in range(len(ws)):
        h = x @ ws[i]
        x = torch.relu(h) if i < len(ws) - 1 else h
    return torch.relu(h).mean(0).double()


def mc(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(NMC // 2, n, generator=g, device=DEV, dtype=DT)
    z = torch.cat([z, -z], 0)
    cov = (z.T @ z).double() / z.shape[0]
    return fwd(ws, z @ torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT))


def truth(ws, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < NTRUTH:
        b = min(1 << 13, NTRUTH - done)
        x = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        acc += fwd(ws, x) * b
        done += b
    return acc / NTRUTH


EPS = [1e-4, 1e-3, 3e-3, 1e-2]
REG = ["A_pilot"] + [f"B_pilot+{e:.0e}" for e in EPS] + [f"C_only{e:.0e}" for e in EPS]
Vt = {r: [] for r in REG}
Bi = {r: [] for r in REG}
Fl = {r: [] for r in REG}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q)
    den = float((tr * tr).sum())
    P = closure_p(ws)
    al = pilot_alive(ws, 4242 + q)
    variants = [("A_pilot", al, PILOT_COST)]
    for e in EPS:
        variants.append((f"B_pilot+{e:.0e}",
                         [al[i] & (P[i] >= e) for i in range(L - 1)], PILOT_COST))
        variants.append((f"C_only{e:.0e}", [(P[i] >= e) for i in range(L - 1)], 0.0))
    for name, alive, extra in variants:
        pw, fl = cut(ws, alive)
        Fl[name].append(fl + extra / max(NMC, 1))     # ціна пілота на семпл
        acc, ms = 0.0, torch.zeros(n, dtype=torch.float64, device=DEV)
        for r in range(MREP):
            p = mc(pw, 808000 + 977 * q + r)
            acc += float(((p - tr) ** 2).sum() / den)
            ms += p
        Vt[name].append(acc / MREP)
        Bi[name].append(float(((ms / MREP - tr) ** 2).sum() / den))
    if q % 7 == 0:
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n")
fA, tA = np.mean(Fl["A_pilot"]), np.mean(Vt["A_pilot"])
bA = np.mean(Bi["A_pilot"])
vA = tA - bA
print("=" * 104)
print(f"ОБРІЗАННЯ ПРОТИ ПРАВИЛЬНОЇ БАЗИ (ПІЛОТ, ЯК У САБМІТІ) · {NNET} мереж × {MREP}")
print("=" * 104)
print(f"{'режим':>16s} {'ціна/семпл':>12s} {'× до A':>8s} {'похибка':>10s} "
      f"{'зміщення':>10s} {'дисперсія':>10s} {'БАЛ ×':>9s}")
best = ("A_pilot", 1.0)
for r in REG:
    f = np.mean(Fl[r])
    tt = np.mean(Vt[r])
    bb = np.mean(Bi[r])
    vv = tt - bb
    sc = f / fA
    gain = (bA + vA) / max(bb + vv * sc, 1e-30)
    print(f"{r:>16s} {f:>12.3e} {sc:>8.4f} {100*tt**0.5:>9.4f}%"
          f" {100*bb**0.5:>9.4f}% {100*vv**0.5:>9.4f}% {gain:>9.4f}")
    if gain > best[1]:
        best = (r, gain)
print(f"\n⛔ P1 БАЗА A: похибка {100*tA**0.5:.4f}% (сабміт ~0.24%)  "
      f"дисперсія {100*vA/(bA+vA):.1f}% від MSE (очікування 91–93%)")
print(f"⛔ P2 ціна пілота {PILOT_COST:.3e} = {100*PILOT_COST/NMC/fA:.2f}% ціни семпла")
print(f"\n📣 ВИРОК: найкраще «{best[0]}» → ×{best[1]:.4f} ⟹ "
      f"{1.86762e-07/best[1]:.4e}")
print(f"   {'✅ Є ЩО БРАТИ' if best[1] > 1.05 else '⛔ ПОВЕРХ ПІЛОТА НІЧОГО НЕМА'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/prune90b.npz",
         **{f"V_{r}": np.array(Vt[r]) for r in REG},
         **{f"B_{r}": np.array(Bi[r]) for r in REG},
         **{f"F_{r}": np.array(Fl[r]) for r in REG})
