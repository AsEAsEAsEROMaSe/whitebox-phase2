"""prune90 — ВИКИНУТИ МЕРТВІ НЕЙРОНИ З МК-ПРОГОНУ (питання ЮЗЕРА про алгоритм).

════════════════ ЗВІДКИ ІДЕЯ ════════════════
ЮЗЕР: «нащо власними числами гратись, якщо в нас є всі горки? може треба
оптимальніший спосіб рахувати». `chk90` зміряв: на шарі 32 **31.38%** нейронів
мовчать у >99.9% семплів, на шарі 24 — 25.72%, на шарі 16 — 19.86%. Ми женемо
через них матмули щосемпла і множимо на нуль.

📐 СХЕМА: замикання (яке в нас УЖЕ Є і коштує копійки) дає `t_j` для кожного
нейрона кожного шару ⟹ `p_j = Φ(t_j)`. Нейрони з `p_j < ε` викидаємо ФІЗИЧНО:
`W[i]` стає `n×(n−k)`, `W[i+1]` стає `(n−k)×n`. Матмул дешевшає на `k/n`.
Звільнений бюджет іде В СЕМПЛИ.

🚨 ЦЕ НЕ «ВИМОРОЖУВАННЯ ПАТЕРНІВ» (закрито в `frozen89`, 0.0000% семплів без
перемикача) — там зливали ШАРИ, тут стискаємо МАТРИЦЮ. Інший об'єкт.

════════════════ ЧОМУ ЦЕ МОЖЕ НЕ СПРАЦЮВАТИ ════════════════
Викидання вносить ЗМІЩЕННЯ, а воно НЕ гаситься семплами: `MSE = b² + V/N`.
Тому рахую бал ЧЕСНО за нашою формулою бюджету, а не «похибку до/після».

════════════════ ПЛОМБИ ════════════════
P1 ε=0: жодного нейрона не викинуто ⟹ результат мусить збігтися з базою
   БІТ-У-БІТ (той самий сід).
P2 ЕКОНОМІЯ: підрахунок флопів мусить дати рівно `Σ_i n·(n−k_i)·2` — звіряю
   з прямим підрахунком розмірів матриць.
P3 ЗМІЩЕННЯ НЕ ГАСИТЬСЯ: при N×4 похибка бази падає вдвічі, а похибка
   обрізаної мусить ВИЙТИ НА ПЛАТО. Якщо не вийшла — це не зміщення, і
   формула балу інша.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
 (а) економія при ε=1e-3 буде 12…20% флопів (шанс 60%);
 (б) зміщення при ε=1e-3 буде 0.05…0.30% (шанс 55%) — порівнянне з самою
     МК-похибкою 0.24%, тобто на межі;
 (в) 🎯 чистий виграш у балі >×1.10 — шанс **35%**;
 (г) оптимальний ε буде ДУЖЕ малий (1e-4…1e-3), бо зміщення росте швидше,
     ніж економія (шанс 70%).
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
NMC = 5432
NTRUTH = 1 << 19
DT = torch.float32
SQ2 = 1.4142135623730951
INV_SQ2PI = 0.3989422804014327
EPS = [0.0, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]

W = np.load("/home/trim/wbdata/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · повторів {MREP} · МК {NMC}\n")


def closure_p(ws):
    """p_j = P(нейрон активний) з дешевого замикання — те, що вже є в сабміті."""
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


def prune(ws, P, eps):
    """Викинути нейрони з p_j < eps: стиснути W[i] по стовпцях, W[i+1] по рядках."""
    if eps <= 0:
        return ws, [n] * L, 2 * L * n * n
    keep = [(P[i] >= eps) for i in range(L - 1)]
    out, sizes, fl = [], [], 0
    prev = torch.ones(n, dtype=torch.bool, device=DEV)
    for i in range(L):
        k = keep[i] if i < L - 1 else torch.ones(n, dtype=torch.bool, device=DEV)
        out.append(ws[i][prev][:, k].contiguous())
        sizes.append(int(k.sum()))
        fl += 2 * int(prev.sum()) * int(k.sum())
        prev = k
    return out, sizes, fl


def fwd(ws, z):
    x = z
    for i in range(len(ws)):
        h = x @ ws[i]
        x = torch.relu(h) if i < len(ws) - 1 else h
    return torch.relu(h).mean(0).double()


def mc(ws, seed, N=NMC, nin=None):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(N // 2, n, generator=g, device=DEV, dtype=DT)
    z = torch.cat([z, -z], 0)
    cov = (z.T @ z).double() / z.shape[0]
    z = z @ torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT)
    return fwd(ws, z)


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


V = {e: [] for e in EPS}
B = {e: [] for e in EPS}
FL = {e: [] for e in EPS}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q)
    den = float((tr * tr).sum())
    P = closure_p(ws)
    for e in EPS:
        pw, sz, fl = prune(ws, P, e)
        FL[e].append(fl)
        acc, ms = 0.0, torch.zeros(n, dtype=torch.float64, device=DEV)
        for r in range(MREP):
            p = mc(pw, 808000 + 977 * q + r)
            acc += float(((p - tr) ** 2).sum() / den)
            ms += p
        V[e].append(acc / MREP)                       # повна MSE
        B[e].append(float(((ms / MREP - tr) ** 2).sum() / den))   # зміщення²
    if q % 4 == 0:
        print(f"  net{q:>3d} ({time.time()-t0:.0f} с)", flush=True)

print(f"\n  час {time.time()-t0:.0f} с\n")
base_fl = np.mean(FL[0.0])
b0 = np.mean(B[0.0])
v0 = np.mean(V[0.0]) - b0
print("=" * 100)
print(f"ОБРІЗАННЯ МЕРТВИХ НЕЙРОНІВ · {NNET} мереж × {MREP} повторів")
print("=" * 100)
print(f"{'ε':>8s} {'викинуто':>10s} {'флопів':>9s} {'сер.похибка':>12s} "
      f"{'зміщення':>11s} {'дисперсія':>11s} {'семплів ×':>10s} {'БАЛ ×':>9s}")
best = None
for e in EPS:
    fl = np.mean(FL[e])
    cut = 1.0 - fl / base_fl
    tot = np.mean(V[e])
    bb = np.mean(B[e])
    vv = tot - bb
    # звільнений бюджет → більше семплів; дисперсія падає пропорційно
    scale = fl / base_fl
    new = bb + vv * scale
    gain = (b0 + v0) / max(new, 1e-30)
    print(f"{e:>8.0e} {100*cut:>9.2f}% {100*scale:>8.2f}% {100*tot**0.5:>11.4f}%"
          f" {100*bb**0.5:>10.4f}% {100*vv**0.5:>10.4f}% {1/scale:>10.3f}"
          f" {gain:>9.4f}")
    if best is None or gain > best[1]:
        best = (e, gain)

print(f"\n⛔ P1 (ε=0 = база): похибка {100*np.mean(V[0.0])**0.5:.4f}%  "
      f"зміщення {100*b0**0.5:.4f}%  дисперсія {100*v0**0.5:.4f}%")
print(f"   частка дисперсії в MSE: {100*v0/(b0+v0):.1f}%  (очікування ~91%)")
print(f"\n📣 ВИРОК: найкраще ε={best[0]:.0e} → бал ×{best[1]:.4f}")
print(f"   ⟹ {1.86762e-07/best[1]:.4e}  (рекорд 1.86762e-07)")
print(f"   {'✅ ОКУПАЄТЬСЯ' if best[1] > 1.03 else '⛔ НЕ ОКУПАЄТЬСЯ (зміщення з’їдає економію)'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/prune90.npz",
         eps=np.array(EPS), V=np.array([V[e] for e in EPS]),
         B=np.array([B[e] for e in EPS]), FL=np.array([FL[e] for e in EPS]))
