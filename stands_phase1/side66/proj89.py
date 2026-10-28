"""proj89 — МК У РЕДУКОВАНОМУ ПІДПРОСТОРІ: ЄДИНЕ, ЩО ДАЄ ПОТРІБНИЙ ПОРЯДОК.

════════════════ ПЕРЕСТАНОВКА ПРІОРИТЕТІВ ════════════════
`hyb89` зміряв: чиста МК при N=5434 дає **0.187%**, замикання — **0.935%**.
МК у 5× ТОЧНІША. Ми весь час полірували гіршу з двох компонент.
Бал ×3.66 (щоб дійти лідера) ⟹ потрібно `N`×3.66 ⟹ **семпл має коштувати
вчетверо дешевше**. Жодна поправка до замикання цього не дасть.

════════════════ ІДЕЯ ════════════════
`tprof88`: ефективний ранг активацій падає **127.6 → 3.1** з глибиною.
Якщо тримати семпл у `r`-вимірних координатах:
    крок = розгортка `c·(PᵀW)` (2rn) + relu (n) + стиснення `a'·P'` (2nr) = 4rn
проти повного `2n²`. При r=32 це рівно ×4, при r=16 — ×8.
Базис `P` беремо з ЗАМИКАННЯ (власні вектори `C^(l)`) — воно й так рахується.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
🚨 Проти цього свідчить `lowrk88` (низький ранг для `C` дав ×0.32) і закон
«негаусовість живе у ХВОСТІ спектру: домінантні компоненти за ЦГТ майже
гаусові». Тобто саме те, що ми відкидаємо, може нести сигнал.
Ставлю: r=64 дасть похибку 0.3…0.6% (проти 0.187% повної МК) — тобто
здешевлення ×2 не окупить втрати точності (шанс 55%). Шанс, що r=32
залишиться під 0.25% (тобто гілка ЖИВА): **20%**.
⚠️ Це ставка з низькою ймовірністю, але з єдиним правильним ПОРЯДКОМ
виграшу — решта перебраного дає ×1.0x, а тут або ×3, або нуль.

════════════════ ЯК МІРЯЄМО ЧЕСНО ════════════════
Порівнюємо при ОДНАКОВОМУ бюджеті: r-вимірний прохід дешевший ⟹ семплів
більше в `2n²/(4rn) = n/(2r)` разів. Дивимось НЕТТО похибку.

ПЛОМБА: r=256 (повний ранг, P=I) мусить відтворити чисту МК бітово.
"""
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 24
DT = torch.float32
SQ2PI = float(np.sqrt(2.0 * np.pi))
NTRUTH = 1 << 18
NBASE = 5434                      # семплів у повній МК за наш бюджет
RS = (16, 32, 64, 128, 256)

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · r {RS} · база N={NBASE}")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def bases(ws, r):
    """Базиси з ЗАМИКАННЯ: топ-r власних векторів `C^(l)` активацій."""
    m = torch.zeros(n, dtype=DT, device=DEV)
    C = torch.eye(n, dtype=DT, device=DEV)
    Ps = []
    for i in range(L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        t = mp / s
        P_, f = Phi(t), phi(t)
        e1 = s * (t * P_ + f)
        e2 = (mp * mp + s * s) * P_ + (mp * s) * f
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        Ca = cp * torch.outer(P_, P_) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        if r >= n:
            Ps.append(None)
        else:
            ev, U = torch.linalg.eigh(Ca.double())
            Ps.append(U[:, -r:].to(DT))
        m, C = e1, Ca
    return Ps, m


def run_proj(ws, r, N, seed, Ps, mref):
    """МК, де стан між шарами тримається в `r` координатах."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(n, dtype=torch.float64, device=DEV)
    done = 0
    while done < N:
        b = min(1 << 13, N - done)
        z = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
        x = torch.cat([z, -z], 0)
        for i in range(L):
            h = x @ ws[i]
            if i == L - 1:
                acc += torch.clamp(h, min=0.0).sum(0).double()
                break
            x = torch.clamp(h, min=0.0)
            if r < n and Ps[i] is not None:
                # СТИСНЕННЯ + РОЗГОРТКА навколо середнього: зберігаємо `m`,
                # проєктуємо лише ФЛУКТУАЦІЮ — інакше втрачаємо перший момент,
                # а він передається у вихід з коефіцієнтом 0.993 (С87).
                mu = x.mean(0, keepdim=True)
                d = x - mu
                x = mu + (d @ Ps[i]) @ Ps[i].T
        done += b
    return acc / (2.0 * done)


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


print(f"\n{'r':>5s} {'ціна кроку':>12s} {'N за бюджет':>13s} {'дешевше в':>11s}")
NS = {}
for r in RS:
    step = 2.0 * n * n if r >= n else 4.0 * r * n
    NS[r] = int(NBASE * (2.0 * n * n) / step)
    print(f"{r:>5d} {step:>12,.0f} {NS[r]:>13,d} {2.0 * n * n / step:>10.2f}×")

res = {r: [] for r in RS}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q).to(DT)
    den = float((tr * tr).mean())
    for r in RS:
        Ps, _ = bases(ws, r)
        p = run_proj(ws, r, NS[r], 41000 + q, Ps, None)
        res[r].append(float(((p.to(DT) - tr) ** 2).mean() / den) ** 0.5)
    if q % 6 == 0:
        print("  net%3d " % q + " ".join(f"r{r}:{100 * res[r][-1]:.3f}%" for r in RS))

print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 76)
print(f"МК У РЕДУКОВАНОМУ ПІДПРОСТОРІ · {NNET} мереж · ОДНАКОВИЙ БЮДЖЕТ")
print("=" * 76)
print(f"{'r':>5s} {'N':>10s} {'віднос.похибка':>16s} {'MSE проти r=256':>17s} {'бал':>12s}")
full = np.array(res[256])
for r in RS:
    a = np.array(res[r])
    gain = (full ** 2).mean() / (a ** 2).mean()
    print(f"{r:>5d} {NS[r]:>10,d} {100 * a.mean():>15.4f}% {'×' + format(gain, '.4f'):>16s}"
          f" {1.8676e-07 / gain:>12.4e}")
br = max(RS, key=lambda r: (full ** 2).mean() / (np.array(res[r]) ** 2).mean())
bg = (full ** 2).mean() / (np.array(res[br]) ** 2).mean()
print(f"\n  🎯 НАЙКРАЩЕ r={br}: ×{bg:.4f} ⟹ бал {1.8676e-07 / bg:.4e}")
print(f"  🔬 ПЛОМБА r=256: {100 * full.mean():.4f}% (hyb89 дав 0.187% при N=5434)")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/proj89.npz",
         **{f"r{r}": np.array(res[r]) for r in RS})
