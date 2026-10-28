"""hyb89 — ГІБРИД: МК ДО ШАРУ j, ДАЛІ ЗАМИКАННЯ ВІД ЕМПІРИЧНИХ m,C.

════════════════ ЧОМУ ЦЕ КАРДИНАЛЬНО ІНШЕ ════════════════
Досі ми робили АБО МК на всі 32 шари, АБО замикання на всі 32, і змішували
РЕЗУЛЬТАТИ з вагою w=0.0883. Режим «МК до шару j, далі замикання від
ЕМПІРИЧНИХ `m,C`» не міряли ЖОДНОГО разу.

Він виграє одразу з двох боків:
  (1) обрізаний прохід коштує `j/L` від повного ⟹ за той самий бюджет
      семплів більше в `L/j` разів ⟹ дисперсія падає в стільки ж;
  (2) детермінований хвіст НЕ ДОДАЄ дисперсії взагалі.
І є зміряна підстава, що шум не пролізе крізь хвіст: С87 дав
`G_σ→вихід = 0.059` — похибка в коваріації передається у вихід у 17× слабше.
🚨 ПРОТИВАГА: `G_m→вихід = 0.993` — похибка в СЕРЕДНЬОМУ передається 1:1,
а хвіст додає зміщення замикання на `L−j` кроках.

════════════════ ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ ════════════════
Ставлю ЗАРАЗ, до першого числа:
  оптимум буде десь на j = 20…28 (шанс 45%), виграш ×1.3…2.0 (шанс 35%),
  або крива виявиться монотонною до j=32 (тобто гібрид нічого не дає) — 40%.
🚨 Головний ризик, який назву наперед: `C^(j)` з `N` семплів — це матриця
256×256, оцінена по N точках. При N~5000 вона ще не вироджена, але шумна,
а замикання далі підносить її до квадрата (`C²` у Мелері) ⟹ шум може
підсилитись. Якщо крива різко гіршає при малих j — саме це й станеться.

════════════════ ЯК МІРЯЄМО ЧЕСНО ════════════════
Бюджет ФІКСОВАНИЙ: `N(j)` підбирається так, щоб `j` шарів МК + `(L−j)` шарів
замикання коштували РІВНО стільки ж, скільки поточний сабміт. Тобто
порівнюємо не «точність», а БАЛ при однаковій ціні.

ПЛОМБИ:
P1  j=L мусить відтворити чисту МК; j=0 — чисте замикання (0.9352%)
P2  при N→∞ гібрид мусить збігтися до «оракул m,C на шарі j + замикання»
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
B = 2.72e11
BUDGET = 0.0838 * B          # стільки флопів сабміт витрачає на семпли
JS = (0, 4, 8, 12, 16, 20, 24, 28, 30, 32)

W = np.load("/tmp/claude-1000/-mnt-win-d-llm-White-Box/"
            "5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/fs_w.npy", mmap_mode="r")
n, L = W.shape[2], W.shape[1]
print(f"{DEV} · мереж {NNET} · j {JS} · бюджет фіксований {BUDGET:.4g} флопів")


def phi(t):
    return torch.exp(-0.5 * t * t) / SQ2PI


def Phi(t):
    return 0.5 * (1.0 + torch.erf(t * 0.7071067811865476))


def n_for(j):
    """Скільки семплів дозволяє бюджет, якщо МК йде лише `j` шарів.

    прохід: 2·n²·j на семпл (антитетика вже врахована в SAMP)
    хвіст:  замикання `L−j` шарів ≈ 2·n³·(L−j) — платиться ОДИН раз
    """
    tail = 2.0 * n ** 3 * max(L - j, 0)
    per = 2.0 * n * n * max(j, 1)
    return max(64, int((BUDGET - tail) / per))


def run(ws, j, N, seed):
    """МК до шару j → емпіричні m,C → замикання до кінця."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    if j == 0:
        m = torch.zeros(n, dtype=DT, device=DEV)
        C = torch.eye(n, dtype=DT, device=DEV)
    else:
        S1 = torch.zeros(n, dtype=torch.float64, device=DEV)
        S2 = torch.zeros(n, n, dtype=torch.float64, device=DEV)
        done = 0
        while done < N:
            b = min(1 << 13, N - done)
            z = torch.randn(b, n, generator=g, device=DEV, dtype=DT)
            x = torch.cat([z, -z], 0)                 # антитетика
            for i in range(j):
                x = torch.clamp(x @ ws[i], min=0.0)
            xd = x.double()
            S1 += xd.sum(0)
            S2 += xd.T @ xd
            done += b
        tot = 2.0 * done
        m = (S1 / tot).to(DT)
        C = (S2 / tot - torch.outer(S1 / tot, S1 / tot)).to(DT)
    # детермінований хвіст
    for i in range(j, L - 1):
        w = ws[i]
        cp = w.T @ (C @ w)
        s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
        mp = m @ w
        t = mp / s
        P, f = Phi(t), phi(t)
        e1 = s * (t * P + f)
        e2 = (mp * mp + s * s) * P + (mp * s) * f
        var = torch.clamp(e2 - e1 * e1, min=1e-30)
        Ca = cp * torch.outer(P, P) + 0.5 * (cp * cp) * torch.outer(f / s, f / s)
        Ca = Ca + torch.diag(var - torch.diagonal(Ca))
        m, C = e1, Ca
    if j >= L:
        return m                                      # чиста МК: m уже вихід
    w = ws[L - 1]
    cp = w.T @ (C @ w)
    s = torch.sqrt(torch.clamp(torch.diagonal(cp), min=1e-30))
    t = (m @ w) / s
    return torch.nan_to_num(s * (t * Phi(t) + phi(t)), nan=0.0,
                            posinf=0.0, neginf=0.0)


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


print(f"\n{'j':>4s} {'N(бюджет)':>11s} {'семпл коштує':>13s}")
for j in JS:
    print(f"{j:>4d} {n_for(j):>11,d} {2.0 * n * n * max(j, 1):>13,.0f}")

res = {j: [] for j in JS}
t0 = time.time()
for q in range(NNET):
    ws = [torch.as_tensor(np.ascontiguousarray(W[q, i]).astype(np.float32),
                          device=DEV) for i in range(L)]
    tr = truth(ws, 7000 + q).to(DT)
    den = float((tr * tr).mean())
    for j in JS:
        p = run(ws, j, min(n_for(j), 1 << 15), 31000 + q)
        res[j].append(float(((p - tr) ** 2).mean() / den) ** 0.5)
    if q % 8 == 0:
        print(f"  net{q:>3d} " + " ".join(f"j{j}:{100 * res[j][-1]:.3f}%"
                                          for j in (0, 16, 28, 32)))

print(f"\n  час {time.time() - t0:.0f} с\n")
print("=" * 78)
print(f"ГІБРИД ПРИ ФІКСОВАНОМУ БЮДЖЕТІ · {NNET} мереж · L={L}")
print("=" * 78)
print(f"{'j':>4s} {'N':>10s} {'віднос.похибка':>16s} {'проти j=32 (чиста МК)':>23s}")
r32 = np.array(res[32])
for j in JS:
    a = np.array(res[j])
    print(f"{j:>4d} {min(n_for(j), 1 << 15):>10,d} {100 * a.mean():>15.4f}%"
          f" {'×' + format((r32 ** 2).mean() ** 0.5 / (a ** 2).mean() ** 0.5, '.4f'):>22s}")
bj = min(JS, key=lambda j: (np.array(res[j]) ** 2).mean())
gain = (r32 ** 2).mean() / (np.array(res[bj]) ** 2).mean()
print(f"\n  🎯 НАЙКРАЩЕ j={bj}: виграш MSE ×{gain:.3f} проти чистої МК")
print(f"     ⟹ бал 1.8676e-07 → {1.8676e-07 / gain:.4e}")
print(f"  🔬 P1: j=0 дає {100 * np.array(res[0]).mean():.4f}% (замикання 0.9352%) "
      f"{'✅' if abs(np.array(res[0]).mean() / 0.009352 - 1) < 0.1 else '⚠️'}")
np.savez("/mnt/win_d/llm/White-Box/arena62/logs/hyb89.npz",
         **{f"j{j}": np.array(res[j]) for j in JS})
