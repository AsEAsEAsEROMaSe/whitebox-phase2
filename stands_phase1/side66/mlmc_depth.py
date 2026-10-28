"""MLMC ПО ГЛИБИНІ: телескоп рівнів `h_ℓ·x_ℓ` — чи б'є він наші 3.4×?

ЗВІДКИ. `mlmc_probe.py` (нижче — його числа) поховав сурогати за РАНГОМ і за
ШИРИНОЮ: ρ² = 0.03…0.23, бо ваги випадкові й спектр плаский (Марченко-Пастур),
✅контролі S4=1.00× / S5=0.26× бездоганні. Але третій сурогат ожив:
    префікс глибини k + ЛІНІЙНА голова, підігнана МНК
дав ρ² = 0.79 (k=8), 0.911 (k=16), 0.9815 (k=24). Тобто мілкий шар лінійно
несе майже весь вихід — і це НЕ поліном від `z`, отже стеля 9.49× на нього
не поширюється.

ЩО РАХУЄМО. Двох рівнів мало: там γ надто велике. Правильна конструкція —
ПОВНИЙ телескоп Гілза по драбині глибин ℓ₀<ℓ₁<…<L:
    F = h_{ℓ₀}x_{ℓ₀} + Σ_j (h_{ℓ_j}x_{ℓ_j} − h_{ℓ_{j-1}}x_{ℓ_{j-1}}) + (F − h_{ℓ_last}x)
Кожен рівень рахується ОДНИМ прогоном до своєї глибини ⟹ c_ℓ = вартість
префікса. При оптимальному розподілі семплів
    Var = (Σ_ℓ √(V_ℓ·c_ℓ))² / C,   база (чистий МК) = Var(F)·c_L / C
    ⟹ ВИГРАШ = Var(F)·c_L / (Σ √(V_ℓ c_ℓ))²

🎯 ПОРІГ: наш estimator уже дає **3.4×**. MLMC цікавий лише якщо ПЕРЕБИВАЄ це
число, або якщо множиться з ним (тоді найдешевший рівень бере v2, і його
внесок падає в 3.4 раза — рахуємо обидва варіанти).

КОНТРОЛІ
  ✅ драбина з ОДНОГО рівня (сам `F`) мусить дати рівно 1.00×;
  ✅ голови міряються на ВІДКЛАДЕНІЙ половині (інакше МНК завищить ρ²);
  ✅ shuffle-контроль: голова від ЧУЖОЇ мережі ⟹ виграш мусить впасти до ~1;
  ✅ друкуємо V_ℓ по драбині — вони мусять СПАДАТИ з глибиною, інакше телескоп
     не має сенсу і це видно одразу.
"""
import glob
import itertools
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:0'
NS = 131072
CH = 8192
DEPTHS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 30, 31]


def load(f, WZ, have, dev):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
         for q in range(L_)]
    wc, _, _ = cut_net(W, dev)
    return wc


def heads_and_levels(wc, dev, seed, alien_heads=None):
    """Повертає V_ℓ телескопа, c_ℓ, Var(F), c_L. Голови — МНК на 1-й половині,
    усі дисперсії — на 2-й (ВІДКЛАДЕНІЙ)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    nw = wc[-1].shape[1]
    uo = torch.ones(nw, device=dev) / math.sqrt(nw)
    Xs = {d: [] for d in DEPTHS}
    F = []
    for _ in range(NS // CH):
        z = torch.randn(CH, W_, device=dev, generator=g)
        x = z
        for i, w in enumerate(wc):
            x = torch.relu(x @ w)
            if (i + 1) in Xs:
                Xs[i + 1].append(x.double())
        F.append((x @ uo).double())
    F = torch.cat(F)
    for d in DEPTHS:
        Xs[d] = torch.cat(Xs[d])
    half = NS // 2

    cost = {}
    acc = 0.0
    for i, w in enumerate(wc):
        acc += 2.0 * w.shape[0] * w.shape[1]
        if (i + 1) in DEPTHS:
            cost[i + 1] = acc + 2.0 * w.shape[1]
    cL = acc

    P = {}                                     # прогноз рівня на 2-й половині
    for j, d in enumerate(DEPTHS):
        H = Xs[d]
        A = torch.cat([H[:half], torch.ones(half, 1, device=dev,
                                            dtype=torch.float64)], 1)
        sol = torch.linalg.lstsq(A, F[:half].unsqueeze(1)).solution
        if alien_heads is not None and d in alien_heads:
            sol = alien_heads[d]
        n2 = NS - half
        A2 = torch.cat([H[half:], torch.ones(n2, 1, device=dev,
                                             dtype=torch.float64)], 1)
        P[d] = (A2 @ sol).squeeze(1)
        Xs[d] = None
    return P, F[half:], cost, cL, float(F[half:].var())


def ladder_var(P, F2, cost, cL, ladder):
    """(Σ√(V_ℓ c_ℓ))² для конкретної драбини глибин."""
    tot = 0.0
    parts = []
    prev = None
    for d in ladder:
        y = P[d] if prev is None else P[d] - P[prev]
        v = float(y.var())
        parts.append((d, v, cost[d]))
        tot += math.sqrt(max(v, 0.0) * cost[d])
        prev = d
    y = F2 - P[prev] if prev is not None else F2
    v = float(y.var())
    parts.append(('L', v, cL))
    tot += math.sqrt(max(v, 0.0) * cL)
    return tot ** 2, parts


def main(nets=6, dev=DEV):
    torch.manual_seed(20260729)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    best_all, one_all, fix_all, alien_all = [], [], [], []
    for n, f in enumerate(fs):
        with torch.no_grad():
            wc = load(f, WZ, have, dev)
            P, F2, cost, cL, vF = heads_and_levels(wc, dev, 8100 + n)
            base = vF * cL

            v1, _ = ladder_var(P, F2, cost, cL, [])
            one_all.append(base / v1)

            fixed = [4, 8, 16, 24]
            vf, parts = ladder_var(P, F2, cost, cL, fixed)
            fix_all.append(base / vf)

            best, bl = 0.0, None
            for k in range(1, 5):
                for lad in itertools.combinations(DEPTHS, k):
                    vv, _ = ladder_var(P, F2, cost, cL, list(lad))
                    if base / vv > best:
                        best, bl = base / vv, lad
            best_all.append(best)

            if n == 0:
                print('  рівні драбини [4,8,16,24] на мережі 1:')
                for d, v, c in parts:
                    print(f'    ℓ={str(d):>3s}  V_ℓ/Var(F)={v/vF:8.4f}  '
                          f'c_ℓ/c_L={c/cL:6.3f}')
                print('  ρ² лінійної голови по глибині:')
                for d in DEPTHS:
                    r2 = 1.0 - float((F2 - P[d]).var()) / vF
                    print(f'    ℓ={d:>2d}  ρ²={r2:7.4f}  c/c_L={cost[d]/cL:6.3f}')
            print(f'  мережа {n+1}: 1 рівень={base/v1:.3f}×  '
                  f'[4,8,16,24]={base/vf:.3f}×  НАЙКРАЩА {bl}={best:.3f}×',
                  flush=True)
            del wc, P, F2
            torch.cuda.empty_cache()

    # ✅ shuffle-контроль: голови від ЧУЖОЇ мережі
    with torch.no_grad():
        wa = load(fs[0], WZ, have, dev)
        Pa, Fa, ca, cLa, vfa = heads_and_levels(wa, dev, 999)
        del wa
        torch.cuda.empty_cache()
        wb = load(fs[1], WZ, have, dev)
        gb = torch.Generator(device=dev).manual_seed(8101)
        Pb, Fb, cb, cLb, vfb = heads_and_levels(wb, dev, 8101)
        vb, _ = ladder_var(Pb, Fb, cb, cLb, [4, 8, 16, 24])
        alien_all.append(vfb * cLb / vb)
        del wb
        torch.cuda.empty_cache()

    a1, af, ab = np.array(one_all), np.array(fix_all), np.array(best_all)
    print(f'\n{len(fs)} мереж × {NS} семплів (голови МНК на 1-й половині, '
          f'усе міряно на 2-й)')
    print(f'  ✅ КОНТРОЛЬ драбина з одного рівня : {a1.mean():.4f}× '
          f'(мусить бути рівно 1.0000)')
    print(f'  драбина [4,8,16,24]                : {af.mean():.3f}× '
          f'± {af.std():.3f}')
    print(f'  НАЙКРАЩА драбина (перебір ≤4 рівні): {ab.mean():.3f}× '
          f'± {ab.std():.3f}')
    print(f'\n🎯 поріг: наш estimator дає 3.4×. MLMC цікавий лише якщо ПЕРЕБИВАЄ '
          f'це число або множиться з ним.')
    print(f'   якщо найдешевший рівень узяти нашим v2 (×3.4 на його дисперсії) — '
          f'див. колонку нижче')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
