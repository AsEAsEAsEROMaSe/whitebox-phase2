"""ЧИ ПРАВИЛЬНИЙ МНОЖНИК ПЕРЕНОСУ КОВАРІАЦІЙНОЇ НЕВ'ЯЗКИ. (бічний стенд до С66)

ЗВІДКИ. `sigma_carry.py` несе нев'язку `S` правилом
    S ← (g⊗g) ⊙ (Wᵀ S W),      g_j = частка семплів із p_j > 0
і доставлена σ-поправка дає 1.06× там, де оракульна дає 2.83×.

ГІПОТЕЗА. Множник узятий як ДОБУТОК МАРГІНАЛЬНИХ ймовірностей, а мусить бути
СПІЛЬНОЮ. Для гаусового `p` теорема Прайса дає точну похідну
    ∂E[relu(p_i)relu(p_j)] / ∂C_ij = P(p_i > 0, p_j > 0)
(бо relu' = 1{·>0}). Отже правильне правило переносу першого порядку —
    S ← P ⊙ (Wᵀ S W),          P_ij = P(p_i>0, p_j>0)
На ДІАГОНАЛІ це вирішальне: P_ii = g_i ≈ 0.5, а код бере g_i² ≈ 0.25.
⟹ діагональ (а `ds_c` — це саме діагональ) затухає ВДВІЧІ швидше за правильну
НА КОЖНОМУ ШАРІ. Через k шарів амплітуда занижена в ~2^k.

ДИЗАЙН — ІЗОЛЮЄМО ОДНУ ОПЕРАЦІЮ, не накопичення.
Для кожного шару беремо ІСТИННУ нев'язку на вході (з оракульної коваріації на
4М семплів), проганяємо ОДИН крок і порівнюємо з ІСТИННОЮ нев'язкою на виході:
    крок через МАТРИЦЮ:  S_p = Wᵀ S W                 ← мусить бути ТОЧНИМ (лінійність)
    крок через RELU:     правило A = (g⊗g) ⊙ S_p      ← нинішнє
                         правило B = P ⊙ S_p          ← гіпотеза
Так «де ламається» відповідає не «на шарі k», а «в якій операції».

КОНТРОЛІ
  * позитивний контроль реалізації: крок через матрицю мусить дати R² = 1.000
    і масштаб 1.000 — якщо ні, винен код або істина, а не правило;
  * істина рахується ДВОМА ПОЛОВИНАМИ, друкуємо розбіжність половин ⟹ видно,
    скільки в цифрах шуму оракула;
  * і накопичувальний варіант (як у проді) — щоб бачити ціну 32 кроків;
  * `g` і `P` беруться з ОДНИХ І ТИХ САМИХ семплів ⟹ різниця лише в правилі.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from sub66_torch import (MIN_PAIRS, PILOT_PAIRS, SEED,        # noqa: E402
                         TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
SEEDS = (12345, 777, 24680, 999331)
SHOW = (1, 2, 3, 4, 6, 8, 12, 16, 24, 31)


def relu_cov_true(w0):
    """ТОЧНА центрована коваріація `relu(z W₀)` при `z ~ N(0,I)` (арккосинус)."""
    S = (w0.T @ w0).double()
    s = S.diagonal().clamp_min(1e-30).sqrt()
    rho = (S / torch.outer(s, s)).clamp(-1.0, 1.0)
    e2 = torch.outer(s, s) / (2.0 * math.pi) * (
        (1.0 - rho * rho).clamp_min(0.0).sqrt() + rho * (math.pi / 2 + torch.asin(rho)))
    return e2 - torch.outer(s, s) / (2.0 * math.pi)


def truth_cov(wc, dev, seed, n_tot, chunk=65536):
    """Оракульні ЦЕНТРОВАНІ коваріації активацій кожного шару, двома половинами."""
    L = len(wc)
    halves = []
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    for _ in range(2):
        acc = [torch.zeros(w.shape[1], w.shape[1], dtype=torch.float64, device=dev)
               for w in wc]
        accm = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
        done = 0
        while done < n_tot // 2:
            n = min(chunk, n_tot // 2 - done)
            x = torch.randn(n, W_, generator=gen, device=dev)
            for i in range(L):
                x = torch.relu(x @ wc[i])
                acc[i] += (x.T @ x).double()
                accm[i] += x.double().sum(0)
            done += n
        nn = float(n_tot // 2)
        halves.append([(acc[i] / nn - torch.outer(accm[i] / nn, accm[i] / nn))
                       for i in range(L)])
    C = [0.5 * (halves[0][i] + halves[1][i]) for i in range(L)]
    # шум оракула: розбіжність половин відносно самої коваріації
    noise = [float((halves[0][i] - halves[1][i]).pow(2).sum().sqrt()
                   / (2.0 * C[i].pow(2).sum().sqrt())) for i in range(L)]
    return C, noise


def draw_sample(wc, seed, dev):
    """Прогін ВИБІРКИ рівно як у sub66: пілот з'їдає потік, антитетика, вибілювання."""
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
    npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used) / (2.0 * per_cut)))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    x = torch.cat([z, -z], 0)
    n2 = float(x.shape[0])
    cov = (x.T @ x) / n2
    ev, evec = torch.linalg.eigh(cov.double())
    wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
    xs, ps = [], [None]
    x = torch.relu(x @ (wh @ wc[0]))
    xs.append(x)
    for i in range(1, len(wc)):
        p = x @ wc[i]
        ps.append(p)
        x = torch.relu(p)
        xs.append(x)
    return xs, ps, n2


def cmp(a, b):
    """R², кореляція і відношення масштабів `a` (оцінка) проти `b` (істина)."""
    tb = float(b.pow(2).sum())
    ta = float(a.pow(2).sum())
    r2 = 1.0 - float((a - b).pow(2).sum()) / tb
    cc = float((a * b).sum()) / math.sqrt(max(ta * tb, 1e-300))
    return r2, cc, math.sqrt(ta / tb)


def main(nets=4, truth_n=4_194_304, dev=DEV, seeds=SEEDS):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    tab, noi = {}, {}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            C, noise = truth_cov(wc, dev, 90001 + k, truth_n)
            C[0] = relu_cov_true(wc[0])       # шар 0 відомий ТОЧНО ⟹ без шуму оракула
            for i, v in enumerate(noise):
                noi.setdefault(i, []).append(v)

            for sd in seeds:
                xs, ps, n2 = draw_sample(wc, sd, dev)
                # істинні нев'язки на КОЖНОМУ шарі (виміряні, не перенесені)
                Sx = []
                for i in range(len(wc)):
                    xd = xs[i].double()
                    m = xd.sum(0) / n2
                    Sx.append((xd.T @ xd) / n2 - torch.outer(m, m) - C[i])
                Sx[0] = (lambda xd, m: (xd.T @ xd) / n2 - torch.outer(m, m)
                         - relu_cov_true(wc[0]))(xs[0].double(), xs[0].double().sum(0) / n2)

                S_run = Sx[0].clone()          # накопичувальний (як у проді), правило A
                S_runB = Sx[0].clone()         # накопичувальний, правило B
                for i in range(1, len(wc)):
                    Wd = wc[i].double()
                    pd = ps[i].double()
                    M = (ps[i] > 0).to(ps[i].dtype)
                    g = (M.sum(0) / n2).double()
                    P = ((M.T @ M) / n2).double()

                    # істинна нев'язка на p (лінійність ⟹ мусить збігтися з переносом)
                    mp = pd.sum(0) / n2
                    Cp_true = Wd.T @ (C[i - 1] @ Wd)
                    Sp_true = (pd.T @ pd) / n2 - torch.outer(mp, mp) - Cp_true
                    Sp_carry = Wd.T @ (Sx[i - 1] @ Wd)

                    out_true = Sx[i]
                    A = Sp_true * torch.outer(g, g)
                    B = Sp_true * P
                    Ar = Sp_carry * torch.outer(g, g)
                    Br = Sp_carry * P
                    S_run = (Wd.T @ (S_run @ Wd)) * torch.outer(g, g)
                    S_runB = (Wd.T @ (S_runB @ Wd)) * P

                    rec = {
                        'lin': cmp(Sp_carry, Sp_true),
                        'A': cmp(A, out_true),
                        'B': cmp(B, out_true),
                        'Ad': cmp(A.diagonal(), out_true.diagonal()),
                        'Bd': cmp(B.diagonal(), out_true.diagonal()),
                        'runA': cmp(S_run.diagonal(), out_true.diagonal()),
                        'runB': cmp(S_runB.diagonal(), out_true.diagonal()),
                        'ArA': cmp(Ar.diagonal(), out_true.diagonal()),
                        'BrB': cmp(Br.diagonal(), out_true.diagonal()),
                    }
                    tab.setdefault(i, []).append(rec)
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    def agg(i, key, j):
        return float(np.mean([r[key][j] for r in tab[i]]))

    print(f'\n{len(fs)} мереж × {len(seeds)} жеребів, істина {truth_n} двома половинами')
    print('шум оракула (розбіжність половин / ‖C‖):', end=' ')
    print(' '.join(f'{i}:{np.mean(noi[i]):.1e}' for i in (0, 1, 8, 16, 31)))

    print('\n=== ОДИН КРОК, ІЗОЛЬОВАНО (вхід = ІСТИННА нев\'язка) ===')
    print(f'{"шар":>4s} | {"лін.R²":>8s} | {"A:g⊗g R²":>9s} {"corr":>6s} {"мсштб":>6s}'
          f' | {"B:P R²":>8s} {"corr":>6s} {"мсштб":>6s}')
    print('-' * 74)
    for i in SHOW:
        if i not in tab:
            continue
        print(f'{i:4d} | {agg(i,"lin",0):8.4f} | {agg(i,"A",0):9.4f} {agg(i,"A",1):6.3f}'
              f' {agg(i,"A",2):6.3f} | {agg(i,"B",0):8.4f} {agg(i,"B",1):6.3f}'
              f' {agg(i,"B",2):6.3f}')

    print('\n=== ТІЛЬКИ ДІАГОНАЛЬ (це і є `ds_c`) ===')
    print(f'{"шар":>4s} | {"A R²":>8s} {"corr":>6s} {"мсштб":>6s}'
          f' | {"B R²":>8s} {"corr":>6s} {"мсштб":>6s}')
    print('-' * 60)
    for i in SHOW:
        if i not in tab:
            continue
        print(f'{i:4d} | {agg(i,"Ad",0):8.4f} {agg(i,"Ad",1):6.3f} {agg(i,"Ad",2):6.3f}'
              f' | {agg(i,"Bd",0):8.4f} {agg(i,"Bd",1):6.3f} {agg(i,"Bd",2):6.3f}')

    print('\n=== НАКОПИЧУВАЛЬНО (як у проді: несемо з шару 0), діагональ ===')
    print(f'{"шар":>4s} | {"A R²":>8s} {"corr":>6s} {"мсштб":>6s}'
          f' | {"B R²":>8s} {"corr":>6s} {"мсштб":>6s}')
    print('-' * 60)
    for i in SHOW:
        if i not in tab:
            continue
        print(f'{i:4d} | {agg(i,"runA",0):8.4f} {agg(i,"runA",1):6.3f}'
              f' {agg(i,"runA",2):6.3f} | {agg(i,"runB",0):8.4f}'
              f' {agg(i,"runB",1):6.3f} {agg(i,"runB",2):6.3f}')

    print('\nЧИТАННЯ: «лін.R²» мусить бути 1.0000 — це позитивний контроль коду.')
    print('Якщо у правила A масштаб ≈0.5, а у B ≈1.0 — гіпотеза Прайса підтверджена.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
         truth_n=int(sys.argv[2]) if len(sys.argv) > 2 else 4_194_304)
