"""КАЛІБРУВАННЯ ВАРІАЦІЙНОЇ ШИРИНИ ОКРЕМО ДЛЯ КОЖНОЇ З ЧОТИРЬОХ ОПОР.

════════ НАВІЩО ════════
`varsub.py` сканує `α` у вікні 0.996–1.005, знайденому для `m`. Але кожна з
чотирьох опор сабміту — це ПРОГНОЗ СВОЄЇ величини:
    `s_cl` → σ доактивацій ·  `m_cl` → E[relu] ·
    `q_cl` → E[relu²]       ·  `f_cl` → P(p>0)
і немає причин, щоб найкращий ефективний гаус був для них ОДИН І ТОЙ САМИЙ.
Якщо оптимум для `q` лежить, скажімо, при 1.02, вузька сітка його не зловить,
і канал буде оголошено мертвим через сітку, а не через механізм.

📣 ПЕРЕДБАЧЕННЯ: оптимальні `α` РОЗІЙДУТЬСЯ між опорами (для `m` вужчий гаус
компенсує важкі хвости, а `q` = другий момент, на який хвости діють у
ПРОТИЛЕЖНИЙ бік ⟹ 📣 для `q` оптимум **α > 1**).
🟢 Вартий другого прогону `varsub`, якщо хоч для однієї опори з великим λ
   (`q`: 0.042, `g`: 0.027) виграш ≥1.5× і оптимум ПОЗА вікном 0.996–1.005.
🚨 ПЛОМБА: `α=1` — рівно базове замикання; ⛔перенос — фіт на непарних мережах.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi               # noqa: E402
from varsub import closure_var                                     # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:2'
K = 8
TR_PAIRS, CHUNK = 131_072, 32768
GRID = tuple(round(0.97 + 0.0025 * k, 6) for k in range(0, 33))
NAMES = ('s (σ доакт.)', 'm (E[relu])', 'q (E[relu²])', 'g (P[p>0])')


def truth_stats(wc, dev, seed, ksig=K):
    """Істинні `σ(p)`, `E[a]`, `E[a²]`, `P(p>0)` для шарів 1..ksig."""
    d = min(ksig + 1, len(wc))
    n = float(2 * TR_PAIRS)
    S1 = [torch.zeros(wc[i].shape[1], dtype=torch.float64, device=dev)
          for i in range(d)]
    S2 = [torch.zeros_like(t) for t in S1]
    A1 = [torch.zeros_like(t) for t in S1]
    A2 = [torch.zeros_like(t) for t in S1]
    Q = [torch.zeros_like(t) for t in S1]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TR_PAIRS:
        k = min(CHUNK, TR_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(d):
            p = a @ wc[i]
            pd = p.double()
            S1[i] += pd.sum(0)
            S2[i] += (pd * pd).sum(0)
            Q[i] += (pd > 0).double().sum(0)
            a = torch.relu(p)
            ad = a.double()
            A1[i] += ad.sum(0)
            A2[i] += (ad * ad).sum(0)
        done += k
    sg = [((S2[i] / n - (S1[i] / n) ** 2).clamp_min(1e-20)).sqrt()
          for i in range(1, d)]
    return (sg, [A1[i] / n for i in range(1, d)],
            [A2[i] / n for i in range(1, d)], [Q[i] / n for i in range(1, d)])


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: оптимальні α РОЗІЙДУТЬСЯ; для `q` очікую α>1')
    print(f'   мереж {nets}, сітка α {GRID[0]}..{GRID[-1]}\n')

    # E[a] по опорах × α; окремо фіт-половина (непарні) і тест (парні)
    E = np.zeros((4, len(GRID)))
    EA = np.zeros((4, len(GRID)))
    EB = np.zeros((4, len(GRID)))
    N = np.zeros(4)
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            tru = truth_stats(wc, dev, 730_000 + 991 * n)
            for j, a in enumerate(GRID):
                cl = closure_var(wc, K, dev, a, s_raw=False)
                # порядок closure_var: (s, m, f=Φ, q) → зводимо до (s,m,q,g)
                cand = (cl[0], cl[1], cl[3], cl[2])
                for c in range(4):
                    e = sum(float(((cand[c][i] - tru[c][i]) ** 2).sum())
                            for i in range(len(tru[c])))
                    E[c, j] += e
                    (EA if n % 2 else EB)[c, j] += e
            for c in range(4):
                N[c] += sum(float((t ** 2).sum()) for t in tru[c])
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    j1 = GRID.index(1.0)
    print(f'\n{nets} мереж, шари 1..{K}. RMS-похибка опори у % від ‖істини‖:')
    print(f'{"опора":>14s} {"λ сабміту":>10s} {"α=1":>9s} {"α*":>8s} '
          f'{"при α*":>9s} {"виграш":>8s} {"⛔перенос":>9s}')
    print('-' * 76)
    LAMS = (0.0088, 0.0092, 0.0415, 0.0273)   # max|λ| відповідної родини
    out = []
    for c in range(4):
        base = (E[c, j1] / N[c]) ** 0.5 * 100
        jb = int(np.argmin(E[c]))
        bst = (E[c, jb] / N[c]) ** 0.5 * 100
        ja = int(np.argmin(EA[c]))
        xf = (EB[c, j1] / EB[c, ja]) ** 0.5
        out.append((GRID[jb], base / max(bst, 1e-12), xf))
        print(f'{NAMES[c]:>14s} {LAMS[c]:10.4f} {base:8.3f}% '
              f'{GRID[jb]:8.4f} {bst:8.3f}% '
              f'{base / max(bst, 1e-12):7.3f}× {xf:8.3f}×')

    print(f'\n🔑 оптимальні α: ' + ' · '.join(
        f'{NAMES[c].split()[0]}={out[c][0]:.4f}' for c in range(4)))
    wide = [c for c in range(4) if not (0.996 <= out[c][0] <= 1.005)]
    print(f'   ПОЗА вікном varsub (0.996–1.005): '
          f'{[NAMES[c].split()[0] for c in wide] or "нема"}')
    hot = [c for c in (2, 3) if out[c][2] >= 1.5]
    print(f'🔑 ВИРОК: другий прогін varsub '
          f'{"ПОТРІБЕН" if hot else "НЕ потрібен"} '
          f'(опори з великим λ і переносним виграшем ≥1.5×: '
          f'{[NAMES[c].split()[0] for c in hot] or "нема"})')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
