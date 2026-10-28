"""ЧИ НЕСЕ МИГАННЯ ЗАТВОРІВ ІНФОРМАЦІЮ ПРО ВНЕСОК СЕМПЛА. (ідея юзера)

ЗВІДКИ. Повна маска не повторюється НІКОЛИ (2²¹⁸ конусів на 6200 семплів),
але ОКРЕМІ нейрони мигають постійно — кожен відкритий приблизно у половини
семплів. Тобто по одному затвору статистика багата.

ПИТАННЯ. Найдешевша зведена статистика мигання — частка відкритих затворів
на шарі: `f_i(z) = (1/n) Σ_j 1{p_ij > 0}`, 31 число на семпл, рахується САМО
СОБОЮ під час прогону. Чи корелює вона з внеском семпла у фінал?
Уся похибка живе в ОДНОМУ скалярі `s(z) = ⟨a(z), û⟩`, де `û` — напрямок
середнього ⟹ ціль саме він. Якщо R² помітний, варіанс-редукція = 1/(1−R²).

⚠️ ЩО ЦЕ НЕ ДОВОДИТЬ. Навіть високий R² не дає готового КЗ: контрольна змінна
потребує ВІДОМОГО середнього `E[f_i]`, а воно відоме лише для шару 0 (рівно
0.5, бо `p₀ = z·W₀[:,j]` центрований гаус) — і там воно вже прибите
антитетикою ТОТОЖНО (`f₀(z) + f₀(−z) = 1`). Тому стенд відповідає на питання
«чи є там сигнал», а не «чи є важіль».

КОНТРОЛІ
  * ЧЕСНИЙ СПЛІТ: коефіцієнти на одній половині семплів, R² на іншій;
  * SHUFFLE: ті самі регресори, переміщані по семплах ⟹ мусить дати R² ≈ 0;
  * ПОЗИТИВНИЙ КОНТРОЛЬ: `‖z‖` мусить дати сигнал (мережа однорідна степеня 1,
    `z·∇f = f` ⟹ s росте з радіусом) — якщо і він нуль, винен стенд;
  * f₀ окремим рядком: мусить бути ~0 на антитетичній вибірці (прибито).
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from sub66_torch import MIN_PAIRS, PILOT_PAIRS, TARGET_UTIL, B_BUDGET  # noqa: E402
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
SEEDS = (12345, 777, 24680, 999331)


def r2_split(X, y):
    """R² на ВІДКЛАДЕНІЙ половині семплів (коефіцієнти — на першій)."""
    n = X.shape[0] // 2
    A, b = X[:n], y[:n]
    A = np.concatenate([A, np.ones((A.shape[0], 1))], 1)
    B = np.concatenate([X[n:], np.ones((X.shape[0] - n, 1))], 1)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = y[n:] - B @ w
    return 1.0 - float((r ** 2).sum() / ((y[n:] - y[n:].mean()) ** 2).sum())


def main(nets=6, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    res = {k: [] for k in ('f1_31', 'f0', 'radius', 'усе', 'shuffle')}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
            used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
            npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used)
                                       / (2.0 * per_cut)))
            for sd in SEEDS:
                rng = np.random.default_rng(int(sd))
                rng.standard_normal((PILOT_PAIRS, W_))
                z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                                 device=dev)
                x = torch.cat([z, -z], 0)
                n2 = float(x.shape[0])
                cov = (x.T @ x) / n2
                ev, evec = torch.linalg.eigh(cov.double())
                wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
                rad = x.norm(dim=1).cpu().numpy().astype(np.float64)

                h = x @ (wh @ wc[0])
                F = [(h > 0).to(h.dtype).mean(1)]
                x2 = torch.relu(h)
                for i in range(1, L_):
                    p = x2 @ wc[i]
                    F.append((p > 0).to(p.dtype).mean(1))
                    x2 = torch.relu(p)

                a = x2.double()
                u = a.mean(0)
                u = u / u.norm()
                y = (a @ u).cpu().numpy()
                Fm = torch.stack(F, 1).double().cpu().numpy()

                idx = np.random.default_rng(7).permutation(len(y))
                res['f1_31'].append(r2_split(Fm[:, 1:], y))
                res['f0'].append(r2_split(Fm[:, :1], y))
                res['radius'].append(r2_split(rad[:, None], y))
                res['усе'].append(r2_split(np.concatenate(
                    [Fm, rad[:, None]], 1), y))
                res['shuffle'].append(r2_split(Fm[idx, 1:], y))
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    print(f'\n{len(fs)} мереж × {len(SEEDS)} жеребів, ~{2*npair} семплів на жереб')
    print(f'{"регресор":28s} {"R² (відкладена половина)":>26s} {"1/(1−R²)":>10s}')
    print('-' * 68)
    for nm, key in (('частки відкритих f₁..f₃₁', 'f1_31'),
                    ('f₀ (мусить бути ~0)', 'f0'),
                    ('‖z‖ (позит. контроль)', 'radius'),
                    ('усе разом', 'усе'),
                    ('SHUFFLE (мусить бути ~0)', 'shuffle')):
        v = np.array(res[key])
        m = float(v.mean())
        print(f'{nm:28s} {m:+14.4f} ± {v.std():.4f}      '
              f'{(1.0/(1.0-m) if m < 0.999 else float("inf")):10.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
