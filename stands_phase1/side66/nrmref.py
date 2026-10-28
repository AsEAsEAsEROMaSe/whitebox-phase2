"""ЯКА НАША ВЛАСНА ВІДНОСНА ПОХИБКА? — калібр для вироку по обрізанню рангу.

ЗВІДКИ. `deeprank.py` дав похибки обрізання у ВІДСОТКАХ від ‖E[relu(x₃₁)]‖.
Щоб сказати «багато» чи «мало», треба ТА САМА величина для діючого сабміту.
Наш бал 2.05e-07 = MSE × 0.1073 ⟹ MSE = 1.91e-06 ⟹ відносна = √MSE / nrm.

🚨 БЕЗ ЦЬОГО ЧИСЛА ВИРОК ВИСИТЬ НА МОЇЙ ОЦІНЦІ `nrm≈0.4` — а вся різниця між
«мертво з запасом» і «гранично» саме тут.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_                            # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
MSE_SUB = 1.91e-06          # 2.05e-07 / 0.1073, крива підтверджена на грейдері


def truth_last(wc, dev, seed, npair=131_072, chunk=16384):
    acc = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(L_):
            a = torch.relu(a @ wc[i])
        acc += a.double().sum(0)
        done += k
    return acc / float(2 * npair)


def main(nets=6, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    nr = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            T = truth_last(wc, dev, 410_000 + 7919 * n)
            nr.append(float((T ** 2).mean().sqrt()))
        del W, wc
        torch.cuda.empty_cache()
    nr = np.array(nr)
    print(f'\n{nets} мереж, істина 262144 семплів')
    print(f'nrm = RMS(E[relu(x₃₁)]) : медіана {np.median(nr):.6f}  '
          f'розкид {nr.min():.6f}–{nr.max():.6f}')
    rel = 100.0 * np.sqrt(MSE_SUB) / np.median(nr)
    print(f'\n📐 НАША ВЛАСНА ВІДНОСНА ПОХИБКА = √{MSE_SUB:.3e} / {np.median(nr):.4f}'
          f' = **{rel:.4f}%**')
    print(f'\n{"додаткова похибка":>18s} {"сумарна":>9s} {"MSE ×":>8s} '
          f'{"треба економії":>15s}')
    print('-' * 56)
    for e in (0.014, 0.044, 0.107, 0.27, 0.35, 0.60):
        tot = np.sqrt(rel ** 2 + e ** 2)
        print(f'{e:17.3f}% {tot:8.4f}% {(tot/rel)**2:7.3f}× '
              f'{(tot/rel)**2:14.3f}×')
    print('-' * 56)
    print('🔑 читати так: «треба економії» — у скільки разів має подешевшати '
          'прохід,\n   щоб обрізання рангу вийшло В НУЛЬ. Порівняти з '
          'колонкою «економія» в `deeprank.py`.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
