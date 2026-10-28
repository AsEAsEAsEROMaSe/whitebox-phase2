"""РОЗПОДІЛ ПОМЕРЕЖНОЇ MSE — ЧИ ТРИМАЄ БАЛ ХВІСТ. (мета: 1e-8)

ГІПОТЕЗА ЮЗЕРА: на ДЕЯКИХ мережах ми вже виходимо на 1e-8, а середнє 2.2e-7
⟹ бал тримає ХВІСТ, і лікувати треба його, а не все підряд.

ЧОМУ ЦЕ ВИРІШАЛЬНО. Фінальний бал — СЕРЕДНЄ MSE по мережах (`scoring.py`),
а не медіана. При важкому хвості середнє визначають кілька мереж, і тоді
20× можна взяти, не чіпаючи решту.

ЩО ДРУКУЄМО
  * квантилі помережної MSE і відношення середнє/медіана — міра важкості хвоста;
  * скільки мереж дають ≤1e-8, ≤3e-8, ≤1e-7;
  * яку частку СУМИ дають найгірші 10% і 25% мереж — це і є ціна хвоста;
  * що з хвостом корелює: ‖a_true‖, ‖a‖²·(відн. похибка)², сама відн. похибка
    ⟹ якщо хвіст = лише масштаб ‖a‖, то відносна похибка стала і хвіст
    НЕ лікується (він у самій задачі); якщо відносна похибка теж гуляє —
    є що лікувати.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from transport_r2 import cut_net                              # noqa: E402
from blink_cv import oracle, run_one                          # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331)


def main(nets=32, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    mse, nrm, rel = [], [], []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            a_true, _ = oracle(wc, dev, 90001 + k, TRUTH_N)
            ee = []
            for sd in SEEDS:
                raw, v, _ = run_one(wc, sd, dev)
                ee.append(((raw - v - a_true) ** 2).mean().item())
            mse.append(float(np.mean(ee)))
            nrm.append(float(a_true.norm()))
            rel.append(float(np.sqrt(np.mean(ee)) * np.sqrt(a_true.numel())
                             / a_true.norm()))
        print(f'  мережа {k+1}/{len(fs)}: MSE {mse[-1]:.3e}', flush=True)

    mse = np.array(mse); nrm = np.array(nrm); rel = np.array(rel)
    o = np.sort(mse)
    print(f'\n{len(mse)} мереж × {len(SEEDS)} жеребів')
    print(f'{"середнє":>10s} {np.mean(mse):.4e}   ← це і є бал')
    print(f'{"медіана":>10s} {np.median(mse):.4e}')
    print(f'{"серед/мед":>10s} {np.mean(mse)/np.median(mse):.2f}× '
          f'(1.0 = рівний розподіл, >3 = важкий хвіст)')
    print(f'\nквантилі: min {o[0]:.3e} · 25% {np.percentile(mse,25):.3e}'
          f' · 75% {np.percentile(mse,75):.3e} · max {o[-1]:.3e}')
    print(f'розмах max/min: {o[-1]/o[0]:.1f}×')
    for t in (1e-8, 3e-8, 1e-7, 3e-7):
        print(f'  мереж із MSE ≤ {t:.0e}: {int((mse <= t).sum())}/{len(mse)}')
    k10 = max(1, len(mse) // 10)
    k25 = max(1, len(mse) // 4)
    print(f'\nчастка СУМИ від найгірших {k10} мереж: {o[-k10:].sum()/o.sum()*100:.1f}%')
    print(f'частка СУМИ від найгірших {k25} мереж: {o[-k25:].sum()/o.sum()*100:.1f}%')
    print(f'якби найгірші {k25} стали як медіана, бал став би '
          f'{(o[:-k25].sum() + k25*np.median(mse))/len(mse):.4e} '
          f'({np.mean(mse)/((o[:-k25].sum() + k25*np.median(mse))/len(mse)):.2f}×)')
    print(f'\nкореляції (по мережах):')
    print(f'  MSE ↔ ‖a_true‖²      r = {np.corrcoef(mse, nrm**2)[0,1]:+.3f}')
    print(f'  ВІДНОСНА похибка ↔ ‖a_true‖ r = {np.corrcoef(rel, nrm)[0,1]:+.3f}')
    print(f'  відносна похибка: сер {rel.mean()*100:.4f}% · розмах '
          f'{rel.max()/rel.min():.2f}×')
    print('\nЧИТАННЯ: якщо MSE↔‖a‖² майже 1.0, а відносна похибка стала —')
    print('хвіст це МАСШТАБ задачі, і лікувати треба не хвіст, а відносну похибку.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 32)
