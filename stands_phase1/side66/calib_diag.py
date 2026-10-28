"""РОЗСУДЖУЄМО ДВА СТЕНДИ: ЧИ КАЛІБРУВАННЯ ПОКРАЩУЄ s НА ТЕСТОВИХ МЕРЕЖАХ.

СУПЕРЕЧНІСТЬ. `feat_calib.py`: регресія на (t,t²), навчена на мережах 0..4,
знімає R²=0.82 на мережах 5..9 і зменшує r удвічі. `final_gain.py`: та сама
регресія, застосована до мереж 8..15, ПОГІРШУЄ бал (0.099 без ваг проти 0.750
без калібрування). Обидва не можуть бути праві.

Перша гіпотеза (вибух `s_c/(1+pred)` при pred≈−1) СПРОСТОВАНА: кламп дільника
не змінив жодної цифри результату.

ЦЕЙ СТЕНД міряє прямо, без жеребів і без переносу:
    |s_c − s_іст|/s_іст   похибка ЗАМИКАННЯ
    |s_k − s_іст|/s_іст   похибка КАЛІБРОВАНОГО замикання
на ТИХ САМИХ мережах 8..15, що й у `final_gain.py`, з коефіцієнтами, навченими
на 0..4. Якщо друге більше — винне калібрування; якщо менше — винен спосіб
його застосування у виробничому стенді.

Друкуємо і СЕРЕДНЄ, і МАКСИМУМ по нейронах: середнє може покращитись, а один
викид — отруїти шар (перенос через 31 шар підсилює саме викиди).
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments                         # noqa: E402
from closure_precision import NQ                              # noqa: E402
from mu_gain import closure_ms                                # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:2'
TRUTH_N = 1_048_576
TRAIN, SKIP, NETS, UPTO = 5, 8, 6, 31
SHOW = (1, 2, 4, 8, 16, 24, 31)


def main(dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    allf = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have]

    def load(f):
        s = int(np.load(f)['seed'])
        return [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
                for q in range(L_)]

    def collect(files, seed0):
        out = {i: {'t': [], 'y': []} for i in range(1, UPTO + 1)}
        for k, f in enumerate(files):
            with torch.no_grad():
                wc, _, _ = cut_net(load(f), dev)
                _, _, sg_t = truth_moments(wc, dev, seed0 + k, TRUTH_N)
                clo = closure_ms(wc, nodes, wts, UPTO)
                for i in range(1, UPTO + 1):
                    m_c, s_c = clo[i]
                    st = sg_t[i - 1].double()
                    out[i]['t'].append((m_c / s_c).cpu().numpy())
                    out[i]['y'].append(((s_c - st) / st).cpu().numpy())
            print(f'    {k+1}/{len(files)}', flush=True)
        return out

    print('навчальні:', flush=True)
    TR = collect(allf[:TRAIN], 90001)
    print('тестові:', flush=True)
    TE = collect(allf[SKIP:SKIP + NETS], 70001)

    print(f'\nнавчання 0..{TRAIN-1}, тест {SKIP}..{SKIP+NETS-1}, істина {TRUTH_N}')
    print(f'{"шар":>4s} {"|зам|сер":>10s} {"|калібр|сер":>12s} {"виграш":>8s}'
          f' {"|зам|макс":>10s} {"|калібр|макс":>13s} {"макс×":>7s}')
    print('-' * 74)
    for i in SHOW:
        t = np.concatenate(TR[i]['t']); y = np.concatenate(TR[i]['y'])
        A = np.stack([np.ones_like(t), t, t * t], 1)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        tt = np.concatenate(TE[i]['t']); yy = np.concatenate(TE[i]['y'])
        B = np.stack([np.ones_like(tt), tt, tt * tt], 1)
        pred = B @ w
        # похибка каліброваного s: s_k = s_c/(1+pred) ⟹ (s_k − st)/st
        rk = (1.0 + yy) / (1.0 + pred) - 1.0
        print(f'{i:4d} {np.abs(yy).mean():10.3e} {np.abs(rk).mean():12.3e}'
              f' {np.abs(yy).mean()/np.abs(rk).mean():8.2f}'
              f' {np.abs(yy).max():10.3e} {np.abs(rk).max():13.3e}'
              f' {np.abs(yy).max()/np.abs(rk).max():7.2f}')
    print('\nЧИТАННЯ: «виграш»>1 ⟹ калібрування покращує в середньому.')
    print('«макс×»<1 ⟹ воно РОЗДУВАЄ найгірший нейрон — саме він і псує перенос.')


if __name__ == '__main__':
    main()
