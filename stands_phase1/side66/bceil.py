"""СТЕЛЯ ОЗНАК (не моделі): скільки `b₃₀` взагалі передбачувано ЛОКАЛЬНО.

ЗВІДКИ (`b30.py`): лінійний фіт на 8 ознаках дав `R²_oos=0.316` і виграш 1.023×.
Драбина вимагає зняти **94%** енергії для 1.7×. Перш ніж вкладатися в банк
(2454 мережі) і нелінійні моделі, треба знати, чи справа В МОДЕЛІ чи В ОЗНАКАХ.

ЩО РАХУЄМО. Непараметрична стеля: розбити нейрони на комірки по ознаках і
взяти СЕРЕДНЄ `b/s` у комірці як передбачення. Це стеля БУДЬ-ЯКОЇ моделі на
цьому наборі ознак (умовне сподівання — найкращий передбачувач у L²).
🚨 Стеля міряється ЧЕСНО: середнє комірки — з ІНШИХ мереж (LOO по мережах),
інакше комірка підглядає у власну відповідь і стеля роздувається до 1.0.

ОЧІКУВАННЯ ДО ПРОГОНУ: стеля **0.65–0.75** ⟹ локальні ознаки ВИЧЕРПАНІ,
потрібна ІСТОРІЯ мережі (рекурсія `b`), а не властивості нейрона.
ПОРІГ: якщо стеля <0.85 — банк на ЛОКАЛЬНИХ ознаках не врятує (для 1.7×
треба 0.94), і копати треба рекурсію джерел, а не більшу модель.

КОНТРОЛІ: ⛔одна комірка (=константа) — має відтворити ≈0.30–0.47 з С70;
⛔мітки мереж перемішані зсувом — стеля має впасти до ≈0.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, _phi, _Phi)  # noqa: E402
from price31 import truth_layers31                                 # noqa: E402

L_ = 32
DEV = 'cuda:3'
KS = 31


def cell_r2(key, y, net, nets, nb):
    """`R²` умовного середнього по комірках `key`, LOO ПО МЕРЕЖАХ."""
    tot = torch.zeros(nb, dtype=torch.float64, device=y.device)
    cnt = torch.zeros(nb, dtype=torch.float64, device=y.device)
    tot.index_add_(0, key, y)
    cnt.index_add_(0, key, torch.ones_like(y))
    ss, st = 0.0, 0.0
    for n in range(nets):
        msk = net == n
        to = torch.zeros(nb, dtype=torch.float64, device=y.device)
        cn = torch.zeros(nb, dtype=torch.float64, device=y.device)
        to.index_add_(0, key[msk], y[msk])
        cn.index_add_(0, key[msk], torch.ones_like(y[msk]))
        mu = (tot - to) / (cnt - cn).clamp_min(1.0)
        pr = mu[key[msk]]
        ss += float(((y[msk] - pr) ** 2).sum())
        st += float(((y[msk] - y.mean()) ** 2).sum())
    return 1.0 - ss / max(st, 1e-300)


def main(nets=48, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    T_, S_, Y_, N_ = [], [], [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            b30 = m_cl[29] - TA['m'][29]
            T_.append(m_cl[29] / s_cl[29])
            S_.append(s_cl[29] / s_cl[29].mean())
            Y_.append(b30 / s_cl[29])
            N_.append(torch.full_like(b30, float(n)))
        del W, wc
        torch.cuda.empty_cache()
        if (n + 1) % 8 == 0:
            print(f'  … {n+1}/{nets}', flush=True)

    t = torch.cat(T_); s = torch.cat(S_); y = torch.cat(Y_)
    net = torch.cat(N_).long()
    print(f'\nнейронів усього: {y.numel()}, мереж {nets}\n')

    def rank_bins(x, nb):
        """Бінінг ПО РАНГУ (рівнозаповнені комірки) — стійкіше за рівномірний."""
        r = torch.argsort(torch.argsort(x))
        return (r.double() * nb / x.numel()).long().clamp(0, nb - 1)

    print(f'{"ознаки / комірок":>34s} {"R² (стеля)":>12s}')
    print('-' * 50)
    print(f'{"⛔одна комірка (=константа)":>34s} '
          f'{cell_r2(torch.zeros_like(net), y, net, nets, 1):12.4f}')
    for nb in (8, 32, 128):
        k = rank_bins(t, nb)
        print(f'{f"лише t, {nb} комірок":>34s} '
              f'{cell_r2(k, y, net, nets, nb):12.4f}')
    for nb in (8, 16, 32):
        k = rank_bins(t, nb) * nb + rank_bins(s, nb)
        print(f'{f"t × s, {nb}×{nb} комірок":>34s} '
              f'{cell_r2(k, y, net, nets, nb * nb):12.4f}')
    # ⛔ КОНТРОЛЬ: мітки мереж зсунуто ⟹ комірка передбачає ЧУЖУ мережу
    ysh = torch.cat([Y_[(n + 1) % nets] for n in range(nets)])
    if ysh.numel() == y.numel():
        k = rank_bins(t, 16) * 16 + rank_bins(s, 16)
        print(f'\n{"⛔ЗСУВ мереж (мусить впасти)":>34s} '
              f'{cell_r2(k, ysh, net, nets, 256):12.4f}')
    print('\n🎯 ПОРІГ: <0.85 ⟹ ЛОКАЛЬНІ ознаки вичерпані, банк їх не врятує; '
          'копати РЕКУРСІЮ ДЖЕРЕЛ, а не більшу модель.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 48)
