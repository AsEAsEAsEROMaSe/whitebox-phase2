"""КАЛІБРУВАННЯ ЗАМИКАННЯ ФУНКЦІЄЮ ВІД ОЗНАК НЕЙРОНА. (мета 1e-8)

ЗВІДКИ. `calib.py`: похибка замикання СИСТЕМАТИЧНА (зсув у 4-11× більший за
розкид між мережами), але константа на шар її не описує — лишок після зняття
зсуву майже дорівнює самій похибці (шар 8: зсув 1.63e-2, лишок 1.77e-2).
Отже похибка має ПОНЕЙРОННУ структуру.

ІДЕЯ. Передбачити її з ознак, які рахуються БЕЗ вибірки (лише ваги + замикання)
⟹ незалежність від вибірки зберігається, а саме вона тут дефіцитна
(`onestep.py`: точніша оцінка, що торкнулась вибірки, дала ГІРШЕ ρ).

ГОЛОВНА ОЗНАКА — фізична. Похибка гаусового наближення для `p_j = Σ_k x_k W_kj`
визначається тим, наскільки працює CLT, а це — ЕФЕКТИВНА КІЛЬКІСТЬ ДОДАНКІВ
    n_eff = (Σ_k a_k)² / Σ_k a_k²,   a_k = W_kj²·C_kk
Мало доданків ⟹ сильна негаусовість ⟹ велика похибка замикання.
Плюс форма `t = m/s` (де саме сидить поріг relu) і глибина.

ЩО МІРЯЄМО. R² передбачення `rel = (s_зам − s_іст)/s_іст` на ВІДКЛАДЕНИХ
мережах, і головне — новий `r після` = лишок/шум_вибірки. Якщо він падає
нижче 1 глибше шару 4, δs стає доступною там, де зараз ρ≈0, а отже β
підскакують і конструкція бере більшу частку оракульної стелі.

КОНТРОЛІ
  * навчання на мережах 0..4, перевірка на 5..9 — РІЗНИХ;
  * SHUFFLE ознак ⟹ R² мусить впасти до ~0;
  * базова лінія «лише константа на шар» (це `calib.py`) — новий має її бити;
  * друкуємо і R², і те, що реально важить: r до / r після.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments                         # noqa: E402
from closure_precision import (relu_cov0, relu_cov_gauss,     # noqa: E402
                               sample_moments, NQ)

W_, L_ = 256, 32
DEV = 'cuda:2'
TRUTH_N = 4_194_304
SHOW = (1, 2, 3, 4, 6, 8, 12, 16, 24, 31)


def main(nets=10, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    X, Y, NET, LAY, NOISE = [], [], [], [], []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            _, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            ms, ss = sample_moments(wc, 12345, dev)

            w0 = wc[0].double()
            m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
            C_est = relu_cov0(wc[0])
            for i in range(1, L_):
                Wd = wc[i].double()
                C_p = Wd.T @ (C_est @ Wd)
                s_p = C_p.diagonal().clamp_min(1e-30).sqrt()
                m_p = m_est @ Wd
                st = sg_t[i - 1].double()

                a = (Wd * Wd) * C_est.diagonal().unsqueeze(1)     # внески у дисперсію
                n_eff = (a.sum(0) ** 2 / (a * a).sum(0).clamp_min(1e-300))
                t = m_p / s_p
                srel = s_p / s_p.median()
                feats = torch.stack([t, t * t, n_eff.log(),
                                     srel.log(), torch.full_like(t, i / 32.0)], 1)
                X.append(feats.cpu().numpy())
                Y.append(((s_p - st) / st).cpu().numpy())
                NET.append(np.full(t.numel(), k))
                LAY.append(np.full(t.numel(), i))
                NOISE.append((((ss[i - 1] - st) / st).abs()).cpu().numpy())
                C_est, m_est = relu_cov_gauss(m_p, C_p, nodes, wts)
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    X = np.concatenate(X); Y = np.concatenate(Y)
    NET = np.concatenate(NET); LAY = np.concatenate(LAY)
    NOISE = np.concatenate(NOISE)
    half = nets // 2
    tr, te = NET < half, NET >= half

    def fit_apply(cols, Xs):
        """Окрема регресія НА КОЖНОМУ шарі; коефіцієнти з навчальних мереж."""
        pred = np.zeros_like(Y)
        for i in range(1, L_):
            mtr = tr & (LAY == i)
            mte = te & (LAY == i)
            A = np.concatenate([Xs[mtr][:, cols], np.ones((mtr.sum(), 1))], 1)
            B = np.concatenate([Xs[mte][:, cols], np.ones((mte.sum(), 1))], 1)
            w, *_ = np.linalg.lstsq(A, Y[mtr], rcond=None)
            pred[mte] = B @ w
        return pred

    rs = np.random.default_rng(3)
    Xsh = X.copy()
    Xsh[:, :4] = X[rs.permutation(len(X))][:, :4]

    variants = (('лише константа (calib.py)', [], X),
                ('t, t²', [0, 1], X),
                ('t, t², log n_eff, log s ★', [0, 1, 2, 3], X),
                ('SHUFFLE ознак', [0, 1, 2, 3], Xsh))
    print(f'\n{nets} мереж (навчання 0..{half-1}, тест {half}..{nets-1})')
    print(f'{"ознаки":30s} {"R² тест":>9s}   ' + ' '.join(f'r@{i}' for i in SHOW))
    print('-' * 96)
    for nm, cols, Xs in variants:
        pred = fit_apply(cols, Xs)
        r2 = 1.0 - ((Y[te] - pred[te]) ** 2).sum() / ((Y[te] - Y[te].mean()) ** 2).sum()
        rr = []
        for i in SHOW:
            m = te & (LAY == i)
            rr.append(np.abs(Y[m] - pred[m]).mean() / NOISE[m].mean())
        print(f'{nm:30s} {r2:+9.4f}   ' + ' '.join(f'{v:4.2f}' for v in rr))
    print('\nЧИТАННЯ: r — це похибка_замикання/шум_вибірки ПІСЛЯ калібрування.')
    print('β = 1/(1+r²); менший r ⟹ більша частка оракульної стелі 2.6-2.9×.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
