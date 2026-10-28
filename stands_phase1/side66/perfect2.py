"""12× — ЦЕ СТЕЛЯ ЗАМИКАННЯ ЧИ ЛИШЕ СТЕЛЯ ПЕРШИХ ДВОХ МОМЕНТІВ?

ЗВІДКИ. `perfect.py` дав стелю **12.1/13.5/11.2×** за ідеальних `m` і `s` на 30
шарах (✅репліка на роздільній істині, ⛔НЕГ 0.93). Ціль 1e-8 вимагає **20.6×**
⟹ клас не дотягує. Але «ідеальні `m` і `s`» — це ідеальні ПЕРШІ ДВА МОМЕНТИ
активацій. Справжнє замикання знало б РОЗПОДІЛ, а не два числа: частку активних,
`E[a²]`, треті й четверті моменти доактивацій. Кожна з цих статистик — окремий
якір із ТОЧНО ВІДОМИМ середнім, тобто окрема контрольна змінна.

🔑 ЧОМУ ЦЕ НЕ «ЩЕ ОДНА ІДЕЯ», А ЗАКРИТТЯ ПИТАННЯ. Якщо додавання вищих моментів
підіймає стелю над 20.6× — шлях до 1e-8 існує всередині замикання, і треба
рахувати вищі моменти. Якщо стеля лишається ~12× — **клас закрито остаточно**,
і 1e-8 треба шукати поза ним. Обидві відповіді змінюють план.

ЯКОРІ (усі з ІСТИННИМИ значеннями — це СТЕЛЯ, не метод)
  `a` = E[a_ℓ] · `ind` = частка активних · `a2` = E[a²] · `p3`,`p4` = 3-й і 4-й
  моменти доактивацій. Кожен ділиться на `s^порядок` заради обумовленості.

КОНТРОЛІ
  * ✅база = СПРАВЖНІЙ САБМІТ; ✅рядок `лише a` мусить відтворити 10.5× з
    `perfect.py` (незалежний шлях через `anchor_multi`);
  * ⛔НЕГ: регресори жеребу `d−1`, помилка жеребу `d`;
  * 📊 три зрізи; глибини 8/16/24/30.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, W_)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import loo, three_slices                             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KS = 30
KINDS = ['a', 'ind', 'a2', 'p3', 'p4']
SETS = [(['a'], '✅лише a (звірка з perfect.py ≈10.5×)'),
        (['a', 'ind'], '+ частка активних'),
        (['a', 'a2'], '+ E[a²]'),
        (['a', 'ind', 'a2'], '+ обидві другого порядку'),
        (['a', 'ind', 'a2', 'p3'], '+ 3-й момент доактивацій'),
        (['a', 'ind', 'a2', 'p3', 'p4'], '🎯УСІ П\'ЯТЬ — СТЕЛЯ РОЗПОДІЛУ')]
DEPTHS = [8, 16, 24, 30]


def truth_stats(wc, dev, seed, npair=524_288, chunk=65536):
    """ІСТИННІ пошарові статистики: E[a], частка активних, E[a²], E[p³], E[p⁴]."""
    ns = [w.shape[1] for w in wc]
    acc = {k: [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
           for k in KINDS}
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = x @ w
            pd = p.double()
            x = torch.relu(p)
            xd = x.double()
            acc['a'][i] += xd.sum(0)
            acc['ind'][i] += (pd > 0).double().sum(0)
            acc['a2'][i] += xd.pow(2).sum(0)
            acc['p3'][i] += pd.pow(3).sum(0)
            acc['p4'][i] += pd.pow(4).sum(0)
        done += k
    n = 2.0 * done
    return {k: [acc[k][i] / n for i in range(1, KS + 1)] for k in KINDS}


def main(nets=24, dev=DEV):
    torch.manual_seed(2024)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    nv = len(SETS)
    base_mse, C = [], []
    G = [[] for _ in range(nv)]
    B = [[] for _ in range(nv)]
    Gn, Bn, Cn = [], [], []
    NK = len(KINDS) + 1        # +1 = σ-канал (блок 0)

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TS = truth_stats(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            refs = [TS[k] for k in KINDS]

            bm, cc = 0.0, 0.0
            gg = [torch.zeros(NK * KS, NK * KS, dtype=torch.float64, device=dev)
                  for _ in range(1)]
            GG = torch.zeros(NK * KS, NK * KS, dtype=torch.float64, device=dev)
            BB = torch.zeros(NK * KS, dtype=torch.float64, device=dev)
            gN = torch.zeros(NK * KS, NK * KS, dtype=torch.float64, device=dev)
            bN = torch.zeros(NK * KS, dtype=torch.float64, device=dev)
            cN, prevA = 0.0, None
            del gg

            for d in range(NDRAW):
                sd = 88_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T) ** 2).sum())
                p0, S = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  collect_sigma=True)
                _, Aa = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                  collect_anchors=True,
                                  anchor_multi=(KINDS, refs))
                A = torch.cat([S, Aa], 0)      # σ-канал завжди, як у perfect.py
                e = p0 - T
                cc += float((e * e).sum())
                GG += A @ A.T
                BB += A @ e
                if prevA is not None:
                    gN += prevA @ prevA.T
                    bN += prevA @ e
                    cN += float((e * e).sum())
                prevA = A
            for v, (ks, _) in enumerate(SETS):
                G[v].append(GG); B[v].append(BB)
            C.append(cc); base_mse.append(bm)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. База = СПРАВЖНІЙ САБМІТ. '
          f'Усі якорі з ІСТИННИМИ значеннями ⟹ це СТЕЛЯ.\n')
    print(f'{"набір статистик":>38s} {"K":>4s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 76)
    for v, (ks, nm) in enumerate(SETS):
        pos = [0] + [KINDS.index(k) + 1 for k in ks]   # 0 = σ-канал
        for K in DEPTHS:
            idx = [q * KS + i for q in pos for i in range(K)]
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            print(f'{nm if K == DEPTHS[0] else "":>38s} {K:4d} '
                  f'{s:8.3f}× {md:8.3f}× {w3:9.3f}×')
        print()
    idx = list(range(NK * KS))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'{"⛔НЕГ (зсув жеребу)":>38s} {30:4d} {s:8.3f}× {md:8.3f}× '
          f'{w3:9.3f}×   ⛔ мусить упасти до ≈1.0')
    print('\n🔑 читати так: якщо «УСІ П\'ЯТЬ» помітно вище за «лише a», то 12× '
          'була стеля ДВОХ МОМЕНТІВ, і замикання, яке рахує розподіл, дає '
          'більше. Якщо ні — клас «краще замикання» закрито на 12×, і 1e-8 '
          'треба шукати ПОЗА ним.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
