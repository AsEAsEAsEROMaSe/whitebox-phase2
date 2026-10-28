"""ГЛИБИНА КАНАЛУ ЗА ПОЧТИ НУЛЬ: `KSIG` 8 → 24.

МЕТА (юзер): точність із того самого ядра, не з бюджету.

ЗВІДКИ. Сабміт тримає `KSIG=8` — σ-канал і μ-якір бачать ЛИШЕ 8 шарів із 31.
Причина історична (8 зашитих `LAM_S`). Але ЦІНА глибшого замикання МАЛА:
`4n³` на крок, `n=256` ⟹ 6.7e7/крок.
| `KSIG` | кроків | флопів замикання | приріст до `B=272e9` | множник | ціна |
|---|---|---|---|---|---|
| 8 | 9 | 6.0e8 | — | 0.1073 | 1.000 |
| 12 | 13 | 8.7e8 | +0.10% | 0.1083 | 1.009 |
| 16 | 17 | 1.14e9 | +0.20% | 0.1093 | 1.019 |
| 20 | 21 | 1.41e9 | +0.30% | 0.1103 | 1.028 |
| 24 | 25 | 1.68e9 | +0.40% | 0.1113 | 1.037 |
🚨 σ-гілка на глибших шарах має зміряний потенціал 2.83× (`d_ℓ` 1..24 → 6.55×),
а `mdepth.py` дав «мілкі 1..16 = 2.409×» з ІДЕАЛЬНИМ `m`. Питання: скільки дає
глибина з РЕАЛЬНИМ (замикальним) `m` і `s`.

ОЧІКУВАННЯ ДО ПРОГОНУ: `b` РОСТЕ з глибиною ⟹ глибокі канали забруднені, і
приріст має НАСИТИТИСЬ. Ставлю на оптимум `KSIG ≈ 12–16` з чистим **1.03–1.08×**.
ПОРІГ: чистий >1.05× ⟹ кандидат на сабміт.

ПЕЧАТКИ: `KSIG=8` мусить відтворити 1.020× (`freecv.py`) · ⛔НЕГ ≈1.0 ·
🚨КРОС-ЦІЛЬ обов'язкова.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import loo, three_slices                             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KMAX = 24
KGRID = [8, 12, 16, 20, 24]
# ціна: (кроків(K) − кроків(8)) · 4n³ / B, перекладена в множник 0.1073
PRICE = {8: 1.000, 12: 1.009, 16: 1.019, 20: 1.028, 24: 1.037}


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    base_mse, C = [], []
    G, B = [], []
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KMAX, dev)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            bm = 0.0
            nreg = 2 * KMAX
            gg = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  collect_sigma=True)
                _, A = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                 collect_anchors=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                R = torch.cat([S, A], 0)
                gg += R @ R.T
                bb += R @ e
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = R
            G.append(gg); B.append(bb); C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. База = САБМІТ.\n')
    print(f'{"KSIG":>6s} {"регр":>5s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s} {"ціна":>7s} {"ЧИСТИЙ":>9s}')
    print('-' * 62)
    best = (0.0, None)
    for K in KGRID:
        idx = list(range(K)) + list(range(KMAX, KMAX + K))
        val = loo([g[idx][:, idx] for g in G], [b[idx] for b in B], C,
                  len(idx))
        s, md, w3 = three_slices(base, val)
        net = s / PRICE[K]
        if net > best[0]:
            best = (net, K)
        print(f'{K:6d} {len(idx):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}× '
              f'{PRICE[K]:7.3f} {net:8.3f}×')
    idx = list(range(16)) + list(range(KMAX, KMAX + 16))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ":>6s} {len(idx):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    print(f'\n🎯 НАЙКРАЩИЙ ЧИСТИЙ: {best[0]:.3f}× при KSIG={best[1]}. '
          f'ПОРІГ сабміта 1.05×.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
