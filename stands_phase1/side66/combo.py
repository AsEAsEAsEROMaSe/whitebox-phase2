"""ФІНАЛЬНИЙ ЗБІР: усе, що дає точність із того самого ядра.

ЩО ВЖЕ ВІДОМО ЦІЄЇ СЕСІЇ:
  * `freecv.py`: `q̂` і `â²` (безкоштовні статистики) = **+0.9%**;
  * `deepk.py`: глибина НЕ допомагає, оптимум `KSIG=8` (1.032×), глибше гірше;
  * конструкція БЕЗ нормування якоря краща (1.032 проти 1.020).

ЩО ПЕРЕВІРЯЄМО ТУТ:
  1. чи канали РОЗДІЛЯЮТЬСЯ по глибині — може, σ терпить глибину, а μ ні
     (`b` росте з глибиною і б'є саме по ЯКОРЮ, бо той міряє `x̄−m`);
  2. `p⁴` — четвертий момент ДОАКТИВАЦІЙ. Ціль `3s⁴` ТОЧНА на шарі 1
     (там `h` строго гаусів) і гаусове наближення глибше. Коштує `O(N·n)`;
  3. усе разом.

ОЧІКУВАННЯ: σ глибше за μ витримає (σ міряє `ŝ−s`, а зміщення `s` росте
повільніше за зміщення `m` — з `perfect.py`: ідеальне `s` 2.26× проти `m` 10.5×
⟹ `s` менш забруднене). Ставлю на (σ=16, μ=8) як найкращу пару, **1.04–1.07×**.
ПОРІГ сабміта: 1.05×.
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
KM = 16                       # максимум шарів у наборі
PRICE16 = 1.019               # ціна замикання на 17 кроків замість 9


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    # рядки: σ 0..15 | μ 16..31 | a2 32..47 | p4 48..63
    def blk(o, k):
        return list(range(o * KM, o * KM + k))
    VAR = [('⛔σ8+μ8 (найкраще з deepk)', blk(0, 8) + blk(1, 8), 1.000),
           ('🎯σ16+μ8', blk(0, 16) + blk(1, 8), PRICE16),
           ('🎯σ8+μ8+â²8', blk(0, 8) + blk(1, 8) + blk(2, 8), 1.000),
           ('🎯σ8+μ8+p⁴8', blk(0, 8) + blk(1, 8) + blk(3, 8), 1.000),
           ('🎯σ8+μ8+â²8+p⁴8', blk(0, 8) + blk(1, 8) + blk(2, 8) + blk(3, 8),
            1.000),
           ('🎯σ16+μ8+â²8+p⁴8',
            blk(0, 16) + blk(1, 8) + blk(2, 8) + blk(3, 8), PRICE16)]
    nv = len(VAR)
    base_mse, C = [], []
    G, B = [], []
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KM, dev)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            # ЦІЛІ: `a2` = E[relu²] із замикання; `p4` = 3s⁴ (гаусове)
            _s, _m, _f, q_cl = closure_parts(wc, KM, dev, extras=True)
            p4_ref = [3.0 * s ** 4 for s in s_cl]

            bm = 0.0
            nreg = 4 * KM
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
                _, AM = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                  collect_anchors=True,
                                  anchor_multi=(['a2', 'p4'],
                                                [q_cl, p4_ref]))
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                R = torch.cat([S, A, AM], 0)
                assert R.shape[0] == nreg, R.shape
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
    np.savez('/mnt/win_d/llm/White-Box/side66/logs/combo.npz',
             base=base, C=np.array(C),
             G=torch.stack(G).cpu().numpy(), B=torch.stack(B).cpu().numpy())
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. База = САБМІТ.\n')
    print(f'{"варіант":>30s} {"регр":>5s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s} {"ціна":>7s} {"ЧИСТИЙ":>9s}')
    print('-' * 84)
    best = (0.0, '')
    for nm, idx, pr in VAR:
        val = loo([g[idx][:, idx] for g in G], [b[idx] for b in B], C,
                  len(idx))
        s, md, w3 = three_slices(base, val)
        net = s / pr
        if net > best[0]:
            best = (net, nm)
        print(f'{nm:>30s} {len(idx):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}× '
              f'{pr:7.3f} {net:8.3f}×')
    idx = VAR[0][1]
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ":>30s} {len(idx):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    print(f'\n🎯 НАЙКРАЩИЙ: {best[0]:.3f}× — {best[1]}. ПОРІГ сабміта 1.05×.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
