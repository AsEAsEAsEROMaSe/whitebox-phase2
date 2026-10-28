"""ВИРІШАЛЬНА ПЕРЕВІРКА: чи дають нові канали щось ПОВЕРХ ТОЧНОГО САБМІТУ.

ЗВІДКИ. `combo.py`: `σ8+μ8+â²8+p⁴8` = **1.052×** за нуль додаткових флопів.
🚨 АЛЕ це з ВІЛЬНИМ `λ` на 32 коефіцієнтах, а сам `σ8+μ8` з вільним `λ` дає
1.032× ⟹ дві третини могли бути ПЕРЕФІТОМ.

🚨🚨 ПРАВИЛО, ЩО НАС УЖЕ ОБПЕКЛО ([[finding-tfit-free-t-feature-gives-1.056x]]
§🚨🔴): «дармовий» виграш із ВІЛЬНОЮ вагою буває НУЛЕМ при ФІКСОВАНІЙ вазі
сабміту — бо зашиті `λ` налаштовані на грейдері під ЗАБРУДНЕНИЙ якір, і вони
працюють У ПАРІ. Кандидата міряти ЛИШЕ проти бази з ЗАШИТИМИ коефіцієнтами.

ЩО РОБИМО. База = `run_side1(lam_s=LAM_S)` — ТОЧНИЙ сабміт, нічого не чіпаємо.
Поверх додаємо ЛИШЕ нові канали: `pred(λ) = pred_саб − λᵀ[A_â²; A_p⁴]`,
`λ` LOO ПО МЕРЕЖАХ (16 коефіцієнтів). Це рівно те, що можна перенести в сабміт
константами.

ОЧІКУВАННЯ: якщо канали несуть НЕЗАЛЕЖНУ інформацію — дадуть 1.02–1.04×;
якщо вся цінність була в перефіті — 1.00×.
ПЕЧАТКИ: `λ=0` мусить дати РІВНО 1.000× · ⛔ШУМ тієї самої норми ≈1.000 ·
⛔НЕГ ≈1.0.
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
K = 8


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NAMES = ['✅λ=0 (пломба 1.000×)', '🎯â² 8 вільних λ',
             '🎯â² лише шари 1..4', '🎯â² ЛИШЕ шар 1 (ціль ТОЧНА)',
             '⛔ШУМ тієї самої норми']
    IDX = [[], list(range(0, K)), list(range(0, 4)), [0],
           list(range(0, K))]
    nv = len(NAMES)
    base_mse, C = [], []
    G = [[] for _ in range(nv)]
    B = [[] for _ in range(nv)]
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, _f, q_cl = closure_parts(wc, K, dev, extras=True)
            p4_ref = [3.0 * s ** 4 for s in s_cl]
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            nreg = 2 * K
            gg = torch.zeros(nv, nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            g = torch.Generator(device=dev); g.manual_seed(1717 + n)

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                # 🔑 БАЗА — ТОЧНИЙ САБМІТ, зашиті LAM_S, нічого не чіпаємо
                pr, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM_S)
                _, AM = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                  collect_anchors=True,
                                  anchor_multi=(['a2', 'p4'],
                                                [q_cl, p4_ref]))
                e1, e2 = pr - T1, pr - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                nz = torch.randn(nreg, AM.shape[1], generator=g, device=dev,
                                 dtype=torch.float64)
                nz = nz * (AM.norm() / nz.norm().clamp_min(1e-30))
                for v in range(nv):
                    R = nz if v == nv - 1 else AM
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = AM
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. '
          f'База = ТОЧНИЙ САБМІТ (зашиті LAM_S).\n')
    print(f'{"варіант":>32s} {"регр":>5s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 70)
    for v, nm in enumerate(NAMES):
        idx = IDX[v]
        if not idx:
            print(f'{nm:>32s} {0:5d} {1.0:8.3f}× {1.0:8.3f}× {1.0:9.3f}×')
            continue
        val = loo([g[idx][:, idx] for g in G[v]],
                  [b[idx] for b in B[v]], C, len(idx))
        s, md, w3 = three_slices(base, val)
        print(f'{nm:>32s} {len(idx):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    # 🔑 КОЕФІЦІЄНТИ ДЛЯ ЗАШИВАННЯ: фіт на ВСІХ мережах + розкид LOO
    ii = list(range(0, K))
    Gt = sum(g[ii][:, ii] for g in G[1])
    bt = sum(b[ii] for b in B[1])
    eye = torch.eye(K, dtype=torch.float64, device=Gt.device)
    dg = torch.diagonal(Gt).clamp_min(1e-300).sqrt()
    lam_all = torch.linalg.solve(Gt / torch.outer(dg, dg) + 1e-8 * eye,
                                 bt / dg) / dg
    per = []
    for q in range(len(G[1])):
        A2 = Gt - G[1][q][ii][:, ii]
        v2 = bt - B[1][q][ii]
        d2 = torch.diagonal(A2).clamp_min(1e-300).sqrt()
        per.append((torch.linalg.solve(A2 / torch.outer(d2, d2) + 1e-8 * eye,
                                       v2 / d2) / d2).cpu().numpy())
    per = np.array(per)
    print('\n🔑 λ для каналу `â²` (фіт на ВСІХ мережах):')
    print('   ' + ', '.join(f'{v:.5f}' for v in lam_all.cpu().numpy()))
    print('   LOO-розкид (std/|сер|): '
          + ', '.join(f'{v:.2f}' for v in
                      (per.std(0) / np.abs(per.mean(0)).clip(1e-12))))
    idx = list(range(2 * K))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ (зсув жеребу)":>32s} {len(idx):5d} {s:8.3f}× '
          f'{md:8.3f}× {w3:9.3f}×')
    print('\n🎯 ЦЕ ЧИСЛО — РЕАЛЬНИЙ КАНДИДАТ НА САБМІТ (нуль додаткових флопів, '
          'база НЕ чіпається, додаються лише константи).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
