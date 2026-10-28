"""ЧИ ОЖИВЛЯЄ ВАРІАЦІЙНЕ ЗАМИКАННЯ ГЛИБИНУ КАНАЛУ (`KSIG` 8 → 24)?

════════ ЗВІДКИ ════════
`KSIG` 8→24 колись ПРОВАЛИВСЬ (1.032→0.973), і `deepanchor.py` пояснив чому:
виграш каналу на шарі = `1 + E[e²]/B²`, а глибше `B²` (похибка замикання)
росте швидше за шум вибірки `E[e²]` ⟹ глибокі канали забруднені зміщенням.
`variational.py`: вужчий гаус (`cp → α²·cp`, `α≈0.9975`) ріже `B` у **1.68×**
саме на глибині (на мілких лише 1.49×) ⟹ `B²` падає у **2.8×** ⟹ відношення
`E[e²]/B²` там же росте у 2.8×. Це рівно та причина, через яку глибина
не працювала. Отже питання прямо: чи вона тепер працює.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
На шарі 30 відношення було 0.043 ⟹ стеля шару 1.043×; після /2.8 стає
0.121 ⟹ стеля 1.121×. Глибокі шари несуть малу частку, тож
📣 ОГОЛОШУЮ: варіаційне замикання ЗСУНЕ оптимум `KSIG` глибше (8 → 12–16) і
   дасть **1.01–1.03×** понад базове замикання, але НЕ подолає поріг сабміту.
🟢 ЖИВИЙ, якщо чистий (з ціною глибшого замикання) >1.05×.

🚨 ПЛОМБИ
  ✅ `α=1` мусить відтворити відомий профіль `deepk` (насичення/спад);
  ✅ `KSIG=8, α=1` — це база, рівно 1.000×;
  ⛔ НЕГ-контроль: регресори з СУСІДНЬОГО жереба (шум тієї ж норми) ≈1.000×.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, run_side1, LAM_S             # noqa: E402
from varsub import closure_var                                     # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402
from price31 import loo, three_slices                              # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:2'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
KMAX = 24
KGRID = [8, 12, 16, 20, 24]
ALPHAS = (1.0, 0.9975, 0.9950)
PRICE = {8: 1.000, 12: 1.009, 16: 1.019, 20: 1.028, 24: 1.037}


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: оптимум `KSIG` зсунеться 8→12–16, приріст '
          f'1.01–1.03× понад базове замикання, поріг 1.05× НЕ подолає')
    print(f'   мереж {nets} × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    base_mse = []
    G = {a: [] for a in ALPHAS}
    Bv = {a: [] for a in ALPHAS}
    C = {a: [] for a in ALPHAS}
    Gn, Bn, Cn = [], [], []
    nreg = 2 * KMAX

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            CL = {a: closure_var(wc, KMAX, dev, a, s_raw=False)
                  for a in ALPHAS}
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            bm = 0.0
            gg = {a: torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
                  for a in ALPHAS}
            bb = {a: torch.zeros(nreg, dtype=torch.float64, device=dev)
                  for a in ALPHAS}
            cc = {a: 0.0 for a in ALPHAS}
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(ndraw):
                sd = 77_000_000 + 733 * n + d
                s0, m0 = CL[1.0][0], CL[1.0][1]
                prb, _ = run_side1(wc, npair, s0[:8], m0[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                for a in ALPHAS:
                    s_cl, m_cl = CL[a][0], CL[a][1]
                    p0, S = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                      collect_sigma=True)
                    _, A = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     lam_s=None, collect_anchors=True)
                    e1, e2 = p0 - T1, p0 - T2
                    e = 0.5 * (e1 + e2)
                    cc[a] += float((e1 * e2).sum())
                    R = torch.cat([S, A], 0)
                    gg[a] += R @ R.T
                    bb[a] += R @ e
                    if a == 1.0:
                        if prevR is not None:
                            gN += prevR @ prevR.T
                            bN += prevR @ e
                            cN += float((e1 * e2).sum())
                        prevR = R
            for a in ALPHAS:
                G[a].append(gg[a]); Bv[a].append(bb[a]); C[a].append(cc[a])
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {ndraw} жеребів. КРОС-ЦІЛЬ. База = САБМІТ '
          f'(KSIG=8, α=1).\n')
    print(f'{"α":>8s} {"KSIG":>6s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s} {"ціна":>7s} {"ЧИСТИЙ":>9s}')
    print('-' * 64)
    best = (0.0, None, None)
    for a in ALPHAS:
        for K in KGRID:
            idx = list(range(K)) + list(range(KMAX, KMAX + K))
            val = loo([g[idx][:, idx] for g in G[a]],
                      [b[idx] for b in Bv[a]], C[a], len(idx))
            s, md, w3 = three_slices(base, val)
            net = s / PRICE[K]
            if net > best[0]:
                best = (net, K, a)
            mark = ' ✅база' if (a == 1.0 and K == 8) else ''
            print(f'{a:8.4f} {K:6d} {s:8.3f}× {md:8.3f}× {w3:9.3f}× '
                  f'{PRICE[K]:7.3f} {net:8.3f}×{mark}')
        print('-' * 64)

    idx = list(range(16)) + list(range(KMAX, KMAX + 16))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'{"⛔НЕГ":>15s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×  '
          f'(мусить бути ≈1.000×)')

    a1 = max(KGRID, key=lambda K: three_slices(base, loo(
        [g[list(range(K)) + list(range(KMAX, KMAX + K))][:,
         list(range(K)) + list(range(KMAX, KMAX + K))] for g in G[1.0]],
        [b[list(range(K)) + list(range(KMAX, KMAX + K))] for b in Bv[1.0]],
        C[1.0], 2 * K))[0] / PRICE[K])
    print(f'\n🎯 НАЙКРАЩИЙ ЧИСТИЙ: {best[0]:.3f}× при KSIG={best[1]}, '
          f'α={best[2]:.4f}  (при α=1 оптимум KSIG={a1})')
    print(f'🔑 ВИРОК: {"ЖИВИЙ" if best[0] > 1.05 else "МЕРТВИЙ"} '
          f'(🟢поріг сабміту 1.05×)')
    print(f'📊 БАЛ: 2.05e-07 → {2.05e-07 / max(best[0], 1e-9):.3e}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
