"""ЧИ КОНВЕРТУЄТЬСЯ AUC 0.755 У ПРИЗ? — доставна маска замість оракульної.

`elembeta.py`: оракульна маска `r<1` дає **1.863×** проти 1.062× рівної ваги.
`rpredict.py`: обчислювані ознаки дають **AUC 0.755** усередині шару на 1..8
(⛔НЕГ 0.499), 0.62 глибше. Юзер: «класифікатор з AUC 0.75 дасть більшу частину
призу». Тут це перевіряється НАПРЯМУ.

ДВА ПРОХОДИ (інакше курка-яйце: маска потребує фіту, фіт — усіх мереж):
  1. ознаки + ІСТИННЕ `r` на кожній мережі (без жеребів, лише замикання+істина);
     рідж-фіт `log r` LOO ПО МЕРЕЖАХ ⟹ передбачений бал для ВІДКЛАДЕНОЇ;
  2. прогони з варіантами ваг: рівно · передбачена маска на квантилях ·
     ⚠️оракульна маска · ⚠️оракульна на тій самій частці · ⛔перемішана.

КОНТРОЛІ
  * ⚠️«оракул q» — оракульна маска, що лишає ТУ САМУ частку нейронів, що й
    передбачена: чесне порівняння «правильний відбір проти нашого»;
  * ⛔НЕГ: передбачений бал, ПЕРЕМІШАНИЙ у шарі (`roll`) — мусить дати рівень
    «рівно»;
  * 📊 три зрізи; `λ_ℓ` фітується LOO ПО МЕРЕЖАХ У КОЖНОМУ варіанті.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts, run_side1, LAM_S  # noqa: E402
from tilt_wide import truth_last                                       # noqa: E402
from price31 import truth_layers31, loo, three_slices                  # noqa: E402
from rpredict import features                                          # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
KS = 30
QGRID = [0.7, 0.5, 0.3]
BANDS = {'1..8': list(range(0, 8)), '24..30': list(range(23, 30)),
         '1..30': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def main(nets=24, dev=DEV):
    torch.manual_seed(606)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    # ---------- ПРОХІД 1: ознаки + істинне r ----------
    P1, keys = [], None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            F, _ = features(wc, dev, npair)
            keys = list(F[0])
            X = [torch.stack([F[i][k] for k in keys], 1).cpu().numpy()
                 for i in range(KS)]
            R = [float_(m_cl[i], TL['m'][i], F[i]) for i in range(KS)]
            P1.append((X, R))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  прохід 1: {n+1}/{nets}', flush=True)

    # ---------- фіт LOO по мережах ----------
    SCORE = []
    for h in range(nets):
        Xf = np.concatenate([np.concatenate(P1[q][0]) for q in range(nets)
                             if q != h])
        yf = np.concatenate([np.concatenate(P1[q][1]) for q in range(nets)
                             if q != h])
        mu, sd = Xf.mean(0), Xf.std(0) + 1e-12
        Z = np.hstack([(Xf - mu) / sd, np.ones((len(Xf), 1))])
        beta = np.linalg.solve(Z.T @ Z + 1e-3 * len(Z) * np.eye(Z.shape[1]),
                               Z.T @ yf)
        SCORE.append([np.hstack([(x - mu) / sd, np.ones((len(x), 1))]) @ beta
                      for x in P1[h][0]])

    # ---------- ПРОХІД 2: виграш ----------
    names = (['рівно'] + [f'ПЕРЕДБАЧЕНА q={q}' for q in QGRID]
             + ['⚠️оракул r<1'] + [f'⚠️оракул q={q}' for q in QGRID]
             + ['⛔НЕГ (перемішана)'])
    base_mse, G, B, C = [], None, None, []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            AW = []
            for i in range(KS):
                sc = torch.tensor(SCORE[n][i], dtype=torch.float64, device=dev)
                tr = torch.tensor(P1[n][1][i], dtype=torch.float64, device=dev)
                rows = [torch.ones_like(sc)]
                for q in QGRID:
                    thr = torch.quantile(sc, q)
                    rows.append((sc <= thr).double())
                rows.append((tr < 0.0).double())         # ⚠️оракул r<1
                for q in QGRID:
                    rows.append((tr <= torch.quantile(tr, q)).double())
                rows.append(torch.roll((sc <= torch.quantile(sc, 0.5)).double(), 7))
                AW.append(torch.stack(rows))
            nv = AW[0].shape[0]
            if G is None:
                G = [[] for _ in range(nv)]
                B = [[] for _ in range(nv)]
            gg = torch.zeros(nv, KS, KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, KS, dtype=torch.float64, device=dev)
            cc = 0.0
            for d in range(NDRAW):
                sd = 21_000_000 + 733 * n + d
                pr, A = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM31, collect_anchors=True,
                                  anchor_w=AW)
                e = pr - T
                cc += float((e * e).sum())
                Am = A.view(nv, KS, -1)
                gg += Am @ Am.transpose(1, 2)
                bb += Am @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc); base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  прохід 2: {n+1}/{nets}', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів, `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ '
          f'варіанті\n')
    for bn, idx in BANDS.items():
        print(f'--- шари {bn} ---')
        print(f'{"варіант":>22s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = '  ← У САБМІТІ' if nm == 'рівно' else ''
            tag = '  ⛔ ≈ рівень «рівно»' if nm.startswith('⛔') else tag
            print(f'{nm:>22s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    print('🔑 читати так: «ПЕРЕДБАЧЕНА» проти «рівно» — це ЧИСТИЙ доставний '
          'виграш (жодної істини). Проти «⚠️оракул q» на тій самій частці — '
          'скільки з нього ще лишилось у якості класифікатора.')


def float_(mcl, mtr, F):
    noise = F['log шум'].exp()
    return ((mcl - mtr).abs() / noise).clamp_min(1e-12).log().cpu().numpy()


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
