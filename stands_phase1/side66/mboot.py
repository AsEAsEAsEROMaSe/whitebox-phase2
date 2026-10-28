"""ІДЕЯ ЮЗЕРА: КУПИТИ НЕЗМІЩЕНІСТЬ ЯКОРЯ ЗА ЧАСТИНУ БЮДЖЕТУ.

ЗВІДКИ. Якір `a = x̄_ℓ − m_замик` мертвий на глибині, бо `m_замик` ЗМІЩЕНЕ
(`b₃₀`). Ідеальне `m₃₀` дає **13.435×** (`mdepth.py`). Замінити `m_замик` на
`m̂` з НЕЗАЛЕЖНИХ точок ⟹ зміщення зникає ЗА ПОБУДОВОЮ. Ціна: точки, прогнані
лише до шару 30, коштують 30/32 = 0.94 повного прогону.

🔑 ПЕРЕФОРМУЛЮВАННЯ ЮЗЕРА, ЯКЕ ЦЕ ВІДКРИЛО: не «інтерполювати між точками»
(⛔`celllin.py`), а «взяти нову точку ДЕШЕВШЕ» — ціна точки ∝ ГЛИБИНІ.

ОЧІКУВАННЯ ДО ПРОГОНУ (порахувати ДО, інакше підженемо пояснення):
`ρ²_еф = 0.9256·M/(M+N)`, ціна `= 1 + 0.94·M/N` ⟹ чистий виграш
0.99 / 0.96 / 0.86 для `M/N` = 0.3 / 1 / 3 — тобто СКРІЗЬ ≤1, і це рівно
закон бюджетної нейтральності (`MSE×бюджет=const`, підтверджено josu).
🚨 Модель ГРУБА (припускає лінійне ділення `ρ²`) ⟹ міряємо.
ПОРІГ: якщо хоч один рядок дає ЧИСТИЙ >1.05× — закон має виняток, і це важливо
саме по собі.

ПЕЧАТКИ: `M→∞` (істина) мусить дати 13.435× · `m_замик` мусить дати 1.007×.
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
from price31 import truth_layers31, loo, three_slices             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KS = 31
SIG31 = list(range(0, 31))
MU30 = list(range(KS, KS + 30))
IDX = SIG31 + MU30
RATIOS = [0.5, 1.0, 2.0]


def m_hat(wc, npt, seed, dev, chunk=4096):
    """`m̂` шарів 1..31 з `npt` НЕЗАЛЕЖНИХ точок (антитетичні пари)."""
    ns = [w.shape[1] for w in wc]
    a1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npt:
        k = min(chunk, npt - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            x = torch.relu(x @ w)
            a1[i] += x.double().sum(0)
        done += k
    return [a1[i] / (2.0 * done) for i in range(1, KS + 1)]


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NAMES = (['⛔m_замик (пломба 1.007×)']
             + [f'🎯m̂ з M/N={r}' for r in RATIOS]
             + ['✅m_іст (пломба 13.435×)'])
    nv = len(NAMES)
    base_mse, C = [], []
    G = [[] for _ in range(nv)]
    B = [[] for _ in range(nv)]

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            mv = [m_cl]
            for r in RATIOS:
                mv.append(m_hat(wc, max(1, int(r * npair)),
                                330_000 + 7919 * n, dev))
            mv.append(TA['m'][:KS])

            bm = 0.0
            nreg = 2 * KS
            gg = torch.zeros(nv, nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     collect_sigma=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                for v in range(nv):
                    _, A = run_side1(wc, npair, s_cl, mv[v], dev, sd,
                                     lam_s=None, collect_anchors=True)
                    R = torch.cat([S_cl, A], 0)
                    gg[v] += R @ R.T
                    bb[v] += R @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. База = САБМІТ.\n')
    print(f'{"варіант":>30s} {"виграш":>9s} {"МЕДІАНА":>9s} {"ціна":>7s} '
          f'{"ЧИСТИЙ":>8s}')
    print('-' * 68)
    for v, nm in enumerate(NAMES):
        val = loo([g[IDX][:, IDX] for g in G[v]],
                  [b[IDX] for b in B[v]], C, len(IDX))
        s, md, _w3 = three_slices(base, val)
        r = RATIOS[v - 1] if 1 <= v <= len(RATIOS) else 0.0
        # ціна: множник `0.1073` росте на частку доданих флопів (30/32 шарів);
        # підлога 0.1 вже пройдена, тож ціна лінійна
        price = 1.0 + 0.94 * r * (0.0942 / 0.1073)
        print(f'{nm:>30s} {s:8.3f}× {md:8.3f}× {price:7.3f} '
              f'{s/price:7.3f}×')
    print('\n🎯 ПОРІГ: ЧИСТИЙ >1.05× хоч в одному рядку ⟹ закон бюджетної '
          'нейтральності має виняток.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
