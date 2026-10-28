"""ЧИ 12× — ЦЕ СТЕЛЯ ІНФОРМАЦІЇ, ЧИ СТЕЛЯ ФІТУ? (спроба дійти до 1e-8, крок 2)

ЗВІДКИ. `perfect.py`: ідеальні `m` і `s` на 30 шарах = **12.13/13.47/11.18×**
проти сабміту; ціль 1e-8 потребує **20.6×**. Але в тому стенді якір шару — це
**256 чисел, стиснутих в ОДИН скаляр `λ_ℓ`**. Тобто 7680 чисел інформації
проходять крізь 30-вимірне горло. Перш ніж казати «клас не дотягує», треба
спитати, чи не горло винне.

ЩО РОБИМО. Даємо фітові БАГАТШИЙ базис: якір шару несеться не з вагою `1`, а з
кількома ОБЧИСЛЮВАНИМИ поелементними вагами (жодної істини у вагах):
  `1` · `v_i` (адьойнт-перенос) · `σ_i` (шум із замикання) · `φ(t_i)` · `t_i`.
Кожна дає свій `λ`, тобто 5 скалярів на шар замість одного. Якщо стеля
підіймається — горло було у фіті, і треба шукати кращу вагу. Якщо ні — 12×
це справді межа ІНФОРМАЦІЇ перших двох моментів, і 1e-8 усередині класу немає.

🚨 ПАРАМЕТРІВ БАГАТО: при K=30 і 5 вагах це 180 коефіцієнтів на 24 мережах.
Урок сьогоднішнього дня («прогін мусить мати багато більше мереж, ніж
параметрів») ⟹ головні рядки — K=8 (48 параметрів) і K=16 (96), а K=30 подано
як ⚠️довідковий. LOO ПО МЕРЕЖАХ карає перефіт, тож зростання на малих K
достовірне.

КОНТРОЛІ
  * ✅рядок `лише вага 1` мусить відтворити `perfect.py` (10.46× при K=30);
  * ⛔НЕГ: ваги, ПЕРЕМІШАНІ всередині шару (зсув) — мусить дати рівень ваги `1`;
  * 📊 три зрізи; база = СПРАВЖНІЙ САБМІТ.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, _phi)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402
from vrank import state_and_v                                     # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KS = 30
WN = ['1', 'v', 'σ', 'φ(t)', 't']
SETS = [([0], '✅лише вага 1 (звірка з perfect.py)'),
        ([0, 1], '+ v (адьойнт)'),
        ([0, 2], '+ σ'),
        ([0, 3], '+ φ(t)'),
        ([0, 1, 2], '+ v і σ'),
        ([0, 1, 2, 3, 4], '🎯УСІ П\'ЯТЬ ВАГ')]
DEPTHS = [8, 16, 24, 30]


def main(nets=24, dev=DEV):
    torch.manual_seed(4321)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NW = len(WN)
    nv = len(SETS) + 1                       # +1 = ⛔НЕГ
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
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            vs, nz = state_and_v(wc, dev, npair)
            mtr = [TL['m'][i] for i in range(KS)]

            # ваги — ЛИШЕ обчислювані, нормовані на середнє (обумовленість)
            AW = []
            for i in range(KS):
                t = m_cl[i] / s_cl[i].clamp_min(1e-30)
                cand = [torch.ones_like(t), vs[i + 1], nz[i + 1], _phi(t), t]
                rows = []
                for c in cand:
                    sc = float(c.abs().mean().clamp_min(1e-30))
                    rows.append(c / sc)
                rows += [torch.roll(r, 7) for r in rows[1:]]   # ⛔НЕГ-ваги
                AW.append(torch.stack(rows))
            NR = AW[0].shape[0]

            bm, cc = 0.0, 0.0
            GG = torch.zeros((NR + 1) * KS, (NR + 1) * KS,
                             dtype=torch.float64, device=dev)
            BB = torch.zeros((NR + 1) * KS, dtype=torch.float64, device=dev)
            for d in range(NDRAW):
                sd = 99_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T) ** 2).sum())
                p0, S = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  collect_sigma=True)
                _, A = run_side1(wc, npair, s_cl, mtr, dev, sd, lam_s=None,
                                 collect_anchors=True, anchor_w=AW)
                R = torch.cat([S, A], 0)
                e = p0 - T
                cc += float((e * e).sum())
                GG += R @ R.T
                BB += R @ e
            for v in range(nv):
                G[v].append(GG); B[v].append(BB)
            C.append(cc); base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. Якір = ІДЕАЛЬНЕ `m`; ваги — лише '
          f'ОБЧИСЛЮВАНІ. База = СПРАВЖНІЙ САБМІТ.\n')
    print(f'{"базис ваг":>36s} {"пар.":>5s} {"K":>4s} {"сума":>9s} '
          f'{"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 82)
    allsets = SETS + [([0] + list(range(NW, NW + NW - 1)),
                       '⛔НЕГ (ваги перемішані в шарі)')]
    for v, (ws, nm) in enumerate(allsets):
        for K in DEPTHS:
            idx = ([q * KS + i for q in range(1) for i in range(K)]
                   + [(1 + q) * KS + i for q in ws for i in range(K)])
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            mark = '  ⚠️' if len(idx) > 3 * nets else ''
            print(f'{nm if K == DEPTHS[0] else "":>36s} {len(idx):5d} {K:4d} '
                  f'{s:8.3f}× {md:8.3f}× {w3:9.3f}×{mark}')
        print()
    print('🔑 читати так: якщо «УСІ ПʼЯТЬ ВАГ» на K=8 і K=16 помітно вище за '
          '«лише вага 1» — горло було у ФІТІ, і стелю класу треба переміряти '
          'із кращою вагою. Якщо ні — 12× це межа ІНФОРМАЦІЇ, і 1e-8 усередині '
          'класу не існує. ⚠️ позначає рядки, де параметрів забагато для 24 '
          'мереж — їх читати лише як довідку.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
