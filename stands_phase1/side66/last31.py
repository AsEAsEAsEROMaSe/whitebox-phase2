"""ЧИ Є ШЛЯХ ДО 1e-8: ЩО ЛИШИЛОСЬ НА ОСТАННЬОМУ ШАРІ.

ЗВІДКИ. `perfect.py` дав стелю класу «краще замикання» = 12.1× при ІДЕАЛЬНИХ
`m` і `s` на шарах 1..30, тобто `ρ²=0.918`. Лишається 8.2% дисперсії виходу,
які НЕ пояснюються жодною пошаровою статистикою шарів 1..30 — і саме вони
відділяють 12× від потрібних 20.6×.

🚨 ЩО ПРОҐАВИЛИ. Мережа має 32 шари (індекси 0..31), ціль — `E[relu(h₃₁)]`.
`perfect.py` брав `KS=30` ⟹ σ-канал і μ-якір бачили шари 1..30, а **шар 31
не бачив НІХТО**. Тим часом за гаусової гіпотези вихід визначається рівно
двома числами: середнім доактивацій шару 31 (воно точне автоматично, коли
`m₃₀` точне — перенос `m₃₀@W₃₁` ЛІНІЙНИЙ) і `s₃₁` = std доактивацій шару 31.
⟹ `s₃₁` — ОСТАННЯ відсутня деталь, і вона НЕ тавтологія.
(Тавтологія — це `m₃₁`: якір `x̄₃₁ − m_іст,₃₁` при λ=1 дає відповідь. Вмикаю
його ОКРЕМО, як пломбу, що канал шару 31 узагалі працює.)

ОЧІКУВАННЯ ДО ПРОГОНУ (щоб не підганяти пояснення під число):
  * ідеальне `s` 1..30            → 2.26×   (пломба, репліка perfect.py)
  * ідеальні `m,s` 1..30          → 12.1×   (пломба, репліка perfect.py)
  * 🎯 `m` 1..30 + `s` 1..31      → 13×…35× ← РОЗВИЛКА
  * ⚠️ `m` 1..31 (тавтологія)      → ≫50×    (пломба)
ПОРІГ РІШЕННЯ: >20× ⟹ шлях до 1e-8 існує ВСЕРЕДИНІ класу і впирається в один
шар. <15× ⟹ залишок 9.6% — це негаусовість останнього шару, клас закритий.

КОНСТРУКЦІЯ. Ланцюг лінійний за обома каналами, `pred(λ) = pred₀ − λᵀ[S; A]`,
`λ` LOO ПО МЕРЕЖАХ. Матриця rows має 62 рядки: 0..30 — σ-канал шарів 1..31,
31..61 — μ-якорі шарів 1..31. Варіанти — це ЗРІЗИ індексів по тих самих
матрицях, тож «1..30» відтворює perfect.py ТОЧНО (регресора шару 31 просто
нема, а не «є, але із замиканням»).

ПЕЧАТКИ
  * ✅база = СПРАВЖНІЙ САБМІТ (KSIG=8, фіксовані LAM_S);
  * ✅«усе замикання» мусить сидіти на ≈1.0–1.06× (вільний λ на 31 шарі проти
    фіксованого на 8 — приросту майже нема, це вже знали);
  * ✅дві пломби-репліки perfect.py (2.26× і 12.1×);
  * ✅РЕПЛІКА головного варіанта на ДРУГІЙ, роздільній істині (сід 910k);
  * ⚠️ТАВТОЛОГІЯ мусить вибухнути — інакше індексація шару 31 хибна;
  * ⛔НЕГ: регресори жеребу `d−1` проти помилки жеребу `d` — мусить упасти ≈1.0.
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
from price31 import truth_layers31, loo, three_slices             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KS = 31                      # 🚨 31, а не 30: шар 31 нарешті в наборі

# rows: 0..30 = σ шарів 1..31, 31..61 = μ шарів 1..31
SIG30 = list(range(0, 30))
SIG31 = list(range(0, 31))
MU30 = list(range(KS, KS + 30))
MU31 = list(range(KS, KS + 31))


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    # (назва, індекс пари rows, індекси регресорів)
    #   пара 0 = [S_замик ; A_замик]   пара 1 = [S_іст ; A_замик]
    #   пара 2 = [S_іст ; A_іст]       пара 3 = [S_істB ; A_істB]
    VAR = [
        ('✅усе замикання (≈1.0–1.06×)', 0, SIG30 + MU30),
        ('⚠️ідеальне s 1..30 (пломба 2.26×)', 1, SIG30 + MU30),
        ('🆕ідеальне s 1..31', 1, SIG31 + MU30),
        ('⚠️ідеальні m,s 1..30 (пломба 12.1×)', 2, SIG30 + MU30),
        ('🎯🆕m 1..30 + s 1..31', 2, SIG31 + MU30),
        ('✅РЕПЛІКА 🎯 (істина B)', 3, SIG31 + MU30),
        ('⚠️ТАВТОЛОГІЯ: m 1..31', 2, SIG31 + MU31),
    ]
    nv = len(VAR)
    NPAIR_ROWS = 4
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
            s_cl, m_cl = closure_parts(wc, KS, dev)
            assert len(s_cl) == KS and len(m_cl) == KS, (len(s_cl), len(m_cl))
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            TB = truth_layers31(wc, dev, 910_000 + 7919 * n)
            # 🚨 ДВІ НЕЗАЛЕЖНІ ЦІЛІ — КРОС-ОЦІНКА MSE. Дим показав, що з ОДНІЄЮ
            # ціллю навіть ТАВТОЛОГІЯ дає лише 22.7×: це не фізика, а стеля
            # ПРИЛАДУ (шум істини 2.3% від MSE ⟹ 1/0.023 ≈ 43×). Оцінка
            # `⟨(pred−T₁)(pred−T₂)⟩` незміщена для MSE і НЕ має стелі, бо шуми
            # `T₁`,`T₂` незалежні й у сподіванні їхній внесок зникає.
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            s_A, m_A = TA['s'][:KS], TA['m'][:KS]
            s_B, m_B = TB['s'][:KS], TB['m'][:KS]
            assert len(s_A) == KS, len(s_A)

            bm = 0.0
            nreg = 2 * KS
            gg = torch.zeros(nv, nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                # ✅ БАЗА — справжній сабміт
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                # 🔑 при lam_s=None база НЕ залежить ні від `s`, ні від `m`
                p0, S_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     collect_sigma=True)
                _, S_A = run_side1(wc, npair, s_A, m_cl, dev, sd,
                                   collect_sigma=True)
                _, S_B = run_side1(wc, npair, s_B, m_cl, dev, sd,
                                   collect_sigma=True)
                _, A_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                    lam_s=None, collect_anchors=True)
                _, A_A = run_side1(wc, npair, s_cl, m_A, dev, sd,
                                   lam_s=None, collect_anchors=True)
                _, A_B = run_side1(wc, npair, s_cl, m_B, dev, sd,
                                   lam_s=None, collect_anchors=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)          # у `loo`: b = R@e, c = Σe₁e₂
                cc += float((e1 * e2).sum())
                pairs = [torch.cat([S_cl, A_cl], 0),
                         torch.cat([S_A, A_cl], 0),
                         torch.cat([S_A, A_A], 0),
                         torch.cat([S_B, A_B], 0)]
                assert len(pairs) == NPAIR_ROWS
                assert pairs[0].shape[0] == nreg, pairs[0].shape
                for v, (_, pi, _idx) in enumerate(VAR):
                    R = pairs[pi]
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                RI = pairs[2]                     # ⛔НЕГ на ІДЕАЛЬНИХ m,s
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = RI
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    np.savez('/mnt/win_d/llm/White-Box/side66/logs/last31.npz',
             base=base, C=np.array(C),
             G=np.stack([torch.stack(g).cpu().numpy() for g in G]),
             B=np.stack([torch.stack(b).cpu().numpy() for b in B]),
             names=np.array([v[0] for v in VAR]))
    print(f'\n{nets} мереж × {NDRAW} жеребів. База = СПРАВЖНІЙ САБМІТ. '
          f'`λ` LOO ПО МЕРЕЖАХ, KS={KS}.\n')
    print(f'{"варіант":>38s} {"регр":>5s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 76)
    for v, (nm, _pi, idx) in enumerate(VAR):
        val = loo([g[idx][:, idx] for g in G[v]],
                  [b[idx] for b in B[v]], C, len(idx))
        s, md, w3 = three_slices(base, val)
        print(f'{nm:>38s} {len(idx):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    idx = SIG31 + MU30
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ (зсув жеребу)":>38s} {len(idx):5d} {s:8.3f}× '
          f'{md:8.3f}× {w3:9.3f}×   ⛔ мусить упасти до ≈1.0')
    print('\n🎯 ПОРІГ: «m 1..30 + s 1..31» > 20× ⟹ шлях до 1e-8 існує всередині '
          'класу і впирається в ОДИН шар. < 15× ⟹ залишок — негаусовість '
          'останнього шару, клас закритий.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
