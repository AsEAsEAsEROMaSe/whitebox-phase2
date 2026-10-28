"""СТЕЛЯ КЛАСУ «КРАЩЕ ЗАМИКАННЯ» + ДРАБИНА ВІДДАЧІ — чи існує шлях до 1e-8.

ЗВІДКИ. Ціль 1e-8 = **20×** від 2.06e-07. На підлозі множника 0.1 треба
`MSE = 1e-7`, тобто **19×** від нинішніх 1.91e-06. Вісь `n` вичерпана свіпом на
грейдері ⟹ усе мусить прийти з оцінювача при тому самому бюджеті.

🚨 ЧОГО МИ НЕ МІРЯЛИ ЖОДНОГО РАЗУ. `price31.py` спрайсив ідеальне `s` (2.41×),
С67 — оракульне `d_ℓ` (2.83×/6.55×). **Ідеальне `m` не прайсили НІКОЛИ**, хоча
саме його зміщення породжує всю усадку `β`, навколо якої крутилися три дні.
Механізм, якщо він є, уже виміряний нами сьогодні збоку:
`corr(перенесений шум шару ℓ, шум шару ℓ+1) = +0.943`
⟹ якщо `m_ℓ` відоме точно, то `x̄_ℓ − m_ℓ` — це ПРЯМИЙ вимір вибіркової похибки
ЦЬОГО жеребу, і вона переноситься до виходу майже без втрат.

ЩО РАХУЄМО. Ланцюг лінійний за обома каналами, і при `lam_s=None` база `pred₀`
не залежить НІ від `s`, НІ від `m` ⟹ один прогін дає базу, два — матриці `S`
(σ-канал, залежить лише від `s`), решта — `A` (μ-канал, залежить лише від `m`).
`pred(λ) = pred₀ − λᵀ[S; A]`, 2·K коефіцієнтів LOO ПО МЕРЕЖАХ, зрізи по глибині
задарма.

🔑 ГОЛОВНЕ — НЕ КІНЦЕВА ТОЧКА, А ДРАБИНА: `m_k = m_іст + k·(m_замик − m_іст)`.
Вона перетворює «стеля 20×» на «щоб дістати X×, треба зменшити зміщення
замикання у 1/k разів» — тобто на технічне завдання.

КОНТРОЛІ (стеля такої величини вимагає максимальної недовіри)
  * ✅база = СПРАВЖНІЙ САБМІТ (`KSIG=8`, фіксовані `LAM_S`);
  * ✅**РЕПЛІКА**: ідеальне `m` із ДРУГОЇ, роздільної істини (сід 910k) — мусить
    дати те саме; інакше працює шум істини, а не сигнал;
  * ⛔НЕГ: регресори жеребу `d−1`, помилка жеребу `d` — рве зв'язок ОДНОГО
    жеребу; при справжньому механізмі мусить упасти до ≈1.0;
  * 📊 три зрізи; глибини K=8/16/30 (менше параметрів — менше перефіту).
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
KS = 30
KGRID = [1.0, 0.5, 0.25, 0.1, 0.0]
DEPTHS = [4, 8, 12, 16, 20, 24, 26, 28, 29, 30]


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = ([f'm: k={k}' + ('  ← ЗАМИКАННЯ' if k == 1.0 else
                             ('  🎯ІДЕАЛЬНЕ m' if k == 0.0 else ''))
              for k in KGRID]
             + ['✅РЕПЛІКА: ідеальне m, істина B',
                '⚠️ідеальне s (контроль ≈2.4×)',
                '🎯⚠️ІДЕАЛЬНІ m і s — СТЕЛЯ КЛАСУ'])
    nv = len(names)
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
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            TB = truth_layers31(wc, dev, 910_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            mgrid = [[TA['m'][i] + k * (m_cl[i] - TA['m'][i])
                      for i in range(KS)] for k in KGRID]
            mgrid.append([TB['m'][i] for i in range(KS)])   # ✅ РЕПЛІКА
            s_tr = TA['s'][:KS]

            bm = 0.0
            gg = torch.zeros(nv, 2 * KS, 2 * KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, 2 * KS, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(2 * KS, 2 * KS, dtype=torch.float64, device=dev)
            bN = torch.zeros(2 * KS, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                # ✅ БАЗА — справжній сабміт: KSIG=8, фіксовані LAM_S
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T) ** 2).sum())
                # 🔑 при lam_s=None база НЕ залежить ні від `s`, ні від `m`
                p0, Scl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                    collect_sigma=True)
                _, Str = run_side1(wc, npair, s_tr, m_cl, dev, sd,
                                   collect_sigma=True)
                As = [run_side1(wc, npair, s_cl, mv, dev, sd, lam_s=None,
                                collect_anchors=True)[1] for mv in mgrid]
                e = p0 - T
                cc += float((e * e).sum())
                rows = ([torch.cat([Scl, A], 0) for A in As]
                        + [torch.cat([Str, As[0]], 0),
                           torch.cat([Str, As[-2]], 0)])
                for v, R in enumerate(rows):
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                RI = rows[KGRID.index(0.0)]          # ⛔НЕГ на ІДЕАЛЬНОМУ m
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e * e).sum())
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
    np.savez('/mnt/win_d/llm/White-Box/side66/logs/perfect24.npz',
             base=base, C=np.array(C),
             G=np.stack([torch.stack(g).cpu().numpy() for g in G]),
             B=np.stack([torch.stack(b).cpu().numpy() for b in B]),
             names=np.array(names))
    print(f'\n{nets} мереж × {NDRAW} жеребів. База = СПРАВЖНІЙ САБМІТ '
          f'(KSIG=8, фіксовані LAM_S). Обидва канали, `λ` LOO ПО МЕРЕЖАХ.\n')
    print(f'{"варіант":>34s} {"K":>4s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 72)
    for v, nm in enumerate(names):
        for K in DEPTHS:
            idx = list(range(K)) + list(range(KS, KS + K))
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            print(f'{nm if K == DEPTHS[0] else "":>34s} {K:4d} '
                  f'{s:8.3f}× {md:8.3f}× {w3:9.3f}×')
        print()
    idx = list(range(2 * KS))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, len(idx))
    s, md, w3 = three_slices(base, vN)
    print(f'{"⛔НЕГ (ідеальне m, зсув жеребу)":>34s} {30:4d} {s:8.3f}× '
          f'{md:8.3f}× {w3:9.3f}×   ⛔ мусить упасти до ≈1.0')
    print('\n🔑 читати так: «ідеальні m і s, K=30» — СТЕЛЯ ВСЬОГО класу '
          '«краще замикання» (ієрархічні матриці, модифіковані рівняння, '
          'Тротер-Йосіда, Мойяль, Крилов, Еджворт, седлова точка, EP — усі '
          'виробляють РІВНО це). Драбина `k` каже, у скільки разів треба '
          'зменшити зміщення замикання заради даного виграшу. ✅РЕПЛІКА мусить '
          'збігтися з `k=0`, ⛔НЕГ — упасти до 1.0.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
