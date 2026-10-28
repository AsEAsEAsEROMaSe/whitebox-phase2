"""КАНАЛ ІНДИКАТОРА ВОРІТ `q̂ − Φ(t)`: ЄДИНА ЖИВА ФОРМА «БЕЗКОШТОВНОЇ» ІДЕЇ (С80).

ЗВІДКИ. [[finding-last-layer-sigma-reopens-the-class]] §12 лишив позначку:
  «🟢 Єдина жива форма цієї ідеї: канал `q̂ − Φ(t)` (індикатор як КЗ з ВІДОМИМ
   середнім, ворота вже рахуються, НУЛЬ флопів) — оцінка 0.5%, НЕ ВЗЯТО.»
Питання юзера про дрижання семплів привело рівно сюди: це єдина величина в
сім'ї «що каже окремий семпл», у якої середнє відоме АНАЛІТИЧНО, а не оцінене
з тієї самої вибірки — тобто єдина, що не ламається об теорему тавтології.

🔑 ЧОМУ ЦЕ БЕЗКОШТОВНО. `base_real.run_side1` УЖЕ рахує `gate` на кожному шарі
(рядок `gate = (p[:GATE_N] > 0).sum(0)/GATE_N`) — він потрібен для переносу `v`.
А еталон `Φ(t)` УЖЕ є в `closure_parts(extras=True)` як `f_o`. Тобто обидві
половини каналу лежать у прогоні й викидаються.
💰 Ціна каналу — лише сам перенос, `O(ksig·n²)` поруч із наявними `S` і `A`.

## ЩО МІРЯЄМО (три регресії, LOO ПО МЕРЕЖАХ, крос-ціль)
  база      `[S_cl; A_cl]`            — що маємо (σ-канал + μ-якір)
  +вільний  `[S_cl; A_cl; A_q(Φ)]`    — 🟢 ДОСЯЖНЕ: еталон із ЗАМИКАННЯ
  +оракул   `[S_cl; A_cl; A_q(істина)]` — стеля каналу: еталон з великого МК

⚠️ ЧОГО ЧЕКАТИ: [[finding-two-state-decomposition-is-shrinkage-of-the-product]]
   зміряв, що «ЧИ пропускає нейрон» несе лише **3.4%** дисперсії виходу
   ⟹ стеля каналу ~**1.031×**. Тобто це не важіль, а перевірка, чи є там
   хоч ті 0.5%, які колись оцінили «на око».

⛔ ПЛОМБИ:
 * база мусить відтворити відоме ≈1.00–1.06× (той самий набір, що в `mdepth`);
 * ⛔НЕГ: канал зі ЗСУНУТОГО жеребу мусить дати ≈базу;
 * оракул мусить бути ≥ вільного (інакше десь тавтологія).

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: вільний канал дасть **1.000–1.005×**, оракульний
   1.01–1.03×. Причина: `Φ(t)` із замикання само по собі зміщене (те саме
   зміщення, що вбило глибокий якір), тож «відоме середнє» тут відоме лише
   приблизно — а КЗ із НЕТОЧНО відомим середнім вносить зміщення.
   🔑 Якщо вільний дасть >1.01× — це безкоштовні флопи в бал.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qchan.py [NETS]
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

L_, W_ = 32, 256
DEV = 'cuda:3'
NDRAW = 64
KS = 31


def truth_ind(wc, dev, seed, npair=524_288, chunk=65536):
    """ІСТИННА частка активних `E[1(p>0)]` шарів 1..KS (для оракульної стелі)."""
    ns = [w.shape[1] for w in wc]
    acc = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = x @ w
            acc[i] += (p > 0).double().sum(0)
            x = torch.relu(p)
        done += k
    n = 2.0 * done
    return [acc[i] / n for i in range(1, KS + 1)]


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

    VAR = ['база (σ-канал + μ-якір)', '🟢 +канал q̂−Φ(t)  ВІЛЬНИЙ',
           '⚠️ +канал q̂−q_іст  ОРАКУЛ', '⛔НЕГ (канал зі зсунутого жеребу)']
    G = [[] for _ in VAR]
    B = [[] for _ in VAR]
    C, base_mse = [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, f_cl, _ = closure_parts(wc, KS, dev, extras=True)
            q_tru = truth_ind(wc, dev, 830_000 + 7919 * n)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            nreg = [2 * KS, 3 * KS, 3 * KS, 3 * KS]
            gg = [torch.zeros(k, k, dtype=torch.float64, device=dev)
                  for k in nreg]
            bb = [torch.zeros(k, dtype=torch.float64, device=dev) for k in nreg]
            cc, bm, prevQ = 0.0, 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     collect_sigma=True)
                _, A_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                    lam_s=None, collect_anchors=True)
                _, Qc = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                  collect_anchors=True,
                                  anchor_multi=(['ind'], [f_cl]))
                _, Qt = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                  collect_anchors=True,
                                  anchor_multi=(['ind'], [q_tru]))
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                RB = torch.cat([S_cl, A_cl], 0)
                for v, R in ((0, RB), (1, torch.cat([RB, Qc], 0)),
                             (2, torch.cat([RB, Qt], 0))):
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                if prevQ is not None:                   # ⛔НЕГ: зсунутий жереб
                    RN = torch.cat([RB, prevQ], 0)
                    gg[3] += RN @ RN.T
                    bb[3] += RN @ e
                prevQ = Qc
            for v in range(len(VAR)):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc); base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, LOO ПО МЕРЕЖАХ.\n')
    print(f'{"варіант":34s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 66)
    out = []
    for v, nm in enumerate(VAR):
        k = G[v][0].shape[0]
        val = loo(G[v], B[v], C, k)
        s, md, w3 = three_slices(base, val)
        out.append(s)
        print(f'{nm:34s} {s:8.4f}× {md:8.4f}× {w3:9.4f}×')
    print(f'\n🔑 ПРИРІСТ ВІЛЬНОГО КАНАЛУ понад базу: '
          f'×{out[1]/out[0]:.4f}   (оракул ×{out[2]/out[0]:.4f})')
    print(f'📊 БАЛ 1.954e-07 ⟹ вільний {1.954e-07/(out[1]/out[0]):.4e}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
