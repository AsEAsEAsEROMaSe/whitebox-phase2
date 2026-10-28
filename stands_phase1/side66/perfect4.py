"""ДРУГИЙ ПОРЯДОК ПЕРЕНОСУ — залишок 9.6% після ідеального `m` (спроба 1e-8, крок 3).

ЗВІДКИ. `perfect.py`: ідеальне `m` на 30 шарах дає **12.06/12.90/11.31×** (лише
μ-канал; σ-канал із замикання додає НУЛЬ і лише коштує параметрів). Драбина
точності дала `ρ² = 0.904`, тобто **9.6% похибки виходу НЕ пояснюється першим
порядком** переносу пошарових похибок. Для 1e-8 треба `ρ² = 0.951`, тобто зняти
половину цього залишку.

🔑 ЧОМУ ДРУГИЙ ПОРЯДОК, І ЧОМУ ЙОГО ЩЕ НЕ МІРЯЛИ В ЦЬОМУ РЕЖИМІ.
С67 міряла «квадрат помилки» і дістала **1.001×** — але тоді якір ішов від
ЗАМИКАННЯ, тобто `δ` була переважно ЗМІЩЕННЯМ, а не вибірковою похибкою, і
перший порядок майже нічого не знімав. Тепер перший порядок знімає 90%, і
залишок за побудовою — вищий порядок. Це рівно наш метарецепт: **міряти той
ланцюг, який лагодиш**.

📐 ФОРМА ЧЛЕНА. `x̄_{ℓ+1} = mean(relu(a_ℓ^(s) W))`. Перший порядок —
`Φ(t)·(δ_ℓ W)`. Наступний іде від КРИВИНИ ReLU: друга похідна — дельта-функція,
згладжена гаусіаном доактивації ⟹
    `сек_{ℓ+1} = (δ_ℓ W_{ℓ+1})² · φ(t_{ℓ+1}) / (2 s_{ℓ+1})`,
і далі несеться тим самим ланцюгом воріт. Жодного нового прогону: `δ_ℓ` уже є
всередині (це той самий якір), `φ(t)/s` — з вибірки.

КОНТРОЛІ
  * ✅рядок `лише 1-й порядок` мусить відтворити 12.06× з `perfect.py`;
  * ⛔НЕГ: члени 2-го порядку з ПОПЕРЕДНЬОГО жеребу проти помилки цього;
  * 📊 три зрізи; база = СПРАВЖНІЙ САБМІТ; `λ` LOO ПО МЕРЕЖАХ, фіт
    стандартизований (інваріантний до масштабу рядків).
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
DEV = 'cuda:2'
NDRAW = 64
KS = 30
DEPTHS = [8, 16, 24, 29]


def main(nets=24, dev=DEV):
    torch.manual_seed(1234)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = ['✅лише 1-й порядок (звірка ≈12.1×)',
             '🎯+ 2-й порядок (кривина ReLU)',
             '⛔НЕГ (2-й порядок із чужого жеребу)']
    nv = len(names)
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
            mtr = [TL['m'][i] for i in range(KS)]

            bm, cc = 0.0, 0.0
            GG = torch.zeros(3 * KS, 3 * KS, dtype=torch.float64, device=dev)
            BB = torch.zeros(3 * KS, dtype=torch.float64, device=dev)
            prev2 = None
            for d in range(NDRAW):
                sd = 55_500_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T) ** 2).sum())
                p0, (A1, A2) = run_side1(wc, npair, s_cl, mtr, dev, sd,
                                         lam_s=None, collect_anchors=True,
                                         anchor_sec=True)
                e = p0 - T
                cc += float((e * e).sum())
                if prev2 is None:
                    prev2 = A2
                R = torch.cat([A1, A2, prev2], 0)
                prev2 = A2
                GG += R @ R.T
                BB += R @ e
            for v in range(nv):
                G[v].append(GG); B[v].append(BB)
            C.append(cc); base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. Якір = ІДЕАЛЬНЕ `m`. '
          f'База = СПРАВЖНІЙ САБМІТ.\n')
    print(f'{"варіант":>38s} {"K":>4s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 78)
    blocks = [[0], [0, 1], [0, 2]]
    for v, nm in enumerate(names):
        for K in DEPTHS:
            idx = [q * KS + i for q in blocks[v] for i in range(K)]
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            print(f'{nm if K == DEPTHS[0] else "":>38s} {K:4d} '
                  f'{s:8.3f}× {md:8.3f}× {w3:9.3f}×')
        print()
    print('🔑 читати так: якщо 2-й порядок помітно підіймає — залишок 9.6% має '
          'ФОРМУ, і клас можна розширити далі (3-й порядок, перехресні члени). '
          'Якщо ні — залишок безструктурний, стеля 12× остаточна, і 1e-8 '
          'усередині цієї сім’ї не існує.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
