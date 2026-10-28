"""ПЕРЕЗАПУСК ЗАМИКАННЯ З ВИБІРКИ: чи стає `m₂₉` точнішим?

ЗВІДКИ. Лідер = точна опора на шарі ≈29 з похибкою ~0.02%; наше замикання
там дає ~0.9%. Перебір «що краще корелює» приречений: щоб зрізати 0.9% до
0.02%, конструкція має пояснити **R²=0.9995** дисперсії зміщення, а це вже
не кореляція, а тотожність (найкращий зміряний R² = 0.792 лишає 0.41%).

ІДЕЯ, ЯКА НЕ Є КОРЕЛЯЦІЄЮ. Похибка замикання накопичується за 29 кроків
гаусового припущення. Візьмемо ЕМПІРИЧНІ `m̂_K, Ĉ_K` з наших же семплів і
пустимо рекурсію лише від `K+1` ⟹ аналітична похибка накопичується за
`29−K` кроків замість 29. Ціна — вибірковий шум у `Ĉ_K` (256×256 із ~8k
семплів, тобто шумна матриця).
🔑 ЩО ПЕРЕВАЖИТЬ — НЕ ОЧЕВИДНО, І САМЕ ЦЕ ВИРІШУЄ ВСЕ.

🚨 ЦЕ НЕ ВИМІР ВИГРАШУ В БАЛІ. Це вимір ТОЧНОСТІ `m₂₉` — тобто чи існує в
принципі спосіб дістати опору потрібної якості. Ціна рахується окремо.

════════ ЩО ДРУКУЄМО ════════
`e(K)` = ‖m̂₂₉(K) − m₂₉‖/‖m₂₉‖ у ВІДСОТКАХ, де K — шар перезапуску;
K=0 — чисте замикання (наша база, ~0.9%); K=29 — тавтологія (чиста вибірка).
Поруч — 📏ЦІЛЬ 0.02% і шум самої вибірки на шарі 29.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 крива буде U-подібною: спершу падає (менше кроків замикання), потім росте
   (шум `Ĉ_K` домінує), мінімум десь K=12..20 на рівні **0.3-0.6%**;
📣 до 0.02% НЕ дійде НІДЕ — не вистачить на порядок.
🟢 ЖИВЕ, якщо мінімум ≤0.10% (тоді опора класу K≈24 стає досяжною).
🚨 ПЛОМБИ
  ✅ K=0 мусить відтворити відому похибку замикання ~0.9%;
  ✅ K=29 (чиста вибірка) = шум вибірки, тобто ~0.2%;
  ✅ обидві межі рахуються ОДНИМ кодом ⟹ крива між ними не може брехати.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_                           # noqa: E402
from jitprof import truth_means                                    # noqa: E402
from jitanim import bank_net                                       # noqa: E402
from q31ab import budget                                           # noqa: E402
from q31dir import _phi, _Phi                                      # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
KS = (0, 2, 4, 8, 12, 16, 20, 24, 27, 29)
TGT = 29


def close_from(wc, m0, C0, k0, kend, dev):
    """Рекурсія замикання з шару `k0+1` по `kend`, старт `(m0, C0)`."""
    m, C = m0.clone(), C0.clone()
    for i in range(k0, kend):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(
            ph / s, ph / s)
        dd = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dd - C.diagonal())
    return m


def draw_states(wc, npair, dev, seed, ks):
    """Емпіричні `(m̂, Ĉ)` на потрібних шарах з ОДНОГО жеребу сабміту."""
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    cv = (x.T @ x) / float(2 * npair)
    ev, V = torch.linalg.eigh(cv.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    a = x @ wh
    out = {}
    if 0 in ks:
        ad = a.double()
        out[0] = (torch.zeros(W_, dtype=torch.float64, device=dev),
                  torch.eye(W_, dtype=torch.float64, device=dev))
    for i in range(len(wc)):
        a = torch.relu(a @ wc[i])
        if (i + 1) in ks:
            ad = a.double()
            mm = ad.mean(0)
            Cc = (ad - mm).T @ (ad - mm) / float(ad.shape[0])
            out[i + 1] = (mm, Cc)
    return out


def main(nets=8, dev=DEV, ndraw=NDRAW):
    print('📣 ПЕРЕДБАЧЕНО: U-подібна крива, мінімум K=12..20 на 0.3-0.6%, '
          'до 0.02% НЕ дійде')
    print('🟢 ЖИВЕ при мінімумі ≤0.10%\n', flush=True)
    E = {k: [] for k in KS}
    for n in range(nets):
        W = bank_net(n, dev)
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            npair, _ = budget(wc, 8)
            mt, _ = truth_means(wc, dev, 310_000 + 877 * n)
            tgt = mt[TGT - 1] if TGT <= d else mt[d - 1]
            nrm = float(tgt.norm())
            for k in range(ndraw):
                st = draw_states(wc, npair, dev, 91_000_000 + 977 * n + k, KS)
                for K in KS:
                    if K not in st:
                        continue
                    m0, C0 = st[K]
                    mm = close_from(wc, m0, C0, K, TGT, dev)
                    E[K].append(float((mm - tgt).norm()) / nrm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {ndraw} жеребів · ціль = `m` на шарі {TGT}')
    print(f'{"K (перезапуск)":>16} {"похибка m₂₉":>14}')
    best, bk = 1e9, None
    for K in KS:
        if not E[K]:
            continue
        v = float(np.mean(E[K])) * 100
        tag = ''
        if K == 0:
            tag = '  ← чисте замикання (база)'
        if K == TGT:
            tag = '  ← чиста вибірка ✅пломба'
        if 0 < K < TGT and v < best:
            best, bk = v, K
        print(f'{K:>16} {v:>13.4f}%{tag}')
    print(f'\n📏 ЦІЛЬ ЛІДЕРА: 0.0200%   ·   наш мінімум: {best:.4f}% (K={bk})')
    print(f'🔑 ВИРОК: {"ЖИВЕ" if best <= 0.10 else "МЕРТВО"} '
          f'(🟢гейт 0.10%) · до цілі не вистачає {best/0.02:.1f}×')
    print('📊 БАЗОВИЙ БАЛ 1.954e-07 · лідер 8e-9')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
