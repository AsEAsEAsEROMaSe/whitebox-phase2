"""ДРУГЕ ГЛОБАЛЬНЕ ЧИСЛО: ОДИН МНОЖНИК `g(мережа)` НА ВСІ λ (С82)

## ІДЕЯ ЮЗЕРА
«замикання по слоях глобально і по мережі глобально» — друга половина.
`side66/blendyn.py` перевіряє глобальний БЛЕНД; тут — глобальну ДОВІРУ до
чотирьох λ-родин: `λ(мережа, ℓ, родина) = g(мережа) · λ_корпусне(ℓ, родина)`.
Одна ручка на мережу замість 48 констант на всіх.

## 🔑 ЧОМУ ЦЕ ОКРЕМЕ ПИТАННЯ
Пошарова динаміка програла шуму
([[finding-shrinkage-weight-is-computable-per-network]]), бо оцінювала 31
число з шумної різниці. Тут оцінюється ОДНЕ число, агреговане по всій
мережі — статистично це зовсім інший режим.

## ВАРІАНТИ (12 мереж × 32 жереби, база = ПОРТ САБМІТА, крос-ціль)
  0 ⛔ корпусні λ (LOO)                — база
  1 🎯 ОРАКУЛ: свій `g` на мережу      — СТЕЛЯ глобальної динаміки
  2 🎯 `g = c·g_форм(мережа)`          — реалізація формулою
  3 ⛔ `g_форм` із ЧУЖОЇ мережі        — контроль
`g_форм` — та сама тотожність, але агрегована по ВСІХ 8 шарах разом:
    `g_форм = Σ_ℓ E[e²]_ℓ / Σ_ℓ E[δ²]_ℓ`.

📣 ПЕРЕДБАЧЕНО: стеля вийде мала (1.02…1.05×), бо 48 корпусних коефіцієнтів
   уже підігнані на 60 мережах і середній масштаб у них правильний; а
   персоналізація одного скаляра не може дати більше, ніж розкид цього
   скаляра по мережах. Якщо стеля <1.02× — глобальний клас закрито ЦІЛКОМ
   (разом із блендом), і це остаточна відповідь на «шукати динамічно».

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/gscale.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, predict91,      # noqa: E402
                        LAM_S, LAM_A, LAM_Q, LAM_G)
# 🚨 ІСТИНА З БАНКУ (вказівка юзера С82: «є ж банк точності до e-10»).
#    ДВІ причини, друга важливіша: (1) банк точніший (шум 1.6e-04 відн. проти
#    моїх 2-3e-04); (2) 🔑 `truth_last(wc)` рахує істину ОБРІЗАНОЇ мережі, а
#    грейдер міряє ПОВНУ — зсув 1.5e-04…6.1e-04 БІЛЬШИЙ за обидва шуми, тобто
#    це ПОХИБКА ОБРІЗАННЯ, яку сабміт робить, а вимір її не бачив.
#    📌 Побічно: банк прибирає 2 виклики `truth_last` на мережу ⟹ прогін
#    прискорився в ~30× (48 мереж за 50 с замість 25 хв).
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
L_, NDRAW, KS = 32, 32, 8
COLS = list(range(KS))
NR = 4 * KS
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

G = [torch.zeros(NR, NR, dtype=torch.float64, device=DEV) for _ in range(NETS)]
B = [torch.zeros(NR, dtype=torch.float64, device=DEV) for _ in range(NETS)]
C0 = np.zeros(NETS)
GF = np.zeros(NETS)
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KS, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        d2s = e2s = 0.0
        for d in range(NDRAW):
            pr, R, DG = predict91(wc, npair, DEV,
                                  seed=77_000_000 + 733 * n + d, cl=cl,
                                  collect=COLS, diag=True, lam_off=True)
            e1, e2 = pr - T1, pr - T2
            C0[n] += float((e1 * e2).sum())
            G[n] += R @ R.T
            B[n] += R @ (0.5 * (e1 + e2))
            for D, E, _ in DG[:KS]:
                d2s += D; e2s += E
        GF[n] = e2s / max(d2s, 1e-300)
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)


def val(n, w):
    return C0[n] - 2.0 * float(w @ B[n]) + float(w @ (G[n] @ w))


# 🚨 ПОПРАВКА ПІСЛЯ ПЕРШОГО ПРОГОНУ: спершу базою був МІЙ LOO-фіт `corp(n)`, і
#    ⛔контроль «g з ЧУЖОЇ мережі» дав 1.1027× — ВИЩЕ за формулу 1.0652×.
#    Це не сигнал, а діагноз: мій фіт недофітований ЗА МАСШТАБОМ, тож будь-яке
#    домноження вгору його лікує. Правильна база — САМЕ САБМІТНІ λ, узяті як
#    ФІКСОВАНИЙ напрямок. Тоді `g` міряє рівно те, що питали: чи варто
#    масштабувати наявні коефіцієнти під мережу.
#    🎓 Це знову [[feedback-verify-code-after-the-experiment]] і
#    [[reference-sub91-torch-port-is-the-real-base]]: контроль вище гіпотези —
#    привід перевіряти БАЗУ, а не радіти стелі.
#    Знак: `pr_сабміт = pr₀ − λᵀR`, а стенд міряє `C0 − 2wᵀB + wᵀGw` для
#    `pr₀ − wᵀR` ⟹ підстановка `w = λ_сабміт` МУСИТЬ відтворити сабміт.
LAMV = torch.tensor([[LAM_S, LAM_A, LAM_Q, LAM_G][f][k]
                     for k in range(KS) for f in range(4)],
                    dtype=torch.float64, device=DEV)


def corp(excl, lam=1e-2):
    return LAMV


RES = {k: 0.0 for k in ('base', 'orac', 'form', 'alien', 'nolam')}
for n in range(NETS):
    w = LAMV
    RES['base'] += val(n, w)
    RES['nolam'] += C0[n]
    # 🎯 оракул: найкращий скаляр `g` при корпусному напрямку `w`
    gw = G[n] @ w
    go = float(w @ B[n]) / max(float(w @ gw), 1e-300)
    RES['orac'] += val(n, go * w)
    for key, pred in (('form', GF), ('alien', np.roll(GF, 1))):
        o = [j for j in range(NETS) if j != n]
        num = sum(pred[j] * float(LAMV @ B[j]) for j in o)
        den = sum(pred[j] ** 2 * float(LAMV @ (G[j] @ LAMV)) for j in o)
        c = num / max(den, 1e-300)
        RES[key] += val(n, (c * pred[n]) * w)

tot0 = RES['base']
print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, LOO ПО МЕРЕЖАХ.')
print(f'🔑 g_форм: сер {GF.mean():.4f} розкид {GF.std():.4f} '
      f'[{GF.min():.4f}…{GF.max():.4f}]')
print(f'\n{"варіант":40s} {"крос-MSE":>12s} {"виграш":>9s}')
print('─' * 64)
for k, nm in (('base', '⛔ САБМІТНІ λ як є'),
              ('nolam', '⛔ БЕЗ λ узагалі (перевірка масштабу)'),
              ('orac', '🎯 ОРАКУЛ: свій g на мережу (СТЕЛЯ)'),
              ('form', '🎯 g = c·g_форм(мережа)'),
              ('alien', '⛔ g_форм із ЧУЖОЇ мережі')):
    print(f'{nm:40s} {RES[k]:12.5e} {tot0/RES[k]:8.4f}×')
print(f'\n📊 БАЛ 1.954e-07 ⟹ стеля глобального `g` '
      f'{1.954e-07/(tot0/RES["orac"]):.4e}')
np.savez('/mnt/win_d/llm/White-Box/arena62/logs/gscale.npz',
         G=torch.stack(G).cpu().numpy(), B=torch.stack(B).cpu().numpy(),
         C0=C0, GF=GF, LAMV=LAMV.cpu().numpy())
