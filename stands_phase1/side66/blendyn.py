"""ГЛОБАЛЬНІ ПО МЕРЕЖІ ВЕЛИЧИНИ: ДИНАМІЧНИЙ БЛЕНД І СПІЛЬНИЙ МНОЖНИК λ (С82)

## ІДЕЯ ЮЗЕРА
«думаю не все ми чекнули. замикання по слоях глобально і по мережі глобально»

Досі динаміку міряли ПОШАРОВО (31 число на мережу) — і вона програла шуму
([[finding-shrinkage-weight-is-computable-per-network]]). Але є два ГЛОБАЛЬНІ
числа, по ОДНОМУ на мережу, і вони на порядок стабільніші:

  1️⃣ **`_W_BLEND`** — сабміт закінчує рядком
     `out = ours + 0.0651·((1+_ALPHA)·m_deep − ours)`,
     тобто змішує наш прогноз із ГЛИБОКИМ замиканням однією константою на всі
     мережі. А вага мусить залежати від того, наскільки добре замикається САМЕ
     ЦЯ мережа.
  2️⃣ **спільний множник `g(мережа)` на ВСІ λ** — одна ручка «наскільки взагалі
     довіряти замиканню цієї мережі».

## 🔑 ЧОМУ ЦЕ МОЖЕ ПРАЦЮВАТИ ТАМ, ДЕ ПОШАРОВЕ ПРОВАЛИЛОСЬ
* пошарова оцінка `B²` — різниця двох шумних величин на 256 нейронах;
  глобальна — та сама різниця, але агрегована по ВСІЙ мережі ⟹ шуму менше;
* бленд діє НА ВИХОДІ, де живе приз, а не на глибині. Пастка «точність
  формули росте з глибиною, а цінність каналу падає»
  ([[finding-deep-channels-die-against-the-real-submission]]) тут не діє.

## 📐 ФОРМУЛА (та сама тотожність, але глобально)
`ours − m_deep = (ours − істина) + (істина − m_deep) = шум − B`, і вони
незалежні ⟹
    `‖ours − m_deep‖² = VO + B²`, де `VO = Σ_j Var(a_j)/(2·n_pairs)` виходу
    ⟹ **`w* = VO / ‖ours − m_deep‖²`** — усе з уже обчисленого, ціна нуль.

## ВАРІАНТИ (12 мереж × 32 жереби, база = ПОРТ САБМІТА, крос-ціль)
  0 ⛔ сабміт як є (`w=0.0651`)
  1 🎯 оракульний `w` НА МЕРЕЖУ — СТЕЛЯ динаміки бленду
  2 🎯 `w = c·w_форм(мережа)`, `c` — корпусна константа LOO
  3 ⛔ `w_форм` із ЧУЖОЇ мережі — контроль
  4 📐 найкращий ЄДИНИЙ `w` по корпусу (LOO) — чи взагалі 0.0651 оптимальне
  + кореляція `w_форм ↔ w_орак` по мережах

📣 ПЕРЕДБАЧЕНО: оракульна стеля бленду буде помітною (1.05…1.15×), бо `w`
   ніколи не персоналізували; але `w_форм` вгадає слабо, бо `VO` — це шум
   ЛИШЕ вибірки, а `ours` має ще й похибку переносу, якої формула не бачить.
   ⟹ чекаю високу стелю і низьку реалізацію. Якщо ж корел >0.6 і виграш
   >1.02× — це прямий приріст за нульову ціну.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/blendyn.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, predict91,      # noqa: E402
                        W_BLEND, ALPHA)
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
L_, NDRAW, KS = 32, 32, 31
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

# на кожну мережу: накопичувачі для оптимального `w`
NUM = np.zeros(NETS)      # Σ⟨T − ours, Δ⟩      (Δ = (1+α)m_deep − ours)
DEN = np.zeros(NETS)      # Σ‖Δ‖²
BASE = np.zeros(NETS)     # крос-MSE сабміту як є
WF = np.zeros(NETS)       # формульний `w`
Z0 = np.zeros(NETS)       # Σ‖ours − T‖ крос (для w=0)
CR = np.zeros((NETS, 3))  # ⟨e0,Δ⟩ для e1, e2 і ⟨Δ,Δ⟩
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, 8, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        d2s, vos = 0.0, 0.0
        for d in range(NDRAW):
            pr, _, _, (ours, m_deep, vout) = predict91(
                wc, npair, DEV, seed=77_000_000 + 733 * n + d, cl=cl, raw=True)
            dl = (1.0 + ALPHA) * m_deep - ours
            a1, a2 = ours - T1, ours - T2
            CR[n, 0] += float((a1 * dl).sum())
            CR[n, 1] += float((a2 * dl).sum())
            CR[n, 2] += float((dl * dl).sum())
            Z0[n] += float((a1 * a2).sum())
            BASE[n] += float(((pr - T1) * (pr - T2)).sum())
            d2s += float(((ours - m_deep) ** 2).sum())
            vos += vout
        WF[n] = vos / max(d2s, 1e-300)
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

# 🔑 крос-MSE як функція `w`. Виводимо ЧЕСНО, бо перша версія помилилась
#    у ЗНАКУ і ⛔пломба це спіймала (розбіжність 11.5%):
#      e₁ = (ours + wΔ) − T₁ = a₁ + wΔ,  e₂ = a₂ + wΔ
#      Σe₁e₂ = Σa₁a₂ + w·Σ(a₁Δ + a₂Δ) + w²·ΣΔ²
#    ⟹ лінійний член іде з ПЛЮСОМ, а мінімум у `w* = −NUM/(2·DEN)`.
NUM = CR[:, 0] + CR[:, 1]
DEN = CR[:, 2]


def mse(wv):
    return float(np.sum(Z0 + wv * NUM + wv * wv * DEN))


tot0 = float(BASE.sum())
seal = mse(np.full(NETS, W_BLEND))
print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ.')
print(f'⛔ ПЛОМБА КВАДРАТИЧНОЇ ФОРМИ: MSE(0.0651) = {seal:.6e} проти прямого '
      f'{tot0:.6e}, відн. {abs(seal-tot0)/tot0:.2e} '
      f'{"✅" if abs(seal-tot0)/tot0 < 1e-9 else "🚨 форма не та"}')

w_or = -NUM / (2.0 * np.maximum(DEN, 1e-300))       # оракул на мережу
c = float(np.corrcoef(WF, w_or)[0, 1])
print(f'\n🎯 КОРЕЛЯЦІЯ w_форм ↔ w_оракул по мережах: **{c:+.4f}**')
print(f'   w_форм:  сер {WF.mean():.4f}  розкид {WF.std():.4f}  '
      f'[{WF.min():.4f}…{WF.max():.4f}]')
print(f'   w_орак:  сер {w_or.mean():.4f}  розкид {w_or.std():.4f}  '
      f'[{w_or.min():.4f}…{w_or.max():.4f}]   (у сабміті {W_BLEND})')

# LOO по мережах для корпусних величин
def loo_scalar(pred):
    """`w_n = c·pred_n`, `c` — з ІНШИХ мереж (LOO)."""
    tot = 0.0
    for n in range(NETS):
        o = [j for j in range(NETS) if j != n]
        num = -sum(NUM[j] * pred[j] for j in o)
        den = 2.0 * sum(DEN[j] * pred[j] ** 2 for j in o)
        cc = num / max(den, 1e-300)
        w = cc * pred[n]
        tot += Z0[n] + w * NUM[n] + w * w * DEN[n]
    return tot


def loo_const():
    tot = 0.0
    for n in range(NETS):
        o = [j for j in range(NETS) if j != n]
        w = -sum(NUM[j] for j in o) / (2.0 * sum(DEN[j] for j in o))
        tot += Z0[n] + w * NUM[n] + w * w * DEN[n]
    return tot


rows = [('⛔ сабміт як є (w=0.0651)', tot0),
        ('🎯 ОРАКУЛ: свій w на мережу (СТЕЛЯ)', mse(w_or)),
        ('🎯 w = c·w_форм (LOO по мережах)', loo_scalar(WF)),
        ('⛔ w_форм із ЧУЖОЇ мережі', loo_scalar(np.roll(WF, 1))),
        ('📐 найкращий ЄДИНИЙ w (LOO)', loo_const()),
        ('⛔ w = 0 (без бленду взагалі)', mse(np.zeros(NETS)))]
print(f'\n{"варіант":38s} {"крос-MSE":>12s} {"виграш":>9s}')
print('─' * 62)
for nm, v in rows:
    print(f'{nm:38s} {v:12.5e} {tot0/v:8.4f}×')
g = tot0 / loo_scalar(WF)
print(f'\n📊 БАЛ 1.954e-07 ⟹ динамічний бленд {1.954e-07/max(g,1e-9):.4e}, '
      f'стеля {1.954e-07/(tot0/mse(w_or)):.4e}')
