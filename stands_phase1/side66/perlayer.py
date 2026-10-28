"""ВАГА КОЖНОГО ШАРУ ДЛЯ КОЖНОЇ МЕРЕЖІ — ОБЧИСЛЕНА, А НЕ ПІДІГНАНА (С82)

## ІДЕЯ ЮЗЕРА
«якщо для кожної мережі свої слої дають інфу то може шукати динамічно» ·
«не можна написати алгоритм що вкладеться в бюджет і зможе шукати для кожної
мережі слої де найбільше вніс у потрібному параметрі»

## 🔑 ЧОМУ ЦЕ ВЗАГАЛІ МОЖЛИВО БЕЗ ІСТИНИ
Канал шару бачить `δ = x̄ − m_замик = e + B`, де `e` — шум вибірки (те, що
знімаємо), `B` — зміщення замикання (сміття, стале для мережі). С73 вивела
оптимум `λ* = E[e²]/(E[e²]+B²)`
([[finding-deep-anchor-dies-on-closure-error-not-on-lambda]]).
**Обидві частини ВИМІРНІ зсередини:**
    `E[e²] = Σ_j Var(a_j)/(2·n_pairs)`   ← чиста арифметика, дисперсія вже є
    `E[δ²] = Σ_j δ_j²`                    ← редукція по вже обчисленому
    `B² = E[δ²] − E[e²]`  (бо `e ⟂ B`)  ⟹  **`λ* = E[e²]/E[δ²]`**
Ціна: дві редукції на шар. НУЛЬ матмулів, нуль додаткових семплів.

## 🚨 ЧОМУ ЦЕ НЕ ТЕ, ЩО ВЖЕ ЗАКРИТО
С2 міряла те саме питання і вперлась: «`r` не передбачається — 5 обчислюваних
кандидатів дають |кореляцію| ≤ 0.18»
([[finding-s2-per-element-beta-is-worth-1.8x-blocker-is-predicting-r]]).
Але там шукали ПРЕДИКТОР `r` серед ознак. Тут `r` не вгадується, а виводиться
з ТОТОЖНОСТІ `δ = e + B`. Це інший об'єкт — і на нього старий вирок не
поширюється, доки не зміряно.
⚠️ Ризик відомий заздалегідь: `B²` береться як РІЗНИЦЯ двох шумних величин,
тож на шарах, де `B² ≪ E[e²]`, вона може вийти від'ємною.

## ЩО МІРЯЄМО (12 мереж × 32 жереби, база = ПОРТ САБМІТА, крос-ціль)
1. `λ_форм(мережа, шар)` — за формулою вище, з вибірки;
2. `λ_орак(мережа, шар)` — оракульний, з регресії залишку на перенесену δ
   ЦІЄЇ мережі (⚠️ завищує: підбір і вимір на тих самих даних);
3. кореляція між ними — **головне число**: чи формула вгадує, де густо;
4. виграш від заміни корпусних λ на `λ_форм`;
5. ⛔ПЛОМБА: `E[δ²] ≥ E[e²]` мусить триматись (інакше оцінка шуму завищена);
6. ⛔КОНТРОЛЬ: `λ_форм` із ЧУЖОЇ мережі.

📣 ПЕРЕДБАЧЕНО: кореляція вийде ВИСОКОЮ на мілких шарах (там `B` мале, `λ*`
   близьке до 1) і впаде на глибоких (там `B²` домінує, а різниця шумна).
   Виграш чекаю малий (≤1.02×), бо корпусні λ вже стоять у середньому
   оптимумі, а розкид по мережах може бути вузьким. 🎯 Але якщо кореляція
   висока — це відкриває ДИНАМІЧНИЙ режим як клас, і тоді сенс не в цьому
   виграші, а в тому, що ваги можна СТАВИТИ, а не підганяти на корпусі.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/perlayer.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import cut_budget91, closure91, predict91        # noqa: E402
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
COLS = list(range(KS))
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

LF = np.zeros((NETS, KS))       # λ за формулою
LO = np.zeros((NETS, KS))       # λ оракульний
NEG = 0
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
        num = torch.zeros(KS, dtype=torch.float64, device=DEV)
        den = torch.zeros(KS, dtype=torch.float64, device=DEV)
        d2 = np.zeros(KS); e2 = np.zeros(KS)
        for d in range(NDRAW):
            pr, R, DG = predict91(wc, npair, DEV,
                                  seed=77_000_000 + 733 * n + d, cl=cl,
                                  collect=COLS, diag=True)
            e = 0.5 * ((pr - T1) + (pr - T2))
            Rm = R[1::4]                       # ⟵ родина μ (f=1)
            num += Rm @ e
            den += (Rm * Rm).sum(1)
            for k, (D, E, _) in enumerate(DG[:KS]):
                d2[k] += D; e2[k] += E
        LO[n] = (num / den.clamp_min(1e-300)).cpu().numpy()
        LF[n] = e2 / np.maximum(d2, 1e-300)
        NEG += int((d2 < e2).sum())
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

print(f'\n{NETS} мереж × {NDRAW} жеребів')
print(f'⛔ ПЛОМБА `E[δ²] ≥ E[e²]`: порушень {NEG} із {NETS*KS} '
      f'{"✅" if NEG == 0 else "🚨 оцінка шуму завищена там, де B² мале"}')

fl, ol = LF.reshape(-1), LO.reshape(-1)
c = float(np.corrcoef(fl, ol)[0, 1])
print(f'\n🎯 КОРЕЛЯЦІЯ λ_форм ↔ λ_орак по всіх (мережа,шар): **{c:+.4f}**')
print(f'   (для порівняння: 5 кандидатів С2 давали |корел| ≤ 0.18)')

print(f'\n{"шар":>4s} {"λ_форм сер":>11s} {"розкид":>9s} {"λ_орак сер":>11s} '
      f'{"розкид":>9s} {"корел ПО МЕРЕЖАХ":>17s}')
print('─' * 66)
for k in range(0, KS, 3):
    cc = (float(np.corrcoef(LF[:, k], LO[:, k])[0, 1])
          if LF[:, k].std() > 1e-12 else float('nan'))
    print(f'{k+1:4d} {LF[:, k].mean():11.4f} {LF[:, k].std():9.4f} '
          f'{LO[:, k].mean():11.4f} {LO[:, k].std():9.4f} {cc:+17.3f}')

print(f'\n🔑 РОЗКИД ПО МЕРЕЖАХ — це і є «чи варто динамічно»:')
rel = LO.std(0) / np.maximum(np.abs(LO.mean(0)), 1e-300)
print(f'   λ_орак: відносний розкид по мережах, медіана {np.median(rel):.3f}, '
      f'шари 1-8 {np.median(rel[:8]):.3f}, шари 24-31 {np.median(rel[23:]):.3f}')
np.savez('/mnt/win_d/llm/White-Box/arena62/logs/perlayer.npz', LF=LF, LO=LO)
print('📊 БАЛ 1.954e-07 · вирок про ВИГРАШ — наступним кроком, тут вирок про '
      'ПРИДАТНІСТЬ формули')
