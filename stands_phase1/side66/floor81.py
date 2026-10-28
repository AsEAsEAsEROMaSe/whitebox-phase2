"""ДЕ НАША ВЛАСНА ПІДЛОГА: ЧИ ТАНЕ ПЕРЕВАГА НАД МК, КОЛИ ГРОШЕЙ БІЛЬШЕ? (С81)

## ЗВІДКИ ПИТАННЯ
`side66/wbudget.py` дав дивну пару на ширині 1024:
    ×16 → `sub91` / МК = **2.2167×**
    ×32 → `sub91` / МК = **1.9443×**   ← БІЛЬШЕ ГРОШЕЙ, МЕНША ПЕРЕВАГА
Якби ми були чистим зниженням дисперсії, відношення трималось би СТАЛИМ:
обидва падають як `1/n`. Спад означає, що в нас є доданок, який `n` НЕ ЛІКУЄ —
власна підлога (зміщення замикання + похибка поправок). МК підлоги не має.

## 🎯 ЧОМУ ЦЕ ГОЛОВНЕ ПИТАННЯ ГІЛКИ
Розрив до лідера **24.4×**. Якщо наша перевага тане з бюджетом, то шлях
«рахувати більше» закритий не лише правилом `MSE×бюджет=const`, а й власною
підлогою — і тоді єдиний хід узагалі — ЗНИЖУВАТИ ПІДЛОГУ, а не купувати
семпли. Якщо ж на 256 відношення ТРИМАЄТЬСЯ до ×64, то підлога далеко, і
спад на 1024 має іншу причину (запас пар над шириною).

## ЩО МІРЯЄМО
Ширина 256, глибина 32, **48 мереж** банку, множник бюджету 1…128.
✅**КРОС-ЦІЛЬ**: `MSE = E[(p−T1)(p−T2)]` по двох незалежних істинах
(`bank_s53/truth2.npz`, власний шум 1.47e-08) — інакше на великих множниках
наша MSE впаде нижче шуму істини і вимір стане беззмістовним.

⛔ ПЛОМБА: при ×1 відношення мусить бути ≈**3.4×** (корпусне число).

📣 ПЕРЕДБАЧЕНО: відношення тримається до ×4…×8, далі повільно тане і до ×128
   сідає в район 1.5…2×. Тобто підлога існує і лежить на 1…2 порядки нижче
   нинішньої MSE. Якщо ж воно тримається пласко до ×128 — підлоги в цьому
   діапазоні нема, і спад на 1024 треба пояснювати ЗАПАСОМ ПАР.

Запуск: python3 side66/floor81.py
"""
import glob
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
import numpy as np                                       # noqa: E402
import flopscope as flops                                # noqa: E402
import flopscope.numpy as fnp                            # noqa: E402
from whestbench import MLP                               # noqa: E402
from bench import load_estimator                         # noqa: E402
from flopscope.errors import BudgetExhaustedError        # noqa: E402

flops.configure(symmetry_warnings=False)
B0, LAM, D, W = 272_000_000_000, 1e11, 32, 256
MULTS = [1, 2, 4, 8, 16, 32, 64, 128]
CAND = [('⛔ чистий МК', SK + '/estimator_mc_plain.py'),
        ('🟠 sub91', SK + '/estimator_sub91.py')]

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
FS = [q for q in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(q)['seed']) in TRUTH2]
print(__doc__.split('Запуск')[0], flush=True)
print(f'мереж із крос-істиною: {len(FS)}\n', flush=True)

NETS = []
for f in FS:
    s0 = int(np.load(f)['seed'])
    j = int(np.where(SD == s0)[0][0])
    NETS.append((s0, j))


def run(path, mult):
    B = B0 * mult
    E = load_estimator(path)()
    ms, cs = [], []
    for s0, j in NETS:
        T1, T2 = TRUTH2[s0]
        Wn = [fnp.array(np.asarray(WZ[j][k], np.float32)) for k in range(D)]
        blown = False
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            try:
                o = E.predict(MLP(width=W, depth=D, weights=Wn, seed=s0), B)
            except BudgetExhaustedError:
                blown, o = True, np.zeros((D, W))
        C = c.flops_used + LAM * c.summary_dict()['residual_wall_time_s']
        a = np.asarray(o)[-1].astype(np.float64)
        if blown or C > B:
            a = np.zeros_like(a)
        ms.append(float(np.mean((a - T1) * (a - T2))))     # ✅крос-ціль
        cs.append(1.0 if (blown or C > B) else max(0.1, C / B))
    return (float(np.mean(ms)), float(np.mean(cs)),
            float(np.mean([x * y for x, y in zip(ms, cs)])))


print(f'{"множник":>8s} {"MSE МК":>11s} {"MSE sub91":>11s} {"ціна":>7s} '
      f'{"MSE МК/наша":>12s} {"БАЛ проти МК":>13s}')
print('-' * 72)
prev = None
for mult in MULTS:
    m_mc, c_mc, s_mc = run(CAND[0][1], mult)
    m_91, c_91, s_91 = run(CAND[1][1], mult)
    flag = ''
    if prev is not None:
        flag = '  🔻ТАНЕ' if m_mc / m_91 < prev * 0.95 else '  ▬тримається'
    prev = m_mc / m_91
    print(f'{"×" + str(mult):>8s} {m_mc:11.4e} {m_91:11.4e} {c_91:7.4f} '
          f'{m_mc / m_91:11.4f}× {s_mc / s_91:12.4f}×{flag}', flush=True)

print('\n🔑 ЧИСТЕ ЗНИЖЕННЯ ДИСПЕРСІЇ дало б СТАЛУ колонку «MSE МК/наша».')
print('   Спад = наша MSE перестала падати як 1/n ⟹ уперлись у ВЛАСНУ ПІДЛОГУ.')
print('📐 для порівняння: на 1024 відношення балів упало 2.2167× (×16) → '
      '1.9443× (×32)')
print('📊 БАЛ 1.954e-07 · розрив до лідера 24.4×')
