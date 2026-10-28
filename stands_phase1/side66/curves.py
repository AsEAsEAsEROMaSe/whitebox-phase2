"""КРИВІ «MSE проти БЮДЖЕТУ» для діючого топу (sub87) і давнього v2.

Обидва естиматори масштабують число семплів від АРГУМЕНТУ `budget`, тож свіп
робиться чесно: `predict(mlp, B*f)`, а ціна МІРЯЄТЬСЯ реальна (флопи + residual),
не постулюється. Це і є та крива, на якій стоїть уся наша арифметика:
    `бал = MSE × max(0.1, C/B)` ⟹ вище підлоги `MSE×бюджет = const` (гіпербола),
    нижче підлоги множник заморожений на 0.1 і зайва економія дає НУЛЬ.

📣 ПЕРЕДБАЧЕНО: обидві криві — прямі з нахилом −1 у лог-лог (MSE ∝ 1/C), у sub87
   константа приблизно в 1.58× нижча (3.09e-7 / 1.954e-7). Обидва оптимуми — РІВНО
   на зламі C/B=0.1.
🟢 ЖИВЕ як вимір, якщо нахил у межах −1.0 ± 0.1 і бал на підлозі відтворює
   1.954e-07 та 3.09e-07 у межах шуму банкової істини.
"""
import glob
import json
import statistics as st
import sys

import numpy as np

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
import flopscope as flops                                # noqa: E402
import flopscope.numpy as fnp                            # noqa: E402
from whestbench import MLP                               # noqa: E402
from bench import load_estimator                         # noqa: E402

B = 272_000_000_000
LAM = 1e11
NETS = int(sys.argv[1]) if len(sys.argv) > 1 else 16
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
FACT = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
CAND = [('sub87 (діючий топ)', 'estimator_sub87.py'),
        ('v2', 'estimator_whiten_v2_submitted_3p09e7.py')]

wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): i for i, v in enumerate(sd)}
fs = [q for q in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(q)['seed']) in have][:NETS]
est = {nm: load_estimator(SK + '/' + p)() for nm, p in CAND}
# ✅КРОС-ЦІЛЬ: дві НЕЗАЛЕЖНІ істини (1M пар кожна, `side66/mktruth.py`).
# 🚨 Банкова `a_true` має власний шум ≈3e-7 по MSE і створювала ФАЛЬШИВЕ
# «плато зміщення» на 6.9e-7 в ОБОХ естиматорів; метадані `avar_true/(2·pairs)`
# занижують цей шум у ~7×. `E[(p−T1)(p−T2)]` незміщена за будь-якого шуму істини.
_T = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TR = {int(v): (_T['T1'][i], _T['T2'][i]) for i, v in enumerate(_T['seeds'])}

acc = {(nm, f): {'cb': [], 'mse': []} for nm, _ in CAND for f in FACT}
for k, q in enumerate(fs):
    zz = np.load(q)
    s0 = int(zz['seed'])
    T1, T2 = TR[s0]
    Wn = [fnp.array(np.asarray(wz[have[s0]][i], np.float32)) for i in range(32)]
    mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)
    for _ in range(2):                                           # прогрів
        with flops.BudgetContext(flop_budget=B, quiet=True):
            est[CAND[0][0]].predict(mlp, B)
    for f in FACT:
        for nm, _p in CAND:                                      # 🔀 чергування
            loc = []
            last = None
            for _ in range(REPS):
                with flops.BudgetContext(flop_budget=int(B * 64), quiet=True) as c:
                    r = est[nm].predict(mlp, int(B * f))
                d = c.summary_dict()
                loc.append((d['flops_used'] + LAM * d['residual_wall_time_s']) / B)
                last = np.asarray(r)[-1].astype(np.float64)
            acc[(nm, f)]['cb'].append(st.median(loc))
            acc[(nm, f)]['mse'].append(float(((last - T1) * (last - T2)).mean()))
    print(f'  … {k+1}/{len(fs)}  {s0}', flush=True)

out = {}
print(f'\n{len(fs)} мереж × {REPS} повторів. Ціна ЗМІРЯНА, не постульована.')
print('✅КРОС-ЦІЛЬ: MSE = E[(p−T1)(p−T2)], дві істини по 1M пар ⟹ шум істини НЕ входить\n')
for nm, _ in CAND:
    print(f'=== {nm} ===')
    print(f'{"×бюджет":>8s} {"C/B":>9s} {"множник":>9s} {"MSE":>11s} {"БАЛ":>11s}')
    print('-' * 54)
    rows = []
    per = {}
    for f in FACT:
        cb = float(np.mean(acc[(nm, f)]['cb']))
        mse = float(np.mean(acc[(nm, f)]['mse']))
        mult = max(0.1, cb)
        rows.append({'f': f, 'cb': cb, 'mult': mult, 'mse': mse,
                     'score': mse * mult,
                     'per_net_mse': acc[(nm, f)]['mse'],
                     'per_net_cb': acc[(nm, f)]['cb']})
        print(f'{f:8.2f} {cb:9.4f} {mult:9.4f} {mse:11.4e} {mse*mult:11.4e}')
    out[nm] = rows
    hi = [r for r in rows if r['cb'] > 0.12]
    if len(hi) >= 2:
        lx = np.log([r['cb'] for r in hi]); ly = np.log([r['mse'] for r in hi])
        print(f'🔑 нахил у лог-лог вище підлоги: {np.polyfit(lx, ly, 1)[0]:+.3f} '
              f'(теорія −1.000)')
        print(f'🔑 константа MSE×бюджет: '
              f'{np.mean([r["mse"]*r["cb"] for r in hi]):.4e}')
    print()
json.dump(out, open('/mnt/win_d/llm/White-Box/arena62/logs/curves.json', 'w'))
print('📊 сирі точки → arena62/logs/curves.json')
