"""A/B ДВОХ ФАЙЛІВ-ЕСТИМАТОРІВ ЯК ВОНИ Є (не копій!). (С73)

🚨 ЧОМУ САМЕ ФАЙЛИ. Сесія 72 змірялa «новий канал» проти `base_real.py`, який
виявився портом АБЛЯЦІЇ (`estimator_side1.py`, три з чотирьох родин занулені) —
і всі пломби при цьому пройшли. Урок: **пломби перевіряють стенд, а не вибір
бази**; єдиний захист — вантажити ТЕ, ЩО ВІДПРАВЛЯЄТЬСЯ.
Тут обидва кандидати вантажаться через `bench.load_estimator` з .py-файлів.

ЩО МІРЯЄМО: ціну (`множник = max(0.1, (F + λ·R)/B)`) І точність (MSE проти
банкової істини) — правка ціни не має права зачепити другу колонку.

КОНТРОЛІ
  ⛔A′ — той самий sub72 ДРУГИЙ раз під іншим ім'ям: підлога роздільності стенда;
  🔀 варіанти ЧЕРГУЮТЬСЯ всередині повтору (без цього стенд міряє ДРЕЙФ:
     перший варіант 11.7 мс, останній 18.3 мс на тих самих даних);
  📊 облік ПО-МЕРЕЖНИЙ (`npair`, а з ним і R, різні на різних мережах).
"""
import glob
import statistics as st
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)

import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
from bench import load_estimator                        # noqa: E402

B = 272_000_000_000
LAM = 1e11
# С82: `_W_BLEND` 0.0651 → 0.0883. Оптимум на БАНКОВІЙ істині (48 мереж),
# чесний спліт-хаф ×1.0056, обидві половини дали 0.0886 / 0.0880.
# С84: sub93 = sub92 + ЛОКАЛЬНА чистка δ_Q Еджвортовою поправкою до E[a²],
# `γ` — вибіркові, з ТОГО Ж проходу. Стенд `side66/qsamp.py` дав ×1.0054
# (12 мереж), обернений знак ×0.9910. Ціна: 2 множення на елемент + 2 редукції
# на шарах 1..8, матмулів НЕ додає ⟹ колонка ціни мусить майже не змінитись.
CAND = [('A sub87 (діючий сабміт)', 'estimator_sub87.py'),
        ('B sub92 (готовий, w=0.0883)', 'estimator_sub92.py'),
        ('C sub93 = sub92 + чистка q', 'estimator_sub93.py'),
        ('D sub95 = sub93 + голограф. масштаб', 'estimator_sub95.py'),
        ("⛔A′ sub87 ще раз (пломба)", 'estimator_sub87.py')]

# ✅КРОС-ЦІЛЬ (С77). 🚨 Банкова `a_true` (400k пар) має ВЛАСНИЙ шум 4.84e-07 по
# MSE — це 24% робочого MSE, і він входив АДИТИВНО в обидві колонки, стискаючи
# надлишок відношення над одиницею у ≈0.81×. Тобто цей стенд систематично
# ЗАНИЖУВАВ усі виграші. Метадані `avar_true/(2·pairs)` про це брешуть у 8.1×.
# Ліки: MSE = E[(p−T1)(p−T2)] по ДВОХ незалежних істинах (`side66/mktruth.py`,
# по 1M пар кожна, власний шум 2.7e-08) ⟹ шум істини має нульове середнє
# в добутку і з відношення зникає.
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i]) for i, v in enumerate(_TT['seeds'])}


def main(reps=11, warm=3, nets=8, off=0):
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    fs = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][off:off + nets]
    est = {nm: load_estimator(SK + '/' + p)() for nm, p in CAND}

    per = {nm: [] for nm, _ in CAND}
    mse = {nm: [] for nm, _ in CAND}
    dmax = {nm: 0.0 for nm, _ in CAND}
    nops = {nm: 0 for nm, _ in CAND}
    allR = []
    for f in fs:
        zz = np.load(f)
        s0 = int(zz['seed'])
        if s0 not in TRUTH2:
            raise SystemExit(f'нема крос-істини для {s0}: доганяй '
                             f'`side66/mktruth.py cuda:1 <більше мереж>`')
        T1, T2 = TRUTH2[s0]
        j = int(np.where(sd == s0)[0][0])
        Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
        mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)

        for _ in range(warm):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                est[CAND[0][0]].predict(mlp, B)
        loc = {nm: [] for nm, _ in CAND}
        last = {}
        for _ in range(reps):
            for nm, _p in CAND:                       # 🔀 чергування
                with flops.BudgetContext(flop_budget=B, quiet=True) as c:
                    r = est[nm].predict(mlp, B)
                d = c.summary_dict()
                loc[nm].append((d['residual_wall_time_s'], c.flops_used))
                nops[nm] = len(c._op_log)     # 🔑 ГОЛОВНА метрика: грейдер
                #    тарифікує ВИКЛИКИ (12.3 мкс кожен), а не локальний час
                allR.append(d['residual_wall_time_s'])
                last[nm] = np.asarray(r)[-1].astype(np.float64)
        ref = last[CAND[0][0]]
        for nm, _p in CAND:
            R = st.median([q[0] for q in loc[nm]])
            F = st.median([q[1] for q in loc[nm]])
            per[nm].append((R, max(0.1, F / B + LAM * R / B)))
            mse[nm].append(float(((last[nm] - T1) * (last[nm] - T2)).mean()))
            dmax[nm] = max(dmax[nm], float(np.abs(last[nm] - ref).max()))
        print(f'  … {s0}  R: ' + '  '.join(
            f'{nm.split()[0]}={per[nm][-1][0]*1000:.2f}' for nm, _ in CAND),
            flush=True)

    a0 = CAND[0][0]
    print(f'\n{len(fs)} мереж × {reps} повторів (прогрів {warm}); медіана по '
          f'повторах, потім по мережах')
    print('✅КРОС-ЦІЛЬ: MSE = E[(p−T1)(p−T2)] ⟹ шум істини НЕ стискає відношення\n')
    print(f'{"кандидат":30s} {"R, мс":>8s} {"ΔR, мс":>8s} {"множник":>9s} '
          f'{"ЦІНА":>8s} {"кращих":>7s} {"MSE":>10s} {"max|Δ|":>9s} {"опер.":>7s} {"Δоп":>6s}')
    print('-' * 110)
    for nm, _ in CAND:
        dR = [per[nm][i][0] - per[a0][i][0] for i in range(len(fs))]
        gn = [per[a0][i][1] / per[nm][i][1] for i in range(len(fs))]
        # 🔑 ПОВНИЙ бал = MSE × множник, відношення СУМ (не середнє відношень)
        sc = sum(mse[nm][i] * per[nm][i][1] for i in range(len(fs)))
        sc0 = sum(mse[a0][i] * per[a0][i][1] for i in range(len(fs)))
        print(f'{nm:30s} {st.median([q[0] for q in per[nm]])*1000:7.3f} '
              f'{st.median(dR)*1000:+8.3f} '
              f'{st.median([q[1] for q in per[nm]]):9.5f} '
              f'{st.median(gn):7.4f}× {sum(1 for q in gn if q>1):4d}/{len(fs)} '
              f'{st.median(mse[nm]):10.3e} {dmax[nm]:9.2e} '
              f'{nops[nm]:7d} {nops[nm]-nops[a0]:+6d}')
        if nm != a0:
            print(f'{"  ⟹ ПОВНИЙ бал (MSE×множник)":30s} {"":8s} {"":8s} '
                  f'{"":9s} {sc0/sc:7.4f}×')
    print('-' * 110)
    print(f'розкид R по ВСІХ прогонах: {min(allR)*1000:.2f}–'
          f'{max(allR)*1000:.2f} мс ⟹ 🚨вимірні лише зміни ≳'
          f'{0.05*st.median([q[0] for q in per[a0]])*1000:.2f} мс')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 11,
         nets=int(sys.argv[2]) if len(sys.argv) > 2 else 8,
         off=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
