"""ЦІНА САБМІТІВ ПІД ДВОМА ВЕРСІЯМИ КОШТОРИСУ. (С85)

🚨 ПРИВІД. Лист AIcrowd 04.08.2026: евалуатори перейшли на flopscope 0.10.0 +
whestbench 0.14.0. Змінилось ЧОТИРИ речі, і кожна б'є саме по нас:
  1. арифметика тарифікується ПО DTYPE (ми рахуємо у float64);
  2. комплексні операції — по РЕАЛЬНИХ, у які розкладаються;
  3. **копії, gather'и, сортування БІЛЬШЕ НЕ БЕЗКОШТОВНІ** (у нас 2×
     `concatenate`, `stack`, `diagonal`, купа `maximum`);
  4. код учасника — на ОДНОМУ фізичному ядрі (у нас R входить у C через 1e11).

🔑 ЧОМУ ЦЕ ЕКСТРЕНО, А НЕ ЦІКАВО. Наш множник = **0.10000**, тобто рівно на
підлозі, а бюджет спожито на **9.97%** — запас до підлоги 0.03%. Будь-яке
подорожчання ≥0.3% зриває нас із підлоги, і бал росте ЛІНІЙНО за ціною.
Тобто це не «трошки гірше»: sub93 може з 1.928e-07 поїхати в рази.

ЩО МІРЯЄ: F (флопи), R (залишковий час), множник = max(0.1, (F+1e11·R)/B),
і кількість зареєстрованих операцій. ТІЛЬКИ ЦІНА — точність від версії
кошторису не залежить, тож істина тут не потрібна.

ЗАПУСК: тим самим файлом у двох середовищах —
  python3                       side66/costver.py   (старе: 0.8.0rc5 / 0.12.0rc3)
  <scratch>/fsnew/bin/python    side66/costver.py   (нове:  0.10.0  / 0.14.0)
"""
import glob
import importlib.metadata as md
import importlib.util
import os
import statistics as st
import sys

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp                            # noqa: F401
from whestbench import MLP

B = 272_000_000_000
LAM = 1e11
SK = '/mnt/win_d/llm/White-Box/whest-starterkit/'
CAND = [('sub87 (діючий)', 'estimator_sub87.py'),
        ('sub92', 'estimator_sub92.py'),
        ('sub93 (найкращий)', 'estimator_sub93.py')]


def load(path):
    spec = importlib.util.spec_from_file_location(
        'e_' + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Estimator


def probe(ctx):
    """Витягти (R, F, n_ops) незалежно від версії API."""
    R = 0.0
    try:
        d = ctx.summary_dict()
        for k in ('residual_wall_time_s', 'residual_wall_time',
                  'wall_time_s', 'residual_s'):
            if k in d:
                R = float(d[k])
                break
    except Exception:
        pass
    F = float(getattr(ctx, 'flops_used', 0.0))
    try:
        n = len(ctx._op_log)
    except Exception:
        n = -1
    return R, F, n


def main(reps=7, warm=3, nets=4, off=0):
    print(f'flopscope {md.version("flopscope")}  '
          f'whestbench {md.version("whestbench")}  '
          f'numpy {np.__version__}')
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    fs = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][off:off + nets]

    est = {nm: load(SK + p)() for nm, p in CAND}
    per = {nm: [] for nm, _ in CAND}
    nops = {nm: 0 for nm, _ in CAND}

    for f in fs:
        zz = np.load(f)
        s0 = int(zz['seed'])
        j = int(np.where(sd == s0)[0][0])
        Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
        mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)

        for _ in range(warm):
            with flops.BudgetContext(flop_budget=B, quiet=True):
                est[CAND[0][0]].predict(mlp, B)
        loc = {nm: [] for nm, _ in CAND}
        for _ in range(reps):
            for nm, _p in CAND:                          # 🔀 чергування
                with flops.BudgetContext(flop_budget=B, quiet=True) as c:
                    est[nm].predict(mlp, B)
                R, F, n = probe(c)
                loc[nm].append((R, F))
                nops[nm] = n
        for nm, _p in CAND:
            R = st.median([q[0] for q in loc[nm]])
            F = st.median([q[1] for q in loc[nm]])
            per[nm].append((R, F, max(0.1, F / B + LAM * R / B)))
        print(f'  … {s0}', flush=True)

    print(f'\n{len(fs)} мереж × {reps} повторів (прогрів {warm})')
    print(f'{"кандидат":20s} {"F, млрд":>10s} {"F/B":>8s} {"R, мс":>8s} '
          f'{"1e11R/B":>9s} {"C/B":>8s} {"МНОЖНИК":>9s} {"запас":>9s} {"опер.":>7s}')
    print('-' * 100)
    for nm, _ in CAND:
        F = st.median([q[1] for q in per[nm]])
        R = st.median([q[0] for q in per[nm]])
        cb = F / B + LAM * R / B
        print(f'{nm:20s} {F/1e9:10.3f} {F/B:8.5f} {R*1000:8.3f} '
              f'{LAM*R/B:9.5f} {cb:8.5f} {max(0.1, cb):9.5f} '
              f'{(0.1-cb)/0.1*100:+8.2f}% {nops[nm]:7d}')
    print('-' * 100)
    print('запас = наскільки C/B нижче підлоги 0.1; мінус ⟹ 🚨 ЗІРВАЛИСЬ')


if __name__ == '__main__':
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 7,
         nets=int(sys.argv[2]) if len(sys.argv) > 2 else 4)
