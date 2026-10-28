"""ДЕ САМЕ sub93 ВИТРАЧАЄ 2927 ВИКЛИКІВ `fnp`. (С85)

🚨 ЧОМУ ЦЕ ВАЖЛИВО. Грейдер бере `C = F + 1e11·R`, і `R` — це залишковий
wall-time, який на грейдері виходить пропорційним КІЛЬКОСТІ ВИКЛИКІВ, а не
арифметиці: калібровка з sub87 (1222 виклики ⟺ λR/B = 0.0059) дає **12.4 мкс
на виклик**. sub93 має 2927 викликів ⟹ за тією ставкою λR/B = 0.0134, і разом
із F/B = 0.0879 це **C/B = 0.1014 — НАД підлогою 0.1**.
На нашій машині цього не видно: тут виклик коштує ~6 мкс, і C/B = 0.0945.

⟹ кожні 100 зрізаних викликів = 1.24 мс = **0.46% бюджету**, які можна
віддати вибірці (`_TARGET_UTIL`), тобто ≈0.46% MSE.

ЩО РОБИТЬ: розбирає `_op_log` по (операція, рядок коду) і показує топ місць.
"""
import collections
import glob
import importlib.util
import os
import sys

import numpy as np

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP

SK = '/mnt/win_d/llm/White-Box/whest-starterkit/'
B = 272_000_000_000


def load(path):
    spec = importlib.util.spec_from_file_location(
        'e_' + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Estimator


def main(fname='estimator_sub93.py', top=25):
    wz = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    sd = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = set(int(v) for v in sd)
    f = [q for q in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(q)['seed']) in have][0]
    s0 = int(np.load(f)['seed'])
    j = int(np.where(sd == s0)[0][0])
    Wn = [fnp.array(np.asarray(wz[j][k], np.float32)) for k in range(32)]
    mlp = MLP(width=256, depth=32, weights=Wn, seed=s0)

    with flops.BudgetContext(flop_budget=B, quiet=True) as c:
        load(SK + fname)().predict(mlp, B)
    log = c._op_log
    print(f'{fname}: {len(log)} викликів, F = {c.flops_used/1e9:.3f} млрд')

    # 🔎 структура запису різна між версіями — дивимось на перший
    e0 = log[0]
    fields = (e0._fields if hasattr(e0, '_fields')
              else (list(e0.keys()) if isinstance(e0, dict) else None))
    print('поля запису:', fields if fields else type(e0))

    def get(e, *names):
        for n in names:
            if hasattr(e, n):
                return getattr(e, n)
            if isinstance(e, dict) and n in e:
                return e[n]
        return None

    cnt = collections.Counter()
    fl = collections.Counter()
    for e in log:
        op = get(e, 'op_name', 'op', 'name') or '?'
        sh = get(e, 'shapes')
        dt = get(e, 'resolved_dtype')
        loc = f'{sh} {dt}'
        cnt[(op, loc)] += 1
        fl[(op, loc)] += float(get(e, 'flop_cost', 'flops') or 0.0)

    print(f'\n{"операція":22s} {"форми / dtype":40s} {"викликів":>9s} '
          f'{"мкс@12.4":>9s} {"%бюдж":>7s} {"флопів,млн":>11s}')
    print('-' * 100)
    tot = 0
    for (op, loc), n in cnt.most_common(top):
        us = n * 12.4
        print(f'{str(op)[:22]:22s} {loc[:40]:40s} {n:9d} '
              f'{us:9.0f} {us*1e-6*1e11/B*100:7.3f} {fl[(op, loc)]/1e6:11.1f}')
        tot += n
    print('-' * 100)
    print(f'показано {tot} з {len(log)}; уся решта — {len(log)-tot}')
    print(f'ВСЬОГО за ставкою 12.4 мкс: {len(log)*12.4/1000:.1f} мс '
          f'⟹ λR/B = {len(log)*12.4e-6*1e11/B:.5f}  '
          f'(F/B = {c.flops_used/B:.5f}, разом C/B = '
          f'{c.flops_used/B + len(log)*12.4e-6*1e11/B:.5f})')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'estimator_sub93.py',
         int(sys.argv[2]) if len(sys.argv) > 2 else 25)
