"""ДІАГНОЗ ЗРИВУ НА ШИРИНІ 1024: MSE чи ЦІНА? (С80)

`dims.py` дав на 1024×32 бал 1.49e-01 проти 5.84e-06 у чистого МК — розвал
у ~25000×. Бал = MSE × множник, тож винне АБО одне, АБО друге. Тут вони
розділені, плюс друкується, скільки семплів естиматор собі дозволив.
"""
import sys
SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
sys.path.insert(0, SK)
import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
from bench import load_estimator                        # noqa: E402

B, LAM = 272_000_000_000, 1e11
Z = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')
E = {n: load_estimator(SK + '/' + p)() for n, p in
     [('sub91', 'estimator_sub91.py'), ('МК', 'estimator_mc_plain.py')]}
print(f'{"конфіг":>12s} {"хто":>7s} {"MSE":>12s} {"флопи/B":>9s} '
      f'{"R, мс":>8s} {"множник":>9s} {"замик.част":>11s}')
print('-' * 78)
for width, depth in [(256, 32), (512, 32), (1024, 32)]:
    # ціна ЗАМИКАННЯ: 2 матмулі width³ на кожен із ksig шарів
    ksig = min(31, depth - 1)
    cl = 2.0 * ksig * 2.0 * width ** 3
    for n in range(2):
        key = f'{width}x{depth}x{n}'
        Wn = [fnp.array(Z[f'{key}_w{i}']) for i in range(depth)]
        mlp = MLP(width=width, depth=depth, weights=Wn, seed=n + 1)
        T = Z[key + '_T'].astype(np.float64)
        for nm in E:
            with flops.BudgetContext(flop_budget=B, quiet=True) as c:
                out = E[nm].predict(mlp, B)
            R = c.summary_dict()['residual_wall_time_s']
            mult = max(0.1, c.flops_used / B + LAM * R / B)
            a = np.asarray(out)[-1].astype(np.float64)
            mse = float(((a - T) ** 2).mean())
            print(f'{width}x{depth} #{n:<4d} {nm:>7s} {mse:12.4e} '
                  f'{c.flops_used/B:9.4f} {1000*R:8.2f} {mult:9.4f} '
                  f'{cl/B if nm=="sub91" else 0:11.4f}', flush=True)
