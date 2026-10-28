"""ЧИ ПРАЦЮЄ САБМІТ НА ІНШИХ РОЗМІРАХ (ширина·глибина)? (С80, питання юзера)

Юзер: «чи спрацює наша формула на мережі 8 шарів або на 1024 ширини? бо в
фінальному естіматорі будуть юзати й інші розмірності».

## 🚨 ЧОМУ ЦЕ РЕАЛЬНИЙ РИЗИК, А НЕ ПАРАНОЯ
`whestbench.create_dataset(width=..., depth=...)` — це ВІЛЬНІ параметри, а не
константи. `estimator_sub91.py` читає `mlp.width` і `len(mlp.weights)`
динамічно, тож ПАДІННЯ не буде. Але всі сталі (`_TARGET_UTIL`, `_ALPHA`,
`_W_BLEND`, `_KSIG`, `_GATE_N=1024`, вектори `λ`) підбирались РІВНО на 256×32.
🔑 Питання не «чи запуститься», а **«чи лишиться кращим за чистий МК»**.

## ЩО МІРЯЄМО
Сітка `(ширина, глибина)`. На кожній: `NETS` свіжих мереж (He, як
`whestbench.generation.sample_mlp`: `N(0, 2/ширина)`, БЕЗ зсувів), істина —
велике МК з антитетикою у fp64, і два естиматори З ФАЙЛІВ:
  * 🟢 `estimator_sub91.py` — кандидат на подачу;
  * ⛔ `estimator_mc_plain.py` — чистий МК, базова лінія.
Рахуємо `бал = MSE × max(0.1, C/B)` для обох ⟹ **виграш = бал_МК / бал_наш**.

⛔ ПЛОМБА: на 256×32 виграш мусить відтворити відоме (~1.2–1.4× проти plain MC).
   Якщо ні — стенд бреше, і решта рядків не читається.
📊 Друкуємо ще МЕДІАНУ І РОЗКИД виграшу ПО МЕРЕЖАХ — це друге питання юзера
   («чому на одних мережах e-8, а на інших майже МК»).

📣 ПЕРЕДБАЧЕНО: виграш ПАДАЄ при відході від 256×32, і найсильніше — по
   ГЛИБИНІ (бо `_KSIG` і `λ` пошарові), слабше — по ширині (там більшість
   формул безрозмірні). На малій глибині (8) може стати <1.0×, тобто ШКОДИТИ.

Запуск: /usr/bin/python3 side66/dims.py [NETS]
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
GRID = [(256, 32), (256, 8), (256, 16), (256, 64),
        (128, 32), (512, 32), (1024, 32), (64, 32)]
CAND = [('🟢 sub91', 'estimator_sub91.py'),
        ('⛔ чистий МК', 'estimator_mc_plain.py')]
# 🚨 ваги й істину готує КРОК 1 (`side66/dims_truth.py`) на GPU — тут лише
# читаємо, бо system python3 має flopscope, але не torch, і навпаки.
Z = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')


def main(nets=6):
    print(__doc__.split('Запуск')[0].split('## ЩО')[0], flush=True)
    est = {nm: load_estimator(SK + '/' + p)() for nm, p in CAND}
    print(f'{"ширина×глибина":>16s} {"🟢sub91 бал":>13s} {"⛔МК бал":>12s} '
          f'{"ВИГРАШ":>9s} {"медіана":>9s} {"гірша мережа":>13s} '
          f'{"частка>1":>9s}')
    print('-' * 92)
    for width, depth in GRID:
        sc = {nm: [] for nm, _ in CAND}
        for n in range(nets):
            key = f'{width}x{depth}x{n}'
            Wn = [fnp.array(Z[f'{key}_w{i}']) for i in range(depth)]
            mlp = MLP(width=width, depth=depth, weights=Wn, seed=n + 1)
            T = Z[key + '_T'].astype(np.float64)
            for nm, _ in CAND:
                with flops.BudgetContext(flop_budget=B, quiet=True) as c:
                    out = est[nm].predict(mlp, B)
                R = c.summary_dict()['residual_wall_time_s']
                mult = max(0.1, c.flops_used / B + LAM * R / B)
                a = np.asarray(out)[-1].astype(np.float64)
                sc[nm].append(float(((a - T) ** 2).mean()) * mult)
        A = np.array(sc['🟢 sub91']); M = np.array(sc['⛔ чистий МК'])
        g = M / np.maximum(A, 1e-300)
        print(f'{width:7d}×{depth:<8d} {A.mean():13.4e} {M.mean():12.4e} '
              f'{M.sum()/A.sum():8.4f}× {np.median(g):8.4f}× '
              f'{g.min():12.4f}× {100*np.mean(g > 1):8.0f}%', flush=True)
    print('\n⛔ПЛОМБА: рядок 256×32 мусить відтворити відоме проти plain MC')
    print('📊 БАЛ 1.954e-07 — це ПЕРЕВІРКА СТІЙКОСТІ, приросту не дає')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
