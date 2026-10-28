"""СКІЛЬКИ ПАР ТРЕБА НА ШИРИНІ 1024 — І ЧИ ВЗАГАЛІ ВИСТАЧАЄ БЮДЖЕТУ (С80).

Питання юзера: «який скор якщо 1024 і семплів 1024 і 2048?»
Причина зриву (`finding-submission-only-works-in-a-narrow-width-band`):
пілот з'їдає 77% дозволеної частки, лишається `_MIN_PAIRS=128` пар на 1024
вимірів ⟹ коваріація для вибілювання ВИРОДЖЕНА. Тут піднімаємо `_MIN_PAIRS`.

🚨 ПРАВИЛО ГРЕЙДЕРА (docs/concepts/scoring-model.md): якщо бюджет ПЕРЕВИЩЕНО,
   усі прогнози замінюються НУЛЯМИ і знижки немає (множник = 1.0). Тут це
   відтворено явно, інакше число було б неправдиво добрим.
"""
import sys
SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
SC = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad'
sys.path.insert(0, SK)
import numpy as np                                      # noqa: E402
import flopscope as flops                               # noqa: E402
import flopscope.numpy as fnp                           # noqa: E402
from whestbench import MLP                              # noqa: E402
from bench import load_estimator                        # noqa: E402
from flopscope.errors import BudgetExhaustedError        # noqa: E402

B = 272_000_000_000
Z = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')
CAND = [('sub91 як є (128)', SK + '/estimator_sub91.py'),
        ('пар 512', SC + '/est_p512.py'),
        ('пар 1024', SC + '/est_p1024.py'),
        ('пар 2048', SC + '/est_p2048.py'),
        ('⛔ чистий МК', SK + '/estimator_mc_plain.py')]
W, D, NETS = 1024, 32, 3
print(f'ширина {W}, глибина {D}, {NETS} мереж\n')
print(f'{"варіант":>18s} {"MSE":>12s} {"флопи/B":>9s} {"множник":>9s} '
      f'{"БАЛ":>12s} {"проти МК":>10s} {"понад бюджет":>13s}')
print('-' * 92)
res = {}
for nm, path in CAND:
    E = load_estimator(path)()
    sc, ms, ov, fr = [], [], 0, []
    for n in range(NETS):
        key = f'{W}x{D}x{n}'
        Wn = [fnp.array(Z[f'{key}_w{i}']) for i in range(D)]
        T = Z[key + '_T'].astype(np.float64)
        blown = False
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            try:
                out = E.predict(MLP(width=W, depth=D, weights=Wn,
                                    seed=n + 1), B)
            except BudgetExhaustedError:   # 🚨 те саме, що «понад бюджет»
                blown, out = True, np.zeros((D, W))
        R = c.summary_dict()['residual_wall_time_s']
        C = c.flops_used + 1e11 * R
        a = np.asarray(out)[-1].astype(np.float64)
        if blown or C > B:                          # 🚨 правило грейдера
            a = np.zeros_like(a); mult = 1.0; ov += 1
        else:
            mult = max(0.1, C / B)
        ms.append(float(((a - T) ** 2).mean()))
        sc.append(ms[-1] * mult)
        fr.append(C / B)
    res[nm] = float(np.mean(sc))
    g = res['⛔ чистий МК'] / res[nm] if '⛔ чистий МК' in res else float('nan')
    print(f'{nm:>18s} {np.mean(ms):12.4e} {np.mean(fr):9.4f} '
          f'{np.mean(sc)/max(np.mean(ms),1e-300):9.4f} {np.mean(sc):12.4e} '
          f'{g:9.4f}× {ov:9d}/{NETS}', flush=True)
mk = res['⛔ чистий МК']
print('\n🔑 виграш проти чистого МК (>1 = краще):')
for nm, _ in CAND:
    print(f'   {nm:>18s} {mk/res[nm]:8.4f}×')
