"""ЧИ ВАРТЕ ЗАМИКАННЯ СВОЄЇ ЦІНИ НА ШИРИНІ 1024 (С80, ціль сесії).

⛔ НИЗЬКИЙ РАНГ `C` ЗАКРИТО (`side66/lowrank_c.py`): щоб похибка зрізу впала
нижче власної похибки замикання (~0.5%), треба `k ≈ ширина/8`, а схема
«діаг + ранг k» лишається ефективною лише поки `k(k+1)/2 ≲ ширина`, тобто
`k ≲ 45`. На 1024 при `k=45` похибка ~0.8% — ГІРША за те, що виправляємо.

🎯 ТОМУ ІНШИЙ ХІД: не здешевлювати замикання, а спитати, чи воно там ПОТРІБНЕ.
На 1024 глибоке замикання їсть 48.95% бюджету, а бал = MSE × ціна ⟹ якщо його
внесок у точність менший за 2×, воно себе НЕ ОКУПАЄ.

ВАРІАНТИ (усі з `_MIN_PAIRS=1152` — оптимум зі свіпу `cliff.py`):
  🟠 sub91 як є        — глибоке замикання 32 шари + бленд `m_deep`
  🟢 мілке (KSIG=8)    — замикання лише 9 шарів, бленд ВИМКНЕНО
  🟢 KSIG=4 · KSIG=2   — ще коротше
  ⛔ чистий МК         — база
🚨 бленд вимкнено СВІДОМО: без `kdeep=depth` величина `m_deep` — це мілке `m`,
   а не прогноз останнього шару; змішувати різні величини не можна.
"""
import sys
SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
SC = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad'
sys.path.insert(0, SK)
import numpy as np                                       # noqa: E402
import flopscope as flops                                # noqa: E402
import flopscope.numpy as fnp                            # noqa: E402
from whestbench import MLP                               # noqa: E402
from bench import load_estimator                         # noqa: E402
from flopscope.errors import BudgetExhaustedError        # noqa: E402

B, LAM, D, NETS, W = 272_000_000_000, 1e11, 32, 6, 1024
Z = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')
CAND = [('⛔ чистий МК', SK + '/estimator_mc_plain.py'),
        ('🔵 2048 (еталон гілки)', SC + '/est_f2048.py'),
        ('🟢 2048, пілот 32', SC + '/est_f2048p32.py'),
        ('🟢 2048, пілот 16', SC + '/est_f2048p16.py'),
        ('🔵 1792', SC + '/est_f1792.py'),
        ('🔵 1536', SC + '/est_f1536.py')]
print(__doc__, flush=True)
print(f'{"варіант":>26s} {"MSE":>11s} {"ціна C/B":>9s} {"БАЛ":>11s} '
      f'{"проти МК":>9s} {"бюдж.":>6s}')
print('-' * 80)
res = {}
for nm, path in CAND:
    E = load_estimator(path)()
    ms, cs, ov = [], [], 0
    for n in range(NETS):
        key = f'{W}x{D}x{n}'
        Wn = [fnp.array(Z[f'{key}_w{i}']) for i in range(D)]
        T = Z[key + '_T'].astype(np.float64)
        blown = False
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            try:
                o = E.predict(MLP(width=W, depth=D, weights=Wn, seed=n + 1), B)
            except BudgetExhaustedError:
                blown, o = True, np.zeros((D, W))
        R = c.summary_dict()['residual_wall_time_s']
        C = c.flops_used + LAM * R
        a = np.asarray(o)[-1].astype(np.float64)
        if blown or C > B:
            a = np.zeros_like(a); ov += 1
        ms.append(float(((a - T) ** 2).mean()))
        cs.append(1.0 if (blown or C > B) else max(0.1, C / B))
    sc = float(np.mean([x * y for x, y in zip(ms, cs)]))
    res[nm] = sc
    g = res['⛔ чистий МК'] / sc
    print(f'{nm:>26s} {np.mean(ms):11.4e} {np.mean(cs):9.4f} {sc:11.4e} '
          f'{g:8.4f}× {ov:4d}/{NETS}', flush=True)
mk = res['⛔ чистий МК']
print('\n🔑 ВИГРАШ ПРОТИ ЧИСТОГО МК на ширині 1024:')
for nm, _ in CAND:
    print(f'   {nm:>26s} {mk/res[nm]:8.4f}×')
print('📊 БАЛ 1.954e-07')
