"""`_KSIG` — ТЕЖ КРИВА, І ЇЇ ОБІРВАЛИ ПОСЕРЕДИНІ (С81)

## 🔑 ЗНАХІДКА В САМОМУ КОДІ СAБМІТА
Родина `_LAM_Q` (поправка на `E[x²]`) по шарах 1…8:
    0.0011 · 0.0000 · 0.0096 · 0.0034 · 0.0190 · 0.0251 · 0.0341 · **0.0415**
Вона РОСТЕ МОНОТОННО і на восьмому шарі НЕ САТУРУЄ. `_KSIG = 8` — це не
оптимум, це межа, до якої колись фітували. Тобто коефіцієнт сам є КРИВОЮ
в індексі шару, і ми обірвали її там, де вона ще піднімалась.

## ЩО МІРЯЄМО
Свіп `_KSIG` вниз (де `λ` є) і вгору (де `λ` треба продовжити):
  * `_KSIG ∈ {4, 6, 8}` — рідними, підігнаними числами;
  * `_KSIG ∈ {10, 12, 16}` × ТРИ правила продовження:
      ⛔ZERO  — нулі (⛔контроль: додає лише ЦІНУ, точності не додає);
      🟢HOLD  — тримати останнє значення;
      🟢LIN   — лінійна екстраполяція по останніх 4 точках кожної родини.

Множники ×1 (РОБОЧА ТОЧКА, тут одразу видно вплив на бал) і ×8 (щоб бачити,
чи оптимальний `_KSIG` їде з бюджетом — теж крива).
24 мережі, ✅крос-ціль.

⛔ ПЛОМБА: `_KSIG=8` рідний при ×1 мусить дати MSE ≈**2.1349e-06**
   (`ablfloor.py`, ті самі 24 мережі).
⛔ КОНТРОЛЬ ZERO: мусить бути НЕ КРАЩИМ за `_KSIG=8` — він додає роботу без
   інформації. Якщо ZERO раптом ВИГРАЄ, стенд бреше.

📣 ПЕРЕДБАЧЕНО: LIN на `_KSIG=10…12` дасть 2…6% кращу MSE при ×1 (бо `_LAM_Q`
   явно недобрана), далі почне псувати, бо екстраполяція піде в шум.
   Свіп ВНИЗ покаже монотонний спад ⟹ підтвердить, що глибші шари корисні.
   Якщо ж усе плоске в межах 1% — `_KSIG=8` справді оптимум, і ця ручка мертва.

Запуск: python3 side66/ksig81.py
"""
import glob
import os
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
OUT = '/mnt/win_d/llm/White-Box/side66/abl81'
sys.path.insert(0, SK)
import numpy as np                                       # noqa: E402
import flopscope as flops                                # noqa: E402
import flopscope.numpy as fnp                            # noqa: E402
from whestbench import MLP                               # noqa: E402
from bench import load_estimator                         # noqa: E402
from flopscope.errors import BudgetExhaustedError        # noqa: E402

flops.configure(symmetry_warnings=False)
B0, LAM, D, W, NETS = 272_000_000_000, 1e11, 32, 256, 24
MULTS = [1, 8]
SRC = open(SK + '/estimator_sub91.py').read()
os.makedirs(OUT, exist_ok=True)
HEAD, TAIL = SRC.split('_KSIG = 8', 1)
BODY, REST = TAIL.split('# Абрамовіц-Стіган', 1)

BASE = {
    '_LAM_S': [0.0017899, 0.0047168, 0.0071225, 0.0040152,
               0.0088285, 0.0007922, 0.0015044, 0.0001062],
    '_LAM_A': [-0.0067395, -0.0091340, -0.0092425, -0.0049866,
               -0.0025822, -0.0030778, 0.0006822, 0.0068061],
    '_LAM_Q': [0.0010829, 0.0000248, 0.0095619, 0.0034287,
               0.0189783, 0.0250559, 0.0340802, 0.0415433],
    '_LAM_G': [-0.0047241, 0.0103696, 0.0100503, 0.0273080,
               -0.0146129, -0.0100612, 0.0167223, 0.0168109],
}


def extend(v, k, rule):
    """Продовжити родину `v` до довжини `k` за правилом `rule`."""
    v = list(v)
    if k <= len(v):
        return v[:k]
    if rule == 'ZERO':
        return v + [0.0] * (k - len(v))
    if rule == 'HOLD':
        return v + [v[-1]] * (k - len(v))
    x = np.arange(len(v) - 4, len(v), dtype=float)      # LIN: по 4 останніх
    a, b = np.polyfit(x, np.array(v[-4:]), 1)
    return v + [float(a * (len(v) + t) + b) for t in range(k - len(v))]


def variant(k, rule):
    body = f'_KSIG = {k}\n'
    for nm, v in BASE.items():
        e = extend(v, k, rule)
        body += (f'{nm} = (' + ', '.join(f'{q:.10g}' for q in e) + ',)\n')
    p = f'{OUT}/ks_{k}_{rule}.py'
    open(p, 'w').write(HEAD + body + '\n# Абрамовіц-Стіган' + REST)
    return p


CAND = [(f'🟠 KSIG={k} рідні', variant(k, 'ZERO')) for k in (4, 6, 8)]
for k in (10, 12, 16):
    for r, tag in (('ZERO', '⛔'), ('HOLD', '🟢'), ('LIN', '🟢')):
        CAND.append((f'{tag} KSIG={k} {r}', variant(k, r)))
CAND.append(('⛔ чистий МК', SK + '/estimator_mc_plain.py'))

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
FS = [q for q in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(q)['seed']) in TRUTH2][:NETS]
NET = [(s, int(np.where(SD == s)[0][0]))
       for s in (int(np.load(f)['seed']) for f in FS)]
print(__doc__.split('Запуск')[0], flush=True)
print('📐 LIN-продовження _LAM_Q до 16 шарів: '
      + ' '.join(f'{q:.4f}' for q in extend(BASE['_LAM_Q'], 16, 'LIN')[8:]),
      flush=True)


def run(path, mult):
    B = B0 * mult
    E = load_estimator(path)()
    ms, cs = [], []
    for s0, j in NET:
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
        ms.append(float(np.mean((a - T1) * (a - T2))))
        cs.append(1.0 if (blown or C > B) else max(0.1, C / B))
    return (float(np.mean(ms)), float(np.mean(cs)),
            float(np.mean([x * y for x, y in zip(ms, cs)])))


for mult in MULTS:
    mm, pr, sc = {}, {}, {}
    for nm, p in CAND:
        mm[nm], pr[nm], sc[nm] = run(p, mult)
        print(f'   … {nm}', flush=True)
    a0 = mm['🟠 KSIG=8 рідні']
    mk = sc['⛔ чистий МК']
    print(f'\nМНОЖНИК ×{mult} · {NETS} мереж · крос-ціль')
    print(f'{"варіант":>22s} {"MSE":>11s} {"ціна":>7s} {"БАЛ":>11s} '
          f'{"проти МК":>9s} {"MSE проти KSIG=8":>17s}')
    print('-' * 84)
    for nm, _ in CAND:
        print(f'{nm:>22s} {mm[nm]:11.4e} {pr[nm]:7.4f} {sc[nm]:11.4e} '
              f'{mk / sc[nm]:8.4f}× {100 * (a0 / mm[nm] - 1):+16.2f}%')

print('\n🔑 «MSE проти KSIG=8» додатна ⟹ варіант ТОЧНІШИЙ; колонка БАЛ '
      'уже враховує ціну')
print('📊 БАЛ 1.954e-07 · робоча точка ×1 ⟹ виграш тут іде ПРЯМО в бал')
