"""З ЧОГО ЗРОБЛЕНА НАША ВЛАСНА ПІДЛОГА: ЧИ ВИННЕ ОБРІЗАННЯ ПІЛОТОМ? (С81)

## ЗВІДКИ ПИТАННЯ
`side66/floor81.py` показав, що перевага над МК ТАНЕ з бюджетом
(3.97× → 2.88× → 1.96× → 1.35× → 1.25× при ×1…×16) ⟹ у нас є доданок, який
семпли НЕ ЛІКУЮТЬ. Треба знайти, з чого він зроблений.

## 🔑 ПЕРШИЙ ПІДОЗРЮВАНИЙ — ПІЛОТ, І ВІН ІМЕННИЙ
`_PILOT_PAIRS = 128` ЗАШИТО і з бюджетом НЕ РОСТЕ. Нейрон оголошується
мертвим, якщо жодного разу не відкрився на 2·128 = 256 семплах. Нейрон, що
відкривається з імовірністю `p`, помилково вбивається з імовірністю
`(1−p)²⁵⁶` — для `p = 0.01` це **7.7%**. Ця похибка НЕ ЗАЛЕЖИТЬ ВІД БЮДЖЕТУ,
тобто поводиться рівно як підлога.
📐 Уже зміряно (`side66/prune.py`): пілот ріже **18.64%** нейронів при
істинно мертвих **9.48%** — тобто вдвічі більше, ніж треба.

## ВАРІАНТИ (усі на множниках ×1 і ×16, 24 мережі, ✅крос-ціль)
  🟠 A · база (`_PILOT_PAIRS=128`)
  🟢 B · пілот 512
  🟢 C · пілот 2048
  🟢 D · **БЕЗ ОБРІЗАННЯ ВЗАГАЛІ** (`alive` завжди True) — дорожче на семпл,
        зате похибки обрізання рівно нуль
  ⛔ E · чистий МК — база відношення

🔑 ЧИТАННЯ: якщо на ×16 варіанти B/C/D помітно кращі за A, підлога значною
   мірою зроблена з обрізання, і її можна знести. Якщо ж усі однакові —
   підлога сидить у ЗАМИКАННІ та `λ`, і пілот ні при чому.

⛔ ПЛОМБА: A при ×1 мусить дати ≈**3.97×** (`floor81.py`, крос-ціль).

📣 ПЕРЕДБАЧЕНО: на ×1 різниці майже не буде (там панує дисперсія, а обрізання
   ще й економить гроші). На ×16 очікую, що D дасть 5…20% кращу MSE за A.
   Якщо D гірша за A і на ×16 — обрізання не тільки безкоштовне, а й
   КОРИСНЕ, і підлогу треба шукати в замиканні.

Запуск: python3 side66/pilotfloor.py
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
MULTS = [1, 16]
SRC = open(SK + '/estimator_sub91.py').read()
os.makedirs(OUT, exist_ok=True)
ALIVE = 'alive.append(fnp.max(h, axis=0) >= 0.0)'
NOCUT = 'alive.append(fnp.max(h, axis=0) >= -1e30)   # 🔪 БЕЗ обрізання'


def variant(tag, pilot=None, nocut=False):
    s = SRC
    if pilot is not None:
        s = s.replace('_PILOT_PAIRS = 128', f'_PILOT_PAIRS = {pilot}', 1)
    if nocut:
        s = s.replace(ALIVE, NOCUT, 1)
    p = f'{OUT}/pf_{tag}.py'
    open(p, 'w').write(s)
    return p


CAND = [('🟠 A · пілот 128 (база)', variant('A')),
        ('🟢 B · пілот 512', variant('B', pilot=512)),
        ('🟢 C · пілот 2048', variant('C', pilot=2048)),
        ('🟢 D · БЕЗ обрізання', variant('D', nocut=True)),
        ('⛔ E · чистий МК', SK + '/estimator_mc_plain.py')]

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
FS = [q for q in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(q)['seed']) in TRUTH2][:NETS]
NET = [(int(np.load(f)['seed']),) for f in FS]
NET = [(s, int(np.where(SD == s)[0][0])) for (s,) in NET]
print(__doc__.split('Запуск')[0], flush=True)


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
    print(f'\nМНОЖНИК ×{mult} · {NETS} мереж · крос-ціль')
    print(f'{"варіант":>24s} {"MSE":>11s} {"ціна":>7s} {"БАЛ":>11s} '
          f'{"проти МК":>9s} {"MSE проти A":>12s}')
    print('-' * 80)
    sc, mm, pr = {}, {}, {}
    for nm, p in CAND:
        mm[nm], pr[nm], sc[nm] = run(p, mult)
        print(f'   … {nm} готово', flush=True)
    a0, mk = mm['🟠 A · пілот 128 (база)'], sc['⛔ E · чистий МК']
    for nm, _ in CAND:
        print(f'{nm:>24s} {mm[nm]:11.4e} {pr[nm]:7.4f} {sc[nm]:11.4e} '
              f'{mk / sc[nm]:8.4f}× {100 * (a0 / mm[nm] - 1):+11.2f}%')

print('\n🔑 «MSE проти A» додатна = варіант ТОЧНІШИЙ за базу '
      '(ціна врахована окремо, у колонці БАЛ)')
print('📐 підлога, яку шукаємо: перевага над МК тане 3.97× (×1) → 1.25× (×16)')
print('📊 БАЛ 1.954e-07')
