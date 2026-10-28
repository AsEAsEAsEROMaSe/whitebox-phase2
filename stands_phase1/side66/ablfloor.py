"""З ЧОГО ЗРОБЛЕНА ПІДЛОГА: ЗАМИКАННЯ, `λ` ЧИ БЛЕНД? (С81, крок 2)

## ЩО ВЖЕ ВІДОМО
`side66/floor81.py` (48 мереж, крос-ціль): наша MSE ВИХОДИТЬ НА ПЛАТО
    ×16 → 4.2955e-07 · ×32 → 3.8575e-07 · ×64 → 3.2662e-07
тоді як МК падає далі (5.37e-07 → 3.32e-07 → 1.22e-07) і **ОБГАНЯЄ НАС**:
відношення 1.2510× → 0.8612× → **0.3733×**. Підлога `F ≈ 3e-07` реальна.
`side66/pilotfloor.py`: пілот НЕ винен — «без обрізання» ГІРШЕ на обох
множниках (−30.25% при ×1, −12.10% при ×16) ⟹ обрізання корисне, не шкідливе.

## 🔑 ГОЛОВНИЙ ПІДОЗРЮВАНИЙ ТЕПЕР — БЛЕНД
Останній рядок сабміта: `out[-1] = ours + _W_BLEND·((1+_ALPHA)·m_deep − ours)`,
`_W_BLEND = 0.0651`. Тобто 6.5% відповіді береться з ЗАМИКАННЯ `m_deep`, у
якого власне зміщення НЕ ЗАЛЕЖИТЬ ВІД БЮДЖЕТУ. Це підлога за побудовою:
скільки семплів не бери, 6.5% похибки замикання лишається.
Другий підозрюваний — `λ`-родини: це РЕГРЕСІЇ, підігнані на мережах при
множнику ×1. Їхній залишок теж фіксований на мережу.

## ВАРІАНТИ (готові файли з `side66/abl81/`, зроблені `ablwidth.py`)
  🟠 A · повний `sub91`      🟢 B · `λ = 0`
  🟢 C · бленд = 0           🟢 D · `λ=0` і бленд=0      ⛔ E · чистий МК

🔑 ЧИТАННЯ: якщо на ×64 варіант C (без бленда) має ПОМІТНО нижчу MSE за A —
   підлога зроблена з бленда, і її знімають одним числом. Якщо ж усі чотири
   стоять на однаковому плато — підлога в САМОМУ ядрі (замикання + вибірка),
   і тоді її не зняти налаштуванням.

⛔ ПЛОМБА: A при ×16 мусить дати MSE ≈**4.30e-07** (`floor81.py`, 48 мереж;
   тут 24, тож допуск ширший).

📣 ПЕРЕДБАЧЕНО: на ×1 бленд КОРИСНИЙ (+6.68% зміряно `ablwidth.py`), на ×64
   він стане ШКІДЛИВИМ, і C обжене A щонайменше на 10%. Якщо ні — підлога
   не в бленді.

Запуск: python3 side66/ablfloor.py
"""
import glob
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
MULTS = [1, 16, 64]
CAND = [('🟠 A · повний sub91', OUT + '/est_A.py'),
        ('🟢 B · λ = 0', OUT + '/est_B.py'),
        ('🟢 C · бленд = 0', OUT + '/est_C.py'),
        ('🟢 D · λ=0 і бленд=0', OUT + '/est_D.py'),
        ('⛔ E · чистий МК', SK + '/estimator_mc_plain.py')]

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


TAB = {}
for mult in MULTS:
    mm, pr, sc = {}, {}, {}
    for nm, p in CAND:
        mm[nm], pr[nm], sc[nm] = run(p, mult)
    TAB[mult] = mm
    a0, mk = mm['🟠 A · повний sub91'], sc['⛔ E · чистий МК']
    print(f'\nМНОЖНИК ×{mult} · {NETS} мереж · крос-ціль')
    print(f'{"варіант":>22s} {"MSE":>11s} {"ціна":>7s} {"проти МК":>9s} '
          f'{"MSE проти A":>12s}')
    print('-' * 68)
    for nm, _ in CAND:
        print(f'{nm:>22s} {mm[nm]:11.4e} {pr[nm]:7.4f} {mk / sc[nm]:8.4f}× '
              f'{100 * (a0 / mm[nm] - 1):+11.2f}%', flush=True)

print('\n🔑 ЯК ЗМІНЮЄТЬСЯ КОРИСНІСТЬ ВУЗЛА З БЮДЖЕТОМ '
      '(+ = вузол ще корисний, − = вже шкодить)')
print(f'{"вузол вимкнено":>22s} ' + ' '.join(f'{"×" + str(m):>10s}'
                                             for m in MULTS))
print('-' * 60)
for nm, _ in CAND[1:4]:
    row = ' '.join(f'{100 * (TAB[m][nm] / TAB[m]["🟠 A · повний sub91"] - 1):+9.2f}%'
                   for m in MULTS)
    print(f'{nm:>22s} {row}')
print('\n📐 плато нашої MSE: 4.30e-07 (×16) → 3.86e-07 (×32) → 3.27e-07 (×64)')
print('📊 БАЛ 1.954e-07 · лідер 8.0e-09 = MSE 8e-08, нижче за нашу підлогу')
