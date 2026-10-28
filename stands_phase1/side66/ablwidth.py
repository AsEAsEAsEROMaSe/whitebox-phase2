"""ЧОМУ НА 1024 МИ 2.22×, А НА 256 — 3.24× ПРИ ОДНАКОВІЙ КАЛІБРОВЦІ? (С81)

`side66/wbudget.py` зняв артефакт бюджету: при утилізаційно-інваріантному
множнику ×16 `sub91` на ширині 1024 працює і дає **2.2167×**. Але на 256 він
дає **3.2381×**. Обидва режими купують ОДНАКОВЕ число семплів МК ⟹ різниця
НЕ у грошах. Це справжня діра, і її треба локалізувати.

## 🔑 ГОЛОВНА ПІДОЗРА
Чотири родини `_LAM_S/_LAM_A/_LAM_Q/_LAM_G` (по 8 чисел) фітувались LOO на
2454 мережах ШИРИНИ 256. Вони стоять при різницях «МК мінус замикання»
(`xm - m_cl`, `sh - s_cl`, …). Масштаб цих різниць залежить від ширини, бо
власна похибка замикання — скінченнорозмірний ефект `~1/ширина`. На 1024
вона вчетверо менша, а коефіцієнти лишились ті самі ⟹ поправка може бути
ВЧЕТВЕРО ЗАВЕЛИКОЮ і псувати замість лікувати.

## АБЛЯЦІЇ (кожна на ОБОХ ширинах, щоб бачити ЗМІНУ ВНЕСКУ, а не рівень)
  🟠 A · повний `sub91`                       — база
  🟢 B · `λ = 0` (усі 4 родини вимкнено)      — чи заважають поправки
  🟢 C · бленд `_W_BLEND = 0`                 — чи заважає домішка `m_deep`
  🟢 D · `λ = 0` І бленд 0                    — голе «замикання + МК»
  ⛔ E · чистий МК                            — база відношення

🔑 ЧИТАННЯ: якщо на 256 `B` помітно ГІРША за `A`, а на 1024 — КРАЩА або рівна,
   то поправки міскаліброзані шириною, і це лагодиться перефітом.
   Якщо ж `B` гірша на обох — діра НЕ в `λ`, і шукати треба в замиканні.

⛔ ПЛОМБА: `A` на 256 при множнику 1 мусить дати **3.2381×** (`wbudget.py`).

📣 ПЕРЕДБАЧЕНО: на 256 вимкнення `λ` коштує 5…15%; на 1024 — коштує МЕНШЕ
   або дає плюс. Ставлю на те, що `λ` пояснює МЕНШУ частину діри, ніж
   здається, бо різниця 3.24→2.22 завелика (−31%) для поправок, які на 256
   коштують одиниці відсотків.

Запуск: python3 side66/ablwidth.py
"""
import sys

SK = '/mnt/win_d/llm/White-Box/whest-starterkit'
OUT = '/mnt/win_d/llm/White-Box/side66/abl81'
sys.path.insert(0, SK)
import os                                                # noqa: E402
import numpy as np                                       # noqa: E402
import flopscope as flops                                # noqa: E402
import flopscope.numpy as fnp                            # noqa: E402
from whestbench import MLP                               # noqa: E402
from bench import load_estimator                         # noqa: E402
from flopscope.errors import BudgetExhaustedError        # noqa: E402

flops.configure(symmetry_warnings=False)
B0, LAM, D, NETS = 272_000_000_000, 1e11, 32, 6
Z = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')
SRC = open(SK + '/estimator_sub91.py').read()
os.makedirs(OUT, exist_ok=True)

ZERO = '_LAM_S = _LAM_A = _LAM_Q = _LAM_G = (0.0,) * 16   # 🔪 абляція\n\n\ndef _phi'
NOBLEND = '_W_BLEND = 0.0'


def variant(tag, zero_lam, no_blend):
    s = SRC
    if zero_lam:
        s = s.replace('def _phi', ZERO, 1)
    if no_blend:
        s = s.replace('_W_BLEND = 0.0651', NOBLEND, 1)
    p = f'{OUT}/est_{tag}.py'
    open(p, 'w').write(s)
    return p


CAND = [('🟠 A · повний sub91', variant('A', False, False)),
        ('🟢 B · λ = 0', variant('B', True, False)),
        ('🟢 C · бленд = 0', variant('C', False, True)),
        ('🟢 D · λ=0 і бленд=0', variant('D', True, True)),
        ('⛔ E · чистий МК', SK + '/estimator_mc_plain.py')]
print(__doc__.split('Запуск')[0], flush=True)


def run(path, width, mult):
    B = B0 * mult
    E = load_estimator(path)()
    ms, cs = [], []
    for n in range(NETS):
        key = f'{width}x{D}x{n}'
        Wn = [fnp.array(Z[f'{key}_w{i}']) for i in range(D)]
        T = Z[key + '_T'].astype(np.float64)
        blown = False
        with flops.BudgetContext(flop_budget=B, quiet=True) as c:
            try:
                o = E.predict(MLP(width=width, depth=D, weights=Wn,
                                  seed=n + 1), B)
            except BudgetExhaustedError:
                blown, o = True, np.zeros((D, width))
        C = c.flops_used + LAM * c.summary_dict()['residual_wall_time_s']
        a = np.asarray(o)[-1].astype(np.float64)
        if blown or C > B:
            a = np.zeros_like(a)
        ms.append(float(((a - T) ** 2).mean()))
        cs.append(1.0 if (blown or C > B) else max(0.1, C / B))
    return float(np.mean([x * y for x, y in zip(ms, cs)]))


REG = {}
for width, mult in ((256, 1), (1024, 16)):
    print(f'\nШИРИНА {width} · множник ×{mult} · {NETS} мереж')
    print(f'{"варіант":>24s} {"БАЛ":>11s} {"проти МК":>9s} {"ΔA":>9s}')
    print('-' * 58)
    sc = {nm: run(p, width, mult) for nm, p in CAND}
    mk, a = sc['⛔ E · чистий МК'], sc['🟠 A · повний sub91']
    REG[width] = {}
    for nm, _ in CAND:
        REG[width][nm] = mk / sc[nm]
        print(f'{nm:>24s} {sc[nm]:11.4e} {mk / sc[nm]:8.4f}× '
              f'{100 * (a / sc[nm] - 1):+8.2f}%', flush=True)

print('\n🔑 ВНЕСОК КОЖНОГО ВУЗЛА (наскільки БЕЗ нього гірше; '
      '+ = вузол корисний)')
print(f'{"вузол":>24s} {"на 256":>10s} {"на 1024":>10s}')
print('-' * 48)
for nm, _ in CAND[1:4]:
    v256 = 100 * (REG[256]['🟠 A · повний sub91'] / REG[256][nm] - 1)
    v1024 = 100 * (REG[1024]['🟠 A · повний sub91'] / REG[1024][nm] - 1)
    print(f'{nm:>24s} {v256:+9.2f}% {v1024:+9.2f}%')
print('\n📐 діра, яку пояснюємо: 3.2381× (256) → 2.2167× (1024) = −31.5%')
print('📊 БАЛ 1.954e-07')
