"""ОБРИВ МІЖ 512 І 1024: де саме і що там цікавого (С80, запит юзера).

Аналітика (`пар/ширина`): 512→1.63 · 640→0.72 · 768→0.34 · 896→0.15 · 1024→0.12
⟹ обрив між 512 і 640. Тут два зрізи:

  A. САБМІТ ЯК Є на ширинах 512…1024 — де реально ламається;
  B. Ширина 1024, СВІП `пар/ширина` від 0.25 до 1.5 — форма переходу і де
     оптимум БАЛУ.

🚨 У зрізі B бюджет навмисно послаблено (4·B), щоб крива не обривалась
   винятком; але БАЛ рахується з РЕАЛЬНИМ `B` ⟹ видно і точність, і ціну.
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

B, LAM, D, NETS = 272_000_000_000, 1e11, 32, 3
Z1 = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims.npz')
Z2 = np.load('/mnt/win_d/llm/White-Box/side66/logs/dims2.npz')


def get(key, i):
    return (Z1 if f'{key}_w0' in Z1 else Z2)[f'{key}_w{i}']


def tru(key):
    return (Z1 if key + '_T' in Z1 else Z2)[key + '_T'].astype(np.float64)


def run(path, width, n, bud):
    E = load_estimator(path)()
    key = f'{width}x{D}x{n}'
    Wn = [fnp.array(get(key, i)) for i in range(D)]
    T = tru(key)
    blown = False
    with flops.BudgetContext(flop_budget=bud, quiet=True) as c:
        try:
            o = E.predict(MLP(width=width, depth=D, weights=Wn, seed=n + 1), bud)
        except BudgetExhaustedError:
            blown, o = True, np.zeros((D, width))
    R = c.summary_dict()['residual_wall_time_s']
    C = c.flops_used + LAM * R
    a = np.asarray(o)[-1].astype(np.float64)
    mse = float(((a - T) ** 2).mean())
    return mse, C / B, blown


print('A. САБМІТ ЯК Є — де ламається\n')
print(f'{"ширина":>7s} {"наш MSE":>11s} {"МК MSE":>11s} {"наша ціна":>10s} '
      f'{"БАЛ наш":>11s} {"БАЛ МК":>11s} {"виграш":>8s}')
print('-' * 76)
for w in (512, 640, 768, 896, 1024):
    sa, sm = [], []
    for n in range(NETS):
        m1, c1, b1 = run(SK + '/estimator_sub91.py', w, n, B)
        m2, c2, b2 = run(SK + '/estimator_mc_plain.py', w, n, B)
        sa.append((m1, 1.0 if b1 else max(0.1, c1)))
        sm.append((m2, 1.0 if b2 else max(0.1, c2)))
    A = np.array([x * y for x, y in sa]); M = np.array([x * y for x, y in sm])
    print(f'{w:7d} {np.mean([x for x,_ in sa]):11.4e} '
          f'{np.mean([x for x,_ in sm]):11.4e} '
          f'{np.mean([y for _,y in sa]):10.4f} {A.mean():11.4e} '
          f'{M.mean():11.4e} {M.mean()/A.mean():7.4f}×', flush=True)

print('\n\nB. ШИРИНА 1024 — СВІП `пар/ширина` (бюджет послаблено, бал з реальним B)\n')
print(f'{"пар":>6s} {"пар/ширина":>11s} {"MSE":>11s} {"ціна C/B":>9s} '
      f'{"БАЛ":>11s} {"проти МК":>9s}')
print('-' * 62)
mk = np.mean([run(SK + '/estimator_mc_plain.py', 1024, n, B)[0]
              for n in range(NETS)]) * 0.1
for pth, npr in [(SC + '/est_p256.py', 256), (SC + '/est_p384.py', 384),
                 (SC + '/est_p512.py', 512), (SC + '/est_p640.py', 640),
                 (SC + '/est_p768.py', 768), (SC + '/est_p896.py', 896),
                 (SC + '/est_p1024.py', 1024), (SC + '/est_p1152.py', 1152),
                 (SC + '/est_p1536.py', 1536)]:
    ms, cs = [], []
    for n in range(NETS):
        m, c, _ = run(pth, 1024, n, 4 * B)
        ms.append(m); cs.append(c)
    sc = np.mean(ms) * max(0.1, np.mean(cs))
    print(f'{npr:6d} {npr/1024:11.3f} {np.mean(ms):11.4e} '
          f'{np.mean(cs):9.4f} {sc:11.4e} {mk/sc:8.4f}×', flush=True)
print('\n📊 БАЛ 1.954e-07 — карта стійкості')
