"""КОЕФІЦІЄНТИ — НЕ ЧИСЛА, А КРИВІ: ЧИ ВИЖИВЕ ПІДЛОГА ПІСЛЯ ПЕРЕПІДГОНКИ? (С81)

## 🚨 ЗАУВАЖЕННЯ ЮЗЕРА, ЯКЕ Б'Є ПО МОЄМУ ВИСНОВКУ
«а хіба не треба підігнати · наші коефіцієнти під потрібну розмірність ·
універсалу скоріш за все немає · всі змінні як криві, ідуть своїм шляхом ·
але в потрібній точці наша задача зв'язати все»

Він правий, і це псує вимір `side66/floor81.py`. Там я показав, що наша MSE
виходить на плато `F ≈ 3e-07` і МК нас обганяє при ×32. Але ВСІ константи
сабміта (`_W_BLEND=0.0651`, `_ALPHA=−0.00751`, чотири родини `λ`) підігнані
на множнику ×1. На ×64 вони просто НЕ ТІ. Тобто я міг зміряти не підлогу
методу, а **замерзлі константи**.

🔑 Правильне питання: скільки підлоги лишиться, якщо в КОЖНІЙ точці взяти
   ОПТИМАЛЬНІ коефіцієнти? І як виглядає КРИВА кожного коефіцієнта?

## 🧮 ЯК ЗНЯТИ ВЕСЬ СКАН МАЙЖЕ ЗАДАРМА
Останній рядок сабміта лінійний за блендом:
    `out[-1] = ours + w·((1+α)·m_deep − ours)`
Отже досить ДВОХ прогонів на точку — `ours` (бленд вимкнено) і `m_deep`
(чисте замикання) — і далі весь двовимірний скан `(w, α)` рахується офлайн
без жодного флопа. Для `λ` так не вийде (поправки входять у рекурсію), тому
глобальний масштаб `g` беремо прогонами: `g ∈ {0, 0.5, 1.0, 1.5}`.

## ⚠️ ОРАКУЛ ПРОТИ ЧЕСНОГО ЧИСЛА
Підбирати `(g,w,α)` і міряти на ТИХ САМИХ мережах — це оракул, він завищує
([[feedback-oracle-sharing-the-draw-inflates]]). Тому 24 мережі ділимо
навпіл: підбір на одній половині, вимір на іншій, і навпаки. Друкуємо ОБА
числа — оракульне (стеля) і чесне (те, що реально буде).

⛔ ПЛОМБА: при `g=1, w=0.0651, α=−0.00751` і множнику ×1 офлайн-формула
   мусить відтворити MSE повного `sub91` = **2.1349e-06** (`ablfloor.py`).

📣 ПЕРЕДБАЧЕНО: оптимальний `w` ПАДАЄ з бюджетом (бленд тягне до замикання,
   чиє зміщення не лікується) і при ×64 сяде майже в нуль; оптимальний `g`
   теж поповзе. Але підлога ВИЖИВЕ — впаде з 3.0e-07 хіба до 2…2.5e-07,
   бо головне зміщення сидить у САМОМУ замиканні, а не в тому, з якою вагою
   ми його домішуємо. Якщо ж підлога впаде нижче 1e-07 — я помилявся,
   і «плато» було артефактом замерзлих констант.

Запуск: python3 side66/curvefit81.py
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
MULTS = [1, 8, 64]
GS = [0.0, 0.5, 1.0, 1.5]
WGRID = np.arange(-0.10, 0.4001, 0.005)
AGRID = np.arange(-0.04, 0.0401, 0.005)
SRC = open(SK + '/estimator_sub91.py').read()
os.makedirs(OUT, exist_ok=True)
BL = '        out[-1] = ours + _W_BLEND * ((1.0 + _ALPHA) * m_deep - ours)'


def variant(tag, g=None, deep=False):
    """`g` — глобальний масштаб усіх `λ`; `deep` — віддати ЧИСТЕ замикання."""
    s = SRC.replace(BL, '        out[-1] = m_deep' if deep
                    else '        out[-1] = ours', 1)
    if g is not None:
        s = s.replace('def _phi', f'''_G = {g}
_LAM_S = tuple(_G * v for v in _LAM_S)
_LAM_A = tuple(_G * v for v in _LAM_A)
_LAM_Q = tuple(_G * v for v in _LAM_Q)
_LAM_G = tuple(_G * v for v in _LAM_G)


def _phi''', 1)
    p = f'{OUT}/cf_{tag}.py'
    open(p, 'w').write(s)
    return p


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
T1 = np.stack([TRUTH2[s][0] for s, _ in NET])
T2 = np.stack([TRUTH2[s][1] for s, _ in NET])
print(__doc__.split('Запуск')[0], flush=True)


def collect(path, mult):
    """Повертає (прогнози (NETS,W), середня ціна)."""
    B = B0 * mult
    E = load_estimator(path)()
    ps, cs = [], []
    for s0, j in NET:
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
        ps.append(a)
        cs.append(1.0 if (blown or C > B) else max(0.1, C / B))
    return np.stack(ps), float(np.mean(cs))


def mse(pred, idx):
    return float(np.mean((pred[idx] - T1[idx]) * (pred[idx] - T2[idx])))


HALF = NETS // 2
IDX = [np.arange(HALF), np.arange(HALF, NETS)]
PATHS = {g: variant(f'g{g}', g=g) for g in GS}
PATHS['deep'] = variant('deep', deep=True)
PATHS['mc'] = SK + '/estimator_mc_plain.py'

CURVE = {}
for mult in MULTS:
    print(f'\n{"=" * 74}\nМНОЖНИК ×{mult} · {NETS} мереж · крос-ціль',
          flush=True)
    P = {}
    for g in GS:
        P[g], price = collect(PATHS[g], mult)
    Pd, _ = collect(PATHS['deep'], mult)
    Pm, _ = collect(PATHS['mc'], mult)
    m_mc = mse(Pm, np.arange(NETS))
    m_deeponly = mse(Pd, np.arange(NETS))

    # ⛔ПЛОМБА: відтворити повний sub91 офлайн-формулою
    seal = P[1.0] + 0.0651 * ((1.0 - 0.00751) * Pd - P[1.0])
    print(f'⛔ПЛОМБА (g=1, w=0.0651, α=−0.00751): MSE = '
          f'{mse(seal, np.arange(NETS)):.4e}', flush=True)
    print(f'   ⛔МК {m_mc:.4e} · 🔵чисте замикання {m_deeponly:.4e}')

    # 🔍 повний скан (g, w, α)
    best = (1e9, None)
    oos = []
    for f in (0, 1):
        fit, ev = IDX[f], IDX[1 - f]
        bl = (1e9, None)
        for g in GS:
            for w in WGRID:
                for al in AGRID:
                    pr = P[g] + w * ((1.0 + al) * Pd - P[g])
                    v = mse(pr, fit)
                    if v < bl[0]:
                        bl = (v, (g, w, al))
        g, w, al = bl[1]
        pr = P[g] + w * ((1.0 + al) * Pd - P[g])
        oos.append(mse(pr, ev))
    for g in GS:
        for w in WGRID:
            for al in AGRID:
                v = mse(P[g] + w * ((1.0 + al) * Pd - P[g]), np.arange(NETS))
                if v < best[0]:
                    best = (v, (g, w, al))
    g, w, al = best[1]
    CURVE[mult] = (g, w, al, best[0], float(np.mean(oos)), m_mc)
    print(f'🎯 ОРАКУЛ (підбір і вимір на тих самих): g={g:.1f} w={w:+.3f} '
          f'α={al:+.3f} ⟹ MSE {best[0]:.4e}')
    print(f'✅ ЧЕСНО (підбір на 12, вимір на інших 12): MSE '
          f'{np.mean(oos):.4e}', flush=True)

print(f'\n{"=" * 74}\n🔑 КРИВІ КОЕФІЦІЄНТІВ І ЩО ЛИШАЄТЬСЯ ВІД ПІДЛОГИ')
print(f'{"множ.":>6s} {"g(λ)":>6s} {"w(бленд)":>9s} {"α":>7s} '
      f'{"MSE оракул":>11s} {"MSE чесно":>11s} {"MSE МК":>11s} {"МК/чесно":>9s}')
print('-' * 78)
for m in MULTS:
    g, w, al, bo, bh, mk = CURVE[m]
    print(f'{"×" + str(m):>6s} {g:6.1f} {w:+9.3f} {al:+7.3f} {bo:11.4e} '
          f'{bh:11.4e} {mk:11.4e} {mk / bh:8.4f}×')
print('\n📐 замерзлими константами було: 3.86e-07 (×32) і 3.27e-07 (×64), '
      'МК обганяв нас уже при ×32')
print('📊 БАЛ 1.954e-07 · робоча точка — ×1, саме там константи й підігнані')
