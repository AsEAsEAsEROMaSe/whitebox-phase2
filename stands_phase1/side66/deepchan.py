"""ЧОТИРИ РОДИНИ КАНАЛІВ НА ГЛИБОКИХ ШАРАХ — ПРОТИ СПРАВЖНЬОГО САБМІТА (С82)

## ЗВІДКИ
Детектор (`infoscan_x.py`) показав, що якір на 2-й момент активацій найсильніший
НЕ на шарі 8, а на 16/24/31. Але міряв проти `estimator_side1`, а не проти
сабміту — і «новий канал» виявився `_LAM_Q`, який у сабміті з С66
([[finding-free-second-moment-channel]]). Тепер база — ПОРТ САБМІТА
`side66/sub91_port.py`, ⛔звірений із живим `estimator_sub91.py` на **1.13e-06**.

## 🔑 ЧОМУ ЦЕ МОЖЕ БУТИ БЕЗКОШТОВНО
Сабміт веде `_closure_parts(..., kdeep=depth)` крізь УСІ 32 шари — заради
`m_deep` для бленду. Отже `s,m,q,P` на шарах 9..31 у нього ВЖЕ рахуються;
зараз вони просто не зберігаються (`if 1 <= i <= ksig`). Додаткова ціна
каналу = кілька елементних операцій на шар, а не матмул.
🚨 Але вибіркові частини (`mh`,`sh`,`xm`,`xq`) на глибоких шарах теж потрібні —
це редукції по вже обчисленому `p`, тобто теж без матмулів
([[finding-pilot-samples-are-computed-and-thrown-away]] — той самий клас).

## ЩО МІРЯЄМО (LOO ПО МЕРЕЖАХ, крос-ціль)
База — сабміт як є. Додаємо δ-ознаки родин (σ · μ · E[a²] · маска) на шарах
8,12,16,20,24,28,31, перенесені до виходу, і фітуємо їхні коефіцієнти.
  1 · лише 2-й момент          2 · лише μ         3 · лише маска
  4 · УСІ ЧОТИРИ РОДИНИ        ⛔контроль: ознаки з ЧУЖОЇ мережі
⛔ ПЛОМБА: додавання ознак шарів 1..8, які в сабміті ВЖЕ Є, мусить дати
   ≈0 приросту — інакше база не та (це рівно та перевірка, якої бракувало).

📣 ПЕРЕДБАЧЕНО: С73 дала глибокому μ-якорю стелю 1.046× на шарі 24, і той
   вимір був чесний. Але 2-й момент — ІНШИЙ канал (корел лише 0.48), і на
   нього стелі ніхто не рахував. Чекаю 1.02…1.06× від нього і ≈0 від μ.
   Якщо ж усі чотири разом дадуть >1.10× — це прямий приріст балу, бо ціна
   каналу нульова.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/deepchan.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import cut_budget91, closure91, predict91, KSIG  # noqa: E402
# 🚨 ІСТИНА З БАНКУ (вказівка юзера С82: «є ж банк точності до e-10»).
#    ДВІ причини, друга важливіша: (1) банк точніший (шум 1.6e-04 відн. проти
#    моїх 2-3e-04); (2) 🔑 `truth_last(wc)` рахує істину ОБРІЗАНОЇ мережі, а
#    грейдер міряє ПОВНУ — зсув 1.5e-04…6.1e-04 БІЛЬШИЙ за обидва шуми, тобто
#    це ПОХИБКА ОБРІЗАННЯ, яку сабміт робить, а вимір її не бачив.
#    📌 Побічно: банк прибирає 2 виклики `truth_last` на мережу ⟹ прогін
#    прискорився в ~30× (48 мереж за 50 с замість 25 хв).
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
L_, NDRAW = 32, 32
DEEP = [7, 11, 15, 19, 23, 27, 30]         # шари 8,12,16,20,24,28,31
SHAL = [0, 1, 2, 3, 4, 5, 6, 7]            # ⛔ пломба: те, що вже в сабміті
FAM = ['σ', 'μ', 'E[a²]', 'маска']
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

COLS = DEEP + SHAL
ND, NS = len(DEEP), len(SHAL)
NR = 4 * len(COLS)
G = [torch.zeros(NR, NR, dtype=torch.float64, device=DEV) for _ in range(NETS)]
B = [torch.zeros(NR, dtype=torch.float64, device=DEV) for _ in range(NETS)]
GX = [torch.zeros(NR, NR, dtype=torch.float64, device=DEV) for _ in range(NETS)]
BX = [torch.zeros(NR, dtype=torch.float64, device=DEV) for _ in range(NETS)]
C0, BASE = [], []
prevR = None
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, 31, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        cc, bm = 0.0, 0.0
        for d in range(NDRAW):
            pr, R = predict91(wc, npair, DEV, seed=77_000_000 + 733 * n + d,
                              cl=cl, collect=COLS)
            e1, e2 = pr - T1, pr - T2
            e = 0.5 * (e1 + e2)
            cc += float((e1 * e2).sum())
            bm += float((e1 * e2).sum())
            G[n] += R @ R.T
            B[n] += R @ e
            if prevR is not None and prevR.shape == R.shape:
                GX[n] += prevR @ prevR.T
                BX[n] += prevR @ e
            prevR = R
        C0.append(cc); BASE.append(bm)
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

base = np.array(BASE)


RIDGE = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def loo(idx, GG, BB, lam=None):
    """LOO по мережах. `lam=None` ⟹ повернути НАЙКРАЩИЙ по сітці гребеня.

    🚨 Пробний прогін (3 мережі, гребінь 1e-8) дав ⛔пломбі 0.28× — тобто
    32 СИЛЬНО КОЛІНЕАРНІ ознаки перефітились на двох тренувальних мережах.
    Це хвороба регресії, не фізики: сам сабміт фітувався гребеневою регресією
    на 60 мережах, і там записано «ознаки сильно колінеарні, від 0 до 1.0
    результат 1.135–1.152». Тому гребінь тут — не деталь, а необхідність."""
    k = len(idx)
    ii = torch.tensor(idx, device=DEV)
    out = []
    for lm in ([lam] if lam is not None else RIDGE):
        tot = 0.0
        for n in range(NETS):
            g = sum(GG[j][ii][:, ii] for j in range(NETS) if j != n)
            b = sum(BB[j][ii] for j in range(NETS) if j != n)
            g = g + lm * torch.eye(k, dtype=torch.float64, device=DEV) * \
                g.diagonal().mean()
            w = torch.linalg.solve(g, b)
            tot += (C0[n] - 2.0 * float(w @ BB[n][ii])
                    + float(w @ (GG[n][ii][:, ii] @ w)))
        out.append(tot)
    return min(out) if lam is None else out[0], out


def rows(fams, cols):
    return [4 * COLS.index(c) + f for c in cols for f in fams]


tot0 = float(base.sum())
print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, LOO ПО МЕРЕЖАХ.')
print('🚨 колонка «виграш» = НАЙКРАЩИЙ гребінь ⟹ ОРАКУЛЬНА (підбір на тій самій\n'
      '   метриці). Чесний вирок — по ВСІЙ кривій праворуч: якщо канал живий,\n'
      '   він виграє на ШИРОКОМУ діапазоні, а не в одній точці.')
print(f'{"варіант":38s} {"сума":>10s} {"оракул":>9s}   '
      + ' '.join(f'{r:g}' for r in RIDGE))
print('─' * 60)
print(f'{"база = САБМІТ sub91 як є":38s} {tot0:10.4e} {"—":>9s}')
CASES = [('⛔ПЛОМБА: родини на шарах 1..8 (вже є)', rows(range(4), SHAL)),
         ('🟢 лише E[a²], шари 8..31', rows([2], DEEP)),
         ('🟢 лише μ, шари 8..31', rows([1], DEEP)),
         ('🟢 лише маска, шари 8..31', rows([3], DEEP)),
         ('🟢 лише σ, шари 8..31', rows([0], DEEP)),
         ('🎯 УСІ ЧОТИРИ, шари 8..31', rows(range(4), DEEP))]
res = {}
for nm, idx in CASES:
    v, cur = loo(idx, G, B)
    res[nm] = tot0 / v
    print(f'{nm:38s} {v:10.4e} {tot0/v:8.4f}×   '
          + ' '.join(f'{tot0/c:.3f}' for c in cur))
vx, _ = loo(rows(range(4), DEEP), GX, BX)
print(f'{"⛔КОНТРОЛЬ: ті самі ознаки з ЧУЖОГО жеребу":38s} {vx:10.4e} '
      f'{tot0/vx:8.4f}×')

g = res['🎯 УСІ ЧОТИРИ, шари 8..31']
print(f'\n🎯 ЧОТИРИ РОДИНИ НА ГЛИБИНІ: ×{g:.4f} проти ⛔контролю '
      f'×{tot0/vx:.4f}')
print(f'📊 БАЛ 1.954e-07 ⟹ {1.954e-07/max(g,1e-9):.4e}')
