"""ДЕТЕКТОР ІНФОРМАЦІЇ, НАВЕДЕНИЙ ПОЗА λ-БАЗИС (С82)

## ЗВІДКИ
`infoscan.py` сканував рівно той базис, на якому λ і працює (σ-ознаки +
μ-якорі) — і не знайшов невзятої нелінійності (0.97–1.19% при підлозі 0.4%,
пломба потужності ✅ [[finding-lambda-basis-has-no-nonlinearity-left]]).
Юзер: «далі метод цього пошуку варто застосувати і до інших речей».
Тут прилад наводиться на статистики, яких у λ-базисі НЕМА ВЗАГАЛІ.

## 🔑 ЧОМУ ЦЕ ДЕШЕВО
`run_side1(..., anchor_multi=(kinds, refs))` уже вміє переносити до виходу
ДОВІЛЬНІ пошарові статистики тим самим ланцюгом воріт. Чотири нові сімейства
беруться ОДНИМ прогоном:
  `ind` — частка активних `q̂ − Φ(t)`      (ворота)
  `a2`  — другий момент активацій          (дисперсія, не середнє)
  `p3`  — третій момент передактивацій     (асиметрія)
  `p4`  — четвертий момент передактивацій  (хвости)
Опори будуються з замикання. Для `p3`,`p4` потрібен `mp`, якого
`closure_parts` не віддає — дістаємо точним оберненням `mp = t·s`,
`t = √2·erfinv(2·f_cl−1)` (той самий трюк, що врятував `qgam.py`).
    гаусові опори: `E[p³] = mp³+3mp·s²`, `E[p⁴] = mp⁴+6mp²s²+3s⁴`.
⛔ ПЛОМБА ЛАНЦЮГА: п'яте сімейство `a` з опорою `m_cl` — це РІВНО штатний
   μ-якір, лише поділений на скаляр. Кореляція мусить бути **1.000000**.
   (Перша версія мала тут беззмістовну «пломбу опор» на границі `t→∞` —
   вона нічого не перевіряла, бо формула точна за побудовою.)

## 🚨 КОНТРОЛІ — ТІ САМІ, ЩО ВИСТОЯЛИ ПЕРЕВІРКУ
LOO по мережах · крос-ціль · перестановка · ознака з ЧУЖОЇ мережі ·
**пломба потужності** (`a³`/`∛a` — бінова мусить лишитись тотожною)
⟹ [[feedback-verify-code-after-the-experiment]].

📣 ПЕРЕДБАЧЕНО: `a2` дасть найбільше з нових (він найближчий до σ-каналу, а
   той працює), `ind` — мало (канал воріт закрито С81), `p3`/`p4` — майже
   нічого поодинці, бо їхній шум вибірки росте зі степенем. Головне питання
   не «скільки», а **чи є хоч в однієї бінова ЗНАЧНО вище за лінійну** —
   бо саме це відрізняє «нової величини нема» від «нової ФОРМИ нема».

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/infoscan_x.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S)
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
L_, KS, NDRAW, NBIN = 32, 31, 32, 40
LAYERS = (0, 1, 3, 7, 15, 23, 30)
KINDS = ['a', 'ind', 'a2', 'p3', 'p4']
TAG = {'a': '⛔пломба μ-якір', 'ind': 'ворота q̂', 'a2': '2-й момент a',
       'p3': '3-й момент p', 'p4': '4-й момент p'}
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

NAMES = ([f'{TAG[k]} шару {l+1}' for k in KINDS for l in LAYERS]
         + ['μ-якір шару 24 (еталон)', 'σ-ознака шару 24 (еталон)']
         + ['🚨пломба: (найкраща)³', '🚨пломба: ∛(найкраща)'])
NF = len(NAMES)
JE = len(KINDS) * len(LAYERS)          # ⟵ індекс еталонного μ-якоря

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

XS, E1, E2, NET = [], [], [], []
SEAL = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        s_cl, m_cl, f_cl, q_cl = closure_parts(wc, KS, DEV, extras=True)
        # 🔑 `mp` точним оберненням `f_cl = Φ(t)`
        refs = [[] for _ in KINDS]
        for i in range(KS):
            s = s_cl[i]
            t = 1.4142135623730951 * torch.erfinv(
                2.0 * f_cl[i].clamp(1e-12, 1.0 - 1e-12) - 1.0)
            mp, s2 = t * s, s * s
            refs[0].append(m_cl[i])          # ⛔ пломба: це штатний μ-якір
            refs[1].append(f_cl[i])
            refs[2].append(q_cl[i])
            refs[3].append(mp ** 3 + 3.0 * mp * s2)
            refs[4].append(mp ** 4 + 6.0 * mp * mp * s2 + 3.0 * s2 * s2)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for d in range(NDRAW):
            sd = 77_000_000 + 733 * n + d
            pred, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], DEV, sd,
                                lam_s=LAM_S)
            _, Q = run_side1(wc, npair, s_cl, m_cl, DEV, sd, lam_s=None,
                             collect_anchors=True, anchor_multi=(KINDS, refs))
            _, S = run_side1(wc, npair, s_cl, m_cl, DEV, sd,
                             collect_sigma=True)
            _, A = run_side1(wc, npair, s_cl, m_cl, DEV, sd, lam_s=None,
                             collect_anchors=True)
            cols = [Q[q * KS + l] for q in range(len(KINDS)) for l in LAYERS]
            cols += [A[23], S[23]]
            if n == 0 and d == 0:
                # ⛔ ПЛОМБА ЛАНЦЮГА ПЕРЕНОСУ: сімейство `a` з опорою `m_cl` —
                #    це РІВНО штатний μ-якір, лише поділений на скаляр
                #    `mean(s_cl)`. Кореляція мусить бути 1.000000.
                u, w_ = Q[23].double(), A[23].double()
                cc = float((u * w_).sum() / (u.norm() * w_.norm() + 1e-300))
                SEAL.append(cc)
            XS.append(torch.stack(cols, 1).cpu().numpy().astype(np.float64))
            E1.append((pred - T1).cpu().numpy().astype(np.float64))
            E2.append((pred - T2).cpu().numpy().astype(np.float64))
            NET.append(np.full(pred.shape[0], n, dtype=np.int32))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

X = np.concatenate(XS); e1 = np.concatenate(E1); e2 = np.concatenate(E2)
net = np.concatenate(NET)
print(f'\nточок: {len(e1)}  ({NETS} мереж × {NDRAW} жеребів)')
print(f'⛔ ПЛОМБА ЛАНЦЮГА ПЕРЕНОСУ (сімейство `a` ≡ штатний μ-якір): '
      f'corr = {SEAL[0]:.6f} '
      f'{"✅" if abs(SEAL[0]-1.0) < 1e-6 else "🚨 ЛАНЦЮГ РОЗІЙШОВСЯ"}')


def score(x, a1, a2, mode):
    q = np.quantile(x, np.linspace(0, 1, NBIN + 1)[1:-1])
    b = np.searchsorted(q, x)
    ee = 0.5 * (a1 + a2)
    base = float(np.mean(a1 * a2))
    fit = np.zeros_like(ee)
    for n in range(NETS):
        te, tr = net == n, net != n
        if mode == 'bin':
            s = np.bincount(b[tr], ee[tr], NBIN)
            c = np.bincount(b[tr], None, NBIN)
            fit[te] = np.where(c > 30, s / np.maximum(c, 1), 0.0)[b[te]]
        else:
            xt, et = x[tr], ee[tr]
            v = xt.var()
            k = 0.0 if v < 1e-300 else float(
                ((xt - xt.mean()) * (et - et.mean())).mean() / v)
            fit[te] = et.mean() + k * (x[te] - xt.mean())
    return 100.0 * (1.0 - float(np.mean((a1 - fit) * (a2 - fit))) / base)


# 🚨 пломба потужності — на НАЙКРАЩІЙ із НОВИХ ознак (визначаємо після скану)
NEW = list(range(len(LAYERS), JE))   # ⟵ пропускаємо пломбове сімейство `a`
prelim = {j: score(X[:, j], e1, e2, 'bin') for j in NEW}
jb = max(NEW, key=lambda j: prelim[j])
a = X[:, jb]
X = np.concatenate([X, np.stack([np.sign(a) * np.abs(a) ** 3,
                                 np.sign(a) * np.abs(a) ** (1.0 / 3.0)], 1)], 1)
NAMES[NF - 2] = f'🚨пломба: ({NAMES[jb]})³'
NAMES[NF - 1] = f'🚨пломба: ∛({NAMES[jb]})'
np.savez_compressed('/mnt/win_d/llm/White-Box/arena62/logs/infoscan_x.npz',
                    X=X.astype(np.float32), e1=e1, e2=e2, net=net)

rng = np.random.default_rng(4242)
perm = rng.permutation(len(e1))
off, shift = 0, np.zeros(len(e1), dtype=np.int64)
for n in range(NETS):
    k = int((net == n).sum())
    shift[off:off + k] = (np.arange(k) + (off + k) % len(e1)) % len(e1)
    off += k

print(f'\n{"ознака":28s} {"лінійна":>9s}{"бінова":>9s}{"⛔перест":>9s}{"⛔чужа":>8s}')
print('─' * 66)
R = []
for j in range(NF):
    r = [score(X[:, j], e1, e2, 'lin'), score(X[:, j], e1, e2, 'bin'),
         score(X[perm, j], e1, e2, 'bin'), score(X[shift, j], e1, e2, 'bin')]
    R.append(r)
    mark = '  ⟵' if j >= JE else ''
    print(f'{NAMES[j]:28s} {r[0]:8.2f}%{r[1]:8.2f}%{r[2]:8.2f}%{r[3]:7.2f}%{mark}')

ok = all(abs(R[j][1] - R[jb][1]) < 0.05 for j in (NF - 2, NF - 1))
print(f'\n⛔ ПЛОМБА ПОТУЖНОСТІ: бінова на ({NAMES[jb]}) та її монотонних замінах: '
      f'{R[jb][1]:.2f}% · {R[NF-2][1]:.2f}% · {R[NF-1][1]:.2f}%')
print(f'   лінійна там же: {R[jb][0]:.2f}% · {R[NF-2][0]:.2f}% · {R[NF-1][0]:.2f}%')
print(f'   ⟹ {"✅ ПРИЛАД ЗРЯЧИЙ" if ok else "🚨 СЛІПИЙ — ВИРОК НЕДІЙСНИЙ"}')

flo = max(max(R[j][2], R[j][3]) for j in NEW)
best = sorted(NEW, key=lambda j: -R[j][1])[:5]
print(f'\n🏆 П\'ЯТЬ НАЙСИЛЬНІШИХ НОВИХ (підлога приладу {flo:.2f}%)')
for j in best:
    print(f'   {NAMES[j]:28s} лін {R[j][0]:6.2f}%  бін {R[j][1]:6.2f}%  '
          f'надлишок {R[j][1]-max(R[j][0], R[j][2], R[j][3]):+6.2f}%')
g = max(R[j][1] - max(R[j][0], R[j][2], R[j][3]) for j in NEW)
tot = max(R[j][1] for j in NEW)
print(f'\n⚖️ НАДЛИШОК НЕЛІНІЙНОСТІ (нові ознаки): {g:.2f}%   ·   '
      f'НАЙБІЛЬШИЙ СИГНАЛ УЗАГАЛІ: {tot:.2f}%')
print(f'📊 БАЛ 1.954e-07 ⟹ якби найсильніша нова бралась уся: '
      f'{1.954e-07*(1-max(tot,0)/100):.4e}')
