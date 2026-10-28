"""ДИНАМІЧНІ λ: КОРПУСНА КОНСТАНТА × ФОРМУЛА ЦІЄЇ МЕРЕЖІ (С82)

## ЗВІДКИ
`side66/perlayer.py` (12 мереж × 32 жереби) показав:
* ⛔ АБСОЛЮТНИЙ масштаб формули `λ*=E[e²]/E[δ²]` зміщений (0.095 проти
  оракульних 0.012 на шарі 31), пломба `E[δ²]≥E[e²]` порушена 49/372;
* 🟢 але КОРЕЛЯЦІЯ формули з оракульним оптимумом ПО МЕРЕЖАХ **росте з
  глибиною**: шар 10 +0.523 · 16 +0.530 · 19 +0.614 · 25 **+0.795** ·
  28 +0.780 · 31 +0.762 — тоді як 5 кандидатів С2 не давали більше 0.18
  ([[finding-s2-per-element-beta-is-worth-1.8x-blocker-is-predicting-r]]).
⟹ формула вгадує, У ЯКОЇ МЕРЕЖІ цей шар кориснішій, але не в яких одиницях.
Зміщений масштаб лікується ОДНІЄЮ корпусною константою на шар — динаміка по
мережах при цьому лишається.

## 🔑 МОДЕЛЬ (і чому вона чесна)
    поправка = Σ_ℓ c_ℓ · λ_форм(мережа, ℓ) · δ_ℓ→вихід
`c_ℓ` — 8 (або 31) корпусних чисел, як зараз. Отже параметрів СТІЛЬКИ Ж, і
порівняння з нинішнім сабмітом чесне: та сама ємність, той самий LOO.
Різниця лише в тому, що ознака ПЕРЕМАСШТАБОВАНА власним числом мережі.
Ціна в сабміті — дві редукції на шар, нуль матмулів.

## ВАРІАНТИ (12 мереж × 32 жереби, база = ПОРТ САБМІТА, крос-ціль, LOO)
  A ⛔ сирі δ, корпусні λ           — відтворює нинішній сабміт
  B 🎯 δ × λ_форм(мережа,шар)       — динаміка
  C ⛔ δ × λ_форм із ЧУЖОЇ мережі   — контроль: чи не просто «ще одна вага»
  D ⛔ δ × ПЕРЕМІШАНІ по шарах       — контроль: чи важить сам ПРОФІЛЬ
на двох наборах шарів: 1..8 (де сабміт має канали) і 1..31 (уся глибина).

📣 ПЕРЕДБАЧЕНО: на шарах 1..8 виграшу НЕ буде — там кореляція формули лише
   0.04…0.19, і саме там канал живий. На 1..31 формула вгадує добре, але
   `deepchan.py` уже показав, що глибокі канали мертві проти сабміту
   ([[finding-deep-channels-die-against-the-real-submission]]).
   🎯 Тобто чекаю ×1.00…1.01 і вирок «динаміка вгадується там, де нема чого
   вигравати». Якщо ж вийде >1.03× — це прямий приріст балу за нульову ціну.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/dynlam.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import cut_budget91, closure91, predict91        # noqa: E402
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
L_, NDRAW, KS = 32, 32, 31
COLS = list(range(KS))
RIDGE = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
VAR = ['A ⛔ сирі δ (нинішній сабміт)', 'B 🎯 δ × λ_форм ЦІЄЇ мережі',
       'C ⛔ δ × λ_форм ЧУЖОЇ мережі', 'D ⛔ δ × λ_форм, шари ПЕРЕМІШАНІ']
NV = len(VAR)
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

G = [[torch.zeros(KS, KS, dtype=torch.float64, device=DEV)
      for _ in range(NETS)] for _ in range(NV)]
B = [[torch.zeros(KS, dtype=torch.float64, device=DEV)
      for _ in range(NETS)] for _ in range(NV)]
C0 = []
LFa = np.zeros((NETS, KS))
rng0 = np.random.default_rng(7)
SHUF = rng0.permutation(KS)
prevLF = None
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KS, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        d2 = np.zeros(KS); e2 = np.zeros(KS)
        cc = 0.0
        Rs, Es = [], []
        for d in range(NDRAW):
            pr, R, DG = predict91(wc, npair, DEV,
                                  seed=77_000_000 + 733 * n + d, cl=cl,
                                  collect=COLS, diag=True)
            e1, e2v = pr - T1, pr - T2
            cc += float((e1 * e2v).sum())
            Rs.append(R[1::4].clone())          # ⟵ родина μ
            Es.append(0.5 * (e1 + e2v))
            for k, (D, E, _) in enumerate(DG[:KS]):
                d2[k] += D; e2[k] += E
        lf = e2 / np.maximum(d2, 1e-300)
        LFa[n] = lf
        # 🔑 усадка мусить лежати в [0,1] — там, де формула дала >1, беремо 1
        lf = np.clip(lf, 0.0, 1.0)
        alt = np.clip(prevLF, 0.0, 1.0) if prevLF is not None else lf
        prevLF = e2 / np.maximum(d2, 1e-300)
        W_ = [np.ones(KS), lf, alt, lf[SHUF]]
        for v in range(NV):
            wv = torch.tensor(W_[v], dtype=torch.float64, device=DEV)
            for R, e in zip(Rs, Es):
                Rw = R * wv.unsqueeze(1)
                G[v][n] += Rw @ Rw.T
                B[v][n] += Rw @ e
        C0.append(cc)
    del W, wc, Rs, Es
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

tot0 = float(sum(C0))


def loo(v, idx):
    ii = torch.tensor(idx, device=DEV)
    out = []
    for lm in RIDGE:
        tot = 0.0
        for n in range(NETS):
            g = sum(G[v][j][ii][:, ii] for j in range(NETS) if j != n)
            b = sum(B[v][j][ii] for j in range(NETS) if j != n)
            g = g + lm * torch.eye(len(idx), dtype=torch.float64,
                                   device=DEV) * g.diagonal().mean()
            w = torch.linalg.solve(g, b)
            tot += (C0[n] - 2.0 * float(w @ B[v][n][ii])
                    + float(w @ (G[v][n][ii][:, ii] @ w)))
        out.append(tot)
    return out


print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, LOO ПО МЕРЕЖАХ.')
print(f'🔑 λ_форм: середнє по корпусу {LFa.mean():.4f}, '
      f'частка >1 (обрізано): {100*float((LFa > 1).mean()):.1f}%')
for nm, idx in (('ШАРИ 1..8 (де сабміт має канали)', list(range(8))),
                ('ШАРИ 1..31 (уся глибина)', list(range(KS)))):
    print(f'\n── {nm} ' + '─' * (44 - len(nm)))
    print(f'{"варіант":34s} ' + ' '.join(f'{r:>7g}' for r in RIDGE))
    for v in range(NV):
        o = loo(v, idx)
        print(f'{VAR[v]:34s} ' + ' '.join(f'{tot0/x:7.4f}' for x in o))
print('\n🎯 ЧИТАТИ: рядок B проти рядка A на ТОМУ САМОМУ гребені. '
      'C і D — підлога.')
print('📊 БАЛ 1.954e-07')
