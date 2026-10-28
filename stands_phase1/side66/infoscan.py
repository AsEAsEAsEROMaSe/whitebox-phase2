"""ДЕТЕКТОР ІНФОРМАЦІЇ: чи є в ознаці сигнал ВЗАГАЛІ — до всякої теорії (С82)

## НАВІЩО
Досі кожна гілка коштувала сесію: вигадати теорію → побудувати → зміряти →
мертво. Питання юзера (С82): «чи є спосіб виявити, чи є там інфа взагалі?»
Є. І він не потребує самої теорії.

## 🔑 ЯК ЦЕ ПРАЦЮЄ (без формул)
Кожна точка — це (мережа, жереб, вихідний нейрон). Для неї відомі ознака `X`
і залишок `e` нашого сабміту. Розбиваємо всі точки на 40 бінів ПО `X` і
дивимось, чи всередині біна залишок систематично зміщений. Якщо так — `X`
щось знає про `e`. Це ловить БУДЬ-ЯКУ одновимірну залежність, а не лише
лінійну: східчаста функція наближає що завгодно.

## 🚨 ТРИ КОНТРОЛІ, БЕЗ ЯКИХ ПРИЛАД БРЕШЕ В НАШ БІК
1. **LOO ПО МЕРЕЖАХ** — бінову функцію будуємо на інших мережах, міряємо на
   відкладеній. Без цього 40 бінів «пояснять» будь-який шум.
2. **КРОС-ЦІЛЬ** — вирок = `E[(e₁−f)(e₂−f)]` з двома НЕЗАЛЕЖНИМИ істинами.
   Шум істини входить у `e₁` і `e₂` різними жеребами ⟹ у добутку гасне
   ([[finding-bank-truth-noise-inflates-every-mse-we-measure]]).
3. **ПЕРЕСТАНОВКА** — та сама процедура на перемішаній `X`. Це ПІДЛОГА
   приладу: усе, що не вище неї, — не сигнал
   ([[feedback-controls-have-three-distinct-diseases]]).

## 🎯 ЩО ВІДРІЗНЯЄ (задля цього все й будувалось)
| бінова ≈ перестановка | інфи НЕМА — математика на цій `X` мертва КЛАСОМ |
| бінова ≈ лінійна > перест. | інфа є, наша λ-регресія вже все бере |
| **бінова ≫ лінійна** | інфа є, наша ФОРМА її не бачить ⟹ ось де новий апарат |

Досі ми вміли міряти лише середній стовпчик і не відрізняли перший від
третього. Саме тому «вичерпано» могло означати «вичерпано ЛІНІЙНО».

## ЦІЛЬ — ЖЕРЕБО-ЗАЛЕЖНА ЧАСТИНА
Зміщення вже відоме й мале (7.6%, стеля 1.08× — [[finding-bias-is-8-percent-not-46]]),
увесь приз у ДИСПЕРСІЇ. Тому міряємо в двох режимах:
  `raw`  — сирий залишок (побачить і зміщення);
  `dev`  — залишок мінус середнє по жеребах ⟹ ЧИСТА флуктуація жеребу.
І дві шкали: сира та поділена на замикальний масштаб нейрона `m_cl[30]`
(бо похибка може бути просто пропорційна величині нейрона — це нелінійність,
яку лінійна регресія по 62 ознаках НЕ бачить, а бінова бачить одразу).

## ОЗНАКИ (усі БЕЗКОШТОВНІ — уже рахуються в сабміті)
σ-ознаки і μ-якорі шарів 1,2,4,8,16,24,31, ПЕРЕНЕСЕНІ до виходу (це рівно
той базис, на якому працює λ), плюс сам прогноз, `s_cl`, `m_cl` виходу.
⛔ ПЛОМБА: ознаки шарів 1..8 УЖЕ вжиті сабмітом лінійно ⟹ на них лінійна
   колонка мусить показати ≈0 (їх внесок уже вирахувано), а якщо бінова там
   покаже багато — це прямий доказ невзятої НЕЛІНІЙНОСТІ.

📣 ПЕРЕДБАЧЕННЯ-1 (3 мережі) — ЧАСТКОВО СПРОСТОВАНЕ. Справдилось: підлога
   перестановки стабільна (−1.2%), пломба тавтології рівно 100.00% ✅.
   НЕ справдилось: я чекав ≤4% надлишку, а вийшло **7.67%** — μ-якір шару 16
   лінійно 6.08%, біновою **13.75%**. Тобто форма поправки НЕ лінійна.
   ⚠️ Але 3 мережі не розсуджують ([[feedback-small-runs-do-not-arbitrate]]),
   і лишались два незакриті артефакти, тому додано ДРУГИЙ контроль —
   ознака з ЧУЖОЇ мережі — і 2D-тест (ознака × масштаб нейрона).

📣 ПЕРЕДБАЧЕННЯ-2 (цей прогін, 12 мереж): надлишок ПРОСЯДЕ, бо частину його
   давало «бінова вгадує, яка це мережа» (масштаби мереж різні, а бін-краї
   глобальні) — контроль «чужа мережа» саме це й ловить. Чекаю 2–4% після
   контролю. Якщо ж утримається ≥6% на ГЛИБОКИХ якорях (16, 24, 31) — це
   означає, що глибокий якір мертвий не за інформацією, а за ФОРМОЮ, і тоді
   [[finding-deep-anchor-dies-on-closure-error-not-on-lambda]] треба
   перевідкривати: там міряли ЛІНІЙНИЙ канал.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/infoscan.py [cuda:N] [NETS]
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
L_, KS, NDRAW, NBIN, NB2 = 32, 31, 32, 40, 16
LAYERS = (0, 1, 3, 7, 15, 23, 30)          # шари 1,2,4,8,16,24,31
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

NAMES = ([f'σ-ознака шару {l+1}' for l in LAYERS]
         + [f'μ-якір шару {l+1}' for l in LAYERS]
         + ['прогноз сабміту', 's_замик виходу', 'm_замик виходу']
         # 🚨🚨 ПЛОМБА ПОТУЖНОСТІ (С82, на вимогу юзера «не рубай, перевір
         #    код»). Питання: чи прилад ВЗАГАЛІ здатен побачити нелінійність
         #    при 12 мережах, чи LOO+біни її вбивають? Відповідь дає ключова
         #    властивість: **бінова функція ІНВАРІАНТНА до монотонного
         #    перетворення ознаки, а лінійна — НІ.** Тож `a³` і `∛a` того
         #    самого μ-якоря шару 24 мусять дати:
         #      бінова  — РІВНО стільки ж, скільки на сирому `a` (8.07%);
         #      лінійна — ІСТОТНО МЕНШЕ (зв'язок спотворений).
         #    Якщо бінова просяде разом із лінійною — прилад СЛІПИЙ, і вирок
         #    «нелінійності нема» НЕДІЙСНИЙ. Витоку тут нема: ознака робиться
         #    з тієї самої ознаки, не з цілі.
         + ['🚨пломба: (μ-якір 24)³', '🚨пломба: ∛(μ-якір 24)'])
NF = len(NAMES)
JA = 12                      # ⟵ індекс μ-якоря шару 24 у стосі ознак

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]

XS, E1, E2, NET, SCL = [], [], [], [], []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        s_cl, m_cl = closure_parts(wc, KS, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        sc = m_cl[KS - 1].clamp_min(1e-12)
        for d in range(NDRAW):
            sd = 77_000_000 + 733 * n + d
            pred, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], DEV, sd,
                                lam_s=LAM_S)
            _, S = run_side1(wc, npair, s_cl, m_cl, DEV, sd,
                             collect_sigma=True)
            _, A = run_side1(wc, npair, s_cl, m_cl, DEV, sd,
                             lam_s=None, collect_anchors=True)
            X = torch.stack([S[l] for l in LAYERS] + [A[l] for l in LAYERS]
                            + [pred.double(), s_cl[KS - 1], m_cl[KS - 1]], 1)
            XS.append(X.cpu().numpy().astype(np.float64))
            E1.append((pred - T1).cpu().numpy().astype(np.float64))
            E2.append((pred - T2).cpu().numpy().astype(np.float64))
            SCL.append(sc.cpu().numpy().astype(np.float64))
            NET.append(np.full(X.shape[0], n, dtype=np.int32))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

X = np.concatenate(XS); e1 = np.concatenate(E1); e2 = np.concatenate(E2)
net = np.concatenate(NET); scl = np.concatenate(SCL)
a = X[:, JA]
X = np.concatenate([X, np.stack([np.sign(a) * np.abs(a) ** 3,
                                 np.sign(a) * np.abs(a) ** (1.0 / 3.0)], 1)], 1)
np.savez_compressed('/mnt/win_d/llm/White-Box/arena62/logs/infoscan_data.npz',
                    X=X.astype(np.float32), e1=e1, e2=e2, net=net, scl=scl)
# 🔑 `dev`-режим: віднімаємо середнє ПО ЖЕРЕБАХ для кожної пари (мережа,нейрон).
#    Рядки складені як [мережа][жереб][нейрон] ⟹ reshape по жеребах.
off, dev1, dev2, devX = 0, e1.copy(), e2.copy(), X.copy()
for n in range(NETS):
    k = int((net == n).sum()) // NDRAW
    blk = slice(off, off + k * NDRAW)
    for arr, src in ((dev1, e1), (dev2, e2)):
        v = src[blk].reshape(NDRAW, k)
        arr[blk] = (v - v.mean(0)).reshape(-1)
    v = X[blk].reshape(NDRAW, k, NF)
    devX[blk] = (v - v.mean(0)).reshape(-1, NF)
    off += k * NDRAW
print(f'\nточок: {len(e1)}  ({NETS} мереж × {NDRAW} жеребів)')


def score(x, a1, a2, nt, mode):
    """Частка крос-MSE, яку знімає поправка з `x`. LOO ПО МЕРЕЖАХ."""
    q = np.quantile(x, np.linspace(0, 1, NBIN + 1)[1:-1])
    b = np.searchsorted(q, x)
    ee = 0.5 * (a1 + a2)
    base = float(np.mean(a1 * a2))
    fit = np.zeros_like(ee)
    for n in range(NETS):
        te = nt == n
        tr = ~te
        if mode == 'bin':
            s = np.bincount(b[tr], ee[tr], NBIN)
            c = np.bincount(b[tr], None, NBIN)
            fit[te] = np.where(c > 30, s / np.maximum(c, 1), 0.0)[b[te]]
        else:                                   # лінійна — той самий LOO
            xt, et = x[tr], ee[tr]
            vx = xt.var()
            k = (0.0 if vx < 1e-300
                 else float(((xt - xt.mean()) * (et - et.mean())).mean() / vx))
            fit[te] = et.mean() + k * (x[te] - xt.mean())
    return 1.0 - float(np.mean((a1 - fit) * (a2 - fit))) / base


def score2(x, y, a1, a2, nt):
    """Те саме, але бінова функція ДВОХ змінних (сітка NB2×NB2)."""
    qx = np.quantile(x, np.linspace(0, 1, NB2 + 1)[1:-1])
    qy = np.quantile(y, np.linspace(0, 1, NB2 + 1)[1:-1])
    b = np.searchsorted(qx, x) * NB2 + np.searchsorted(qy, y)
    ee = 0.5 * (a1 + a2)
    base = float(np.mean(a1 * a2))
    fit = np.zeros_like(ee)
    for n in range(NETS):
        te, tr = nt == n, nt != n
        s = np.bincount(b[tr], ee[tr], NB2 * NB2)
        c = np.bincount(b[tr], None, NB2 * NB2)
        fit[te] = np.where(c > 30, s / np.maximum(c, 1), 0.0)[b[te]]
    return 1.0 - float(np.mean((a1 - fit) * (a2 - fit))) / base


rng = np.random.default_rng(4242)
perm = rng.permutation(len(e1))
TAUT = 100.0 * score(devX[:, NF - 3], dev1, dev2, net, 'lin')
print(f'⛔ ПЛОМБА ТАВТОЛОГІЇ (прогноз проти власної флуктуації): {TAUT:.2f}% '
      f'{"✅" if TAUT > 99.0 else "🚨 ПРИЛАД ЗЛАМАНИЙ"}')

# 🚨 ДРУГИЙ КОНТРОЛЬ, сильніший за перестановку: ознака з ЧУЖОЇ мережі. Він
#    зберігає РОЗПОДІЛ ознаки (перестановка теж, але руйнує і зв'язок «мережа
#    ↔ масштаб»), тож ловить артефакт «бінова просто вгадує, яка це мережа».
off, shift = 0, np.zeros(len(e1), dtype=np.int64)
for n in range(NETS):
    k = int((net == n).sum())
    nxt = (off + k) % len(e1)
    shift[off:off + k] = (np.arange(k) + nxt) % len(e1)
    off += k

# 🚨 ПОПРАВКА С82: перша версія ділила на сирий `m_замик`, і лінійна колонка
#    давала РІВНО 0.42% для всіх ознак підряд — ознака виродженого фіту. Причина:
#    `m_замик` подекуди майже нуль ⟹ `X/SC` вибухає, дисперсію з'їдають викиди,
#    нахил падає в нуль, і 0.42% — це внесок самого лише інтерсепта.
#    ⟹ клампимо знизу часткою медіани. 🎓 ОДНАКОВЕ ЧИСЛО В РІЗНИХ РЯДКАХ —
#    ЗАВЖДИ БАГ, А НЕ ЗБІГ.
SC = np.maximum(scl, 0.05 * float(np.median(scl)))
print(f'\n{"ознака":26s} │{"СИРА ОЗНАКА":^37s}│{"/ масштаб нейрона":^19s}')
print(f'{"":26s} │{"лін.":>8s}{"бінова":>9s}{"перест.":>9s}{"чужа":>9s}'
      f'│{"лін.":>9s}{"бінова":>9s}')
print('─' * 86)
ROWS = []
for j, nm in enumerate(NAMES):
    xr, xn = X[:, j], X[:, j] / SC
    r = [nm,
         100.0 * score(xr, e1, e2, net, 'lin'),
         100.0 * score(xr, e1, e2, net, 'bin'),
         100.0 * score(X[perm, j], e1, e2, net, 'bin'),
         100.0 * score(X[shift, j], e1, e2, net, 'bin'),
         100.0 * score(xn, e1, e2, net, 'lin'),
         100.0 * score(xn, e1, e2, net, 'bin')]
    ROWS.append(r)
    print(f'{nm:26s} │{r[1]:7.2f}%{r[2]:8.2f}%{r[3]:8.2f}%{r[4]:8.2f}%'
          f'│{r[5]:8.2f}%{r[6]:8.2f}%')

# 🔑 ДВОВИМІРНИЙ ТЕСТ на трьох найкращих: чи похибка залежить від ознаки
#    МОДУЛЬОВАНО масштабом нейрона — саме та форма, якої лінійна λ не має.
NR = NF - 2                     # ⟵ реальні ознаки (без двох пломбових)
sb, sl = ROWS[JA][2], ROWS[JA][1]
print(f'\n⛔ ПЛОМБА ПОТУЖНОСТІ (чи бачить прилад нелінійність при {NETS} мережах)')
print(f'   сирий μ-якір 24:  лінійна {sl:6.2f}%   бінова {sb:6.2f}%')
for j in (NF - 2, NF - 1):
    print(f'   {NAMES[j]:22s}  лінійна {ROWS[j][1]:6.2f}%   '
          f'бінова {ROWS[j][2]:6.2f}%')
ok = (abs(ROWS[NF - 2][2] - sb) < 1.0 and abs(ROWS[NF - 1][2] - sb) < 1.0
      and ROWS[NF - 2][1] < sl - 1.0)
print(f'   ⟹ {"✅ ПРИЛАД ЗРЯЧИЙ" if ok else "🚨 ПРИЛАД СЛІПИЙ — ВИРОК НЕДІЙСНИЙ"}'
      ': бінова інваріантна до монотонної заміни, лінійна — ні')

best = sorted(range(NR), key=lambda j: -ROWS[j][2])[:3]
print(f'\n2D (ознака × масштаб нейрона), сітка {NB2}×{NB2}:')
for j in best:
    v = 100.0 * score2(X[:, j], SC, e1, e2, net)
    c = 100.0 * score2(X[perm, j], SC, e1, e2, net)
    print(f'  {NAMES[j]:26s} 2D {v:7.2f}%   ⛔перест. {c:6.2f}%   '
          f'(1D бінова {ROWS[j][2]:.2f}%)')

print('\n🎯 ЩО ШУКАЄМО: «бінова» ЗНАЧНО вище за «лін.» ⟹ невзята НЕЛІНІЙНІСТЬ.\n'
      '   Усе, що не вище «перест.» І «чужа», — підлога приладу, не сигнал.')
g = max(ROWS[j][2] - max(ROWS[j][1], ROWS[j][3], ROWS[j][4])
        for j in range(NR))
jb = max(range(NR), key=lambda j: ROWS[j][2] - max(ROWS[j][1], ROWS[j][3],
                                                  ROWS[j][4]))
print(f'⚖️ НАЙБІЛЬШИЙ ЧИСТИЙ НАДЛИШОК НАД ЛІНІЙНОЮ: {g:.2f}%  ({NAMES[jb]})')
print(f'📊 БАЛ 1.954e-07 ⟹ якби надлишок брався весь: {1.954e-07*(1-g/100):.4e}')
