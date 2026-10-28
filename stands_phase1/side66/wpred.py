"""ПОШУК ПРЕДИКТОРА ДЛЯ ГЛОБАЛЬНИХ ВАГ `w` І `g` (С82)

## ТЗ, ВИВЕДЕНЕ ВИМІРАМИ
`blendyn.py`: оптимальна вага бленду різна по мережах на **±57%**, стеля
персоналізації **×1.0289** ([[finding-blend-weight-varies-57-percent-across-networks]]).
`gscale.py`: те саме для спільного множника на всі 48 λ — стеля **×1.0257**.
⛔ В обох випадках предиктора НЕМА: моя формула `VO/‖ours−m_deep‖²` дає корел
лише **+0.14**. ⟹ 🎯 мішень: величина, обчислювана зсередини, з кореляцією
до оракульного оптимуму ЗНАЧНО вищою за 0.14.

## 🚨 ЧОМУ ТУТ ІНША СХЕМА ВИБІРКИ
Кореляція міряється **ПО МЕРЕЖАХ**, тож 12 мереж = 12 точок, і похибка
кореляції ≈0.3 — більша за сам сигнал. Тому: **48 мереж × 8 жеребів**
замість 12 × 32. Той самий час, учетверо більше точок для кореляції.
`w_орак` при 8 жеребах оцінюється по 8×256 ≈ 2000 значеннях — цього досить.

## КАНДИДАТИ (усі — ГЛОБАЛЬНІ скаляри мережі, доступні в сабміті задарма)
 1 `VO/‖ours−m_deep‖²`     — нинішня формула (корел 0.14)
 2 `‖ours−m_deep‖/‖ours‖`  — відносна розбіжність із замиканням
 3 `VO/‖ours‖²`            — відносний шум прогнозу
 4 `n_pairs`               — розмір вибірки після обрізання
 5 частка ЖИВИХ нейронів   — скільки лишилось після пілота
 6 `‖m_deep‖/‖ours‖`       — масштаб замикання проти нашого
 7 `⟨ours, m_deep⟩` норм.  — кут між прогнозом і замиканням
 8 `Σ_ℓ E[δ²]/Σ_ℓ E[e²]`   — агрегована якість замикання по 8 шарах
 9 `Σ_ℓ B²` норм. на ‖ours‖² — те саме, але тільки зміщення
10 медіана `s_cl` виходу / `‖ours‖`

⛔ ПЛОМБИ: (а) кандидат 1 мусить відтворити корел ≈0.14;
   (б) ⛔НЕГ — кандидат зі зсунутим порядком мереж, корел мусить бути ≈0;
   (в) ⛔ЧИСТИЙ ШУМ як кандидат.

📣 ПЕРЕДБАЧЕНО: найкращим стане кандидат 8 або 9 (пряма міра «наскільки
   грубе замикання цієї мережі»), корел 0.3…0.5. Це НЕ вистачить для
   ×1.029 повністю, але половину стелі взяти може. Якщо ж усі ≤0.2 —
   гілка «динамічні глобальні ваги» закривається як безпредикторна.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/wpred.py [cuda:N] [NETS] [NDRAW]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, predict91,      # noqa: E402
                        W_BLEND, ALPHA, LAM_S, LAM_A, LAM_Q, LAM_G)
# 🚨 ІСТИНА — З БАНКУ, НЕ `truth_last` (вказівка юзера С82: «є ж банк точності»).
#    ДВІ причини, і друга важливіша за першу:
#    1) банк точніший: шум однієї істини 1.6e-04 відносно проти мого ~2-3e-04;
#    2) 🔑 `truth_last(wc)` рахує істину ОБРІЗАНОЇ мережі, а грейдер міряє
#       ПОВНУ. Зміряно: зсув мій↔банк **1.5e-04…6.1e-04** — БІЛЬШИЙ за обидва
#       шуми ⟹ це не шум, а ПОХИБКА ОБРІЗАННЯ, яку сабміт реально робить,
#       а мій вимір її не бачив (до 20% робочого MSE).
#    🎓 `side66/ab_files.py` — головний прилад порівняння сабмітів — читає банк
#       ЩЕ З С79. Нові стенди С82 були регресом.
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L_, KS = 32, 8
COLS = list(range(KS))
NR = 4 * KS
NAMES = ['1 VO/‖ours−m_deep‖² (нинішня)', '2 ‖ours−m_deep‖/‖ours‖',
         '3 VO/‖ours‖²', '4 n_pairs', '5 частка живих нейронів',
         '6 ‖m_deep‖/‖ours‖', '7 кут(ours, m_deep)',
         '8 Σ E[δ²]/Σ E[e²] по 8 шарах', '9 Σ B² / ‖ours‖²',
         '10 медіана s_cl виходу/‖ours‖', '⛔НЕГ зсунутий', '⛔ЧИСТИЙ ШУМ']
NFT = len(NAMES)
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
NETS = len(FS)

X = np.zeros((NETS, NFT))
CRW = np.zeros((NETS, 3))       # ⟨a1,Δ⟩ ⟨a2,Δ⟩ ⟨Δ,Δ⟩ — для бленду
Z0 = np.zeros(NETS)
GG = torch.zeros(NETS, NR, NR, dtype=torch.float64, device=DEV)
BB = torch.zeros(NETS, NR, dtype=torch.float64, device=DEV)
C0 = np.zeros(NETS)
LAMV = torch.tensor([[LAM_S, LAM_A, LAM_Q, LAM_G][f][k]
                     for k in range(KS) for f in range(4)],
                    dtype=torch.float64, device=DEV)
rng0 = np.random.default_rng(11)
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        alive = sum(float(w.shape[1]) for w in wc[:-1])
        full = 256.0 * (len(wc) - 1)
        cl = closure91(wc, KS, DEV, kdeep=L_)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        acc = np.zeros(8)
        for d in range(NDRAW):
            pr, R, DG, (ours, m_deep, vout) = predict91(
                wc, npair, DEV, seed=77_000_000 + 733 * n + d, cl=cl,
                collect=COLS, diag=True, raw=True, lam_off=True)
            # 🔑 бленд бере ЛИШЕ (1−w) від λ-поправки ⟹ рядки `R` масштабуємо,
            #    інакше реконструкція `pr₀ − λᵀR` розходиться з сабмітом
            #    (у `gscale.py` це дало 0.12% на базі).
            Rs = R * (1.0 - W_BLEND)
            # 🚨 ПОПРАВКА: бленд у сабміті бере `ours` УЖЕ З λ. Перша версія
            #    будувала `Δ` від `ours` БЕЗ λ, і `w_орак` вийшов 0.021 замість
            #    0.095 (`blendyn.py`). Порядок: λ діють ДО бленду ⟹ до `ours`
            #    поправка йде БЕЗ множника (1−w), а до кінцевого `pr` — З ним.
            ours_l = ours - LAMV @ R
            o2 = float((ours_l * ours_l).sum())
            dm = ours_l - m_deep
            dl = (1.0 + ALPHA) * m_deep - ours_l
            a1, a2 = ours_l - T1, ours_l - T2
            CRW[n, 0] += float((a1 * dl).sum())
            CRW[n, 1] += float((a2 * dl).sum())
            CRW[n, 2] += float((dl * dl).sum())
            Z0[n] += float((a1 * a2).sum())
            e1, e2 = pr - T1, pr - T2
            C0[n] += float((e1 * e2).sum())
            GG[n] += Rs @ Rs.T
            BB[n] += Rs @ (0.5 * (e1 + e2))
            d2 = float((dm * dm).sum())
            sd_, se_ = 0.0, 0.0
            for D, E, _ in DG[:KS]:
                sd_ += D; se_ += E
            acc += np.array([vout / max(d2, 1e-300),
                             np.sqrt(d2 / max(o2, 1e-300)),
                             vout / max(o2, 1e-300), 0.0, 0.0,
                             float((m_deep * m_deep).sum() ** 0.5)
                             / max(o2 ** 0.5, 1e-300),
                             float((ours_l * m_deep).sum())
                             / max((o2 * float((m_deep * m_deep).sum()))
                                   ** 0.5, 1e-300),
                             sd_ / max(se_, 1e-300)])
            if d == 0:
                X[n, 8] = max(sd_ - se_, 0.0) / max(o2, 1e-300)
                X[n, 9] = float(cl[0][KS - 1].median()) / max(o2 ** 0.5, 1e-30)
        acc /= NDRAW
        X[n, :3] = acc[:3]
        X[n, 3] = npair
        X[n, 4] = alive / full
        X[n, 5:8] = acc[5:8]
        X[n, 11] = rng0.standard_normal()
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}  (npair={npair})', flush=True)

X[:, 10] = np.roll(X[:, 0], 1)                       # ⛔НЕГ

NUMW = CRW[:, 0] + CRW[:, 1]
DENW = CRW[:, 2]
w_or = -NUMW / (2.0 * np.maximum(DENW, 1e-300))
g_or = np.array([float(LAMV @ BB[n]) /
                 max(float(LAMV @ (GG[n] @ LAMV)), 1e-300)
                 for n in range(NETS)])

print(f'\n{NETS} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ.')
print(f'🔑 w_орак: сер {w_or.mean():.4f} розкид {w_or.std():.4f} '
      f'[{w_or.min():.4f}…{w_or.max():.4f}]   (сабміт {W_BLEND})')
print(f'🔑 g_орак: сер {g_or.mean():.4f} розкид {g_or.std():.4f} '
      f'[{g_or.min():.4f}…{g_or.max():.4f}]   (сабміт 1.0)')
print(f'📐 корел(w_орак, g_орак) = {np.corrcoef(w_or, g_or)[0,1]:+.4f}  '
      f'⟹ {"ОДНА величина" if abs(np.corrcoef(w_or,g_or)[0,1])>0.7 else "РІЗНІ"}')

se = 1.0 / np.sqrt(max(NETS - 3, 1))
print(f'\n{"кандидат":34s} {"корел w":>9s} {"корел g":>9s}   '
      f'(похибка ±{se:.2f})')
print('─' * 60)
best = (None, 0.0)
for j, nm in enumerate(NAMES):
    if X[:, j].std() < 1e-300:
        continue
    cw = float(np.corrcoef(X[:, j], w_or)[0, 1])
    cg = float(np.corrcoef(X[:, j], g_or)[0, 1])
    mark = ''
    if j < 10 and abs(cw) > abs(best[1]):
        best = (nm, cw)
    if abs(cw) > 2 * se:
        mark = '  🟢'
    print(f'{nm:34s} {cw:+9.3f} {cg:+9.3f}{mark}')
print(f'\n🎯 НАЙСИЛЬНІШИЙ ПРЕДИКТОР `w`: {best[0]} ({best[1]:+.3f}) · '
      f'поріг «краще за нинішній» = 0.14')
np.savez('/mnt/win_d/llm/White-Box/arena62/logs/wpred.npz',
         X=X, w_or=w_or, g_or=g_or, NUMW=NUMW, DENW=DENW, Z0=Z0,
         C0=C0, G=GG.cpu().numpy(), B=BB.cpu().numpy())
# 🎯 СТЕЛІ, ПЕРЕРАХОВАНІ НА БАНКОВІЙ ІСТИНІ (раніше — на істині обрізаної
#    мережі; порівняння з 1.0289 / 1.0257 показує, чи вижив висновок)
base_w = float(np.sum(Z0 + W_BLEND * NUMW + W_BLEND ** 2 * DENW))
orac_w = float(np.sum(Z0 + w_or * NUMW + w_or * w_or * DENW))
GGc, BBc = GG.cpu().numpy(), BB.cpu().numpy()
LV = LAMV.cpu().numpy()
base_g = float(np.sum([C0[n] - 2 * LV @ BBc[n] + LV @ (GGc[n] @ LV)
                       for n in range(NETS)]))
orac_g = float(np.sum([C0[n] - 2 * g_or[n] * (LV @ BBc[n])
                       + g_or[n] ** 2 * (LV @ (GGc[n] @ LV))
                       for n in range(NETS)]))
print(f'\n🎯 СТЕЛІ НА БАНКОВІЙ ІСТИНІ ({NETS} мереж)')
print(f'   бленд:      база {base_w:.5e} → оракул {orac_w:.5e} = '
      f'×{base_w/orac_w:.4f}   (на істині обрізаної було ×1.0289)')
print(f'   λ-масштаб:  база {base_g:.5e} → оракул {orac_g:.5e} = '
      f'×{base_g/orac_g:.4f}   (було ×1.0257)')
print(f'📊 БАЛ 1.954e-07 ⟹ стеля бленду {1.954e-07/(base_w/orac_w):.4e}, '
      f'λ-масштабу {1.954e-07/(base_g/orac_g):.4e}')
