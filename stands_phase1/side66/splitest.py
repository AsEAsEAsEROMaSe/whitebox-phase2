"""А ЯКЩО КОЖЕН МЕТОД НА СВОЇХ СЕМПЛАХ? (питання юзера, С77)

На СПІЛЬНИХ семплах похибки скорельовані (`ρ=0.91`) і бленд дає нуль
([[finding-two-estimators-on-one-draw-carry-no-direction]]). На НЕЗАЛЕЖНИХ
похибки НЕзалежні, і зважений бленд у принципі б'є обидва — але за це треба
ПОДІЛИТИ БЮДЖЕТ, а він у нас фіксований.

🧮 ТЕОРІЯ (і вона тут вирішує). Хай `MSE = k/C` (закон 1/C, у нас перевірений).
Ділимо `C = C_A + C_B`. Для НЕЗАЛЕЖНИХ похибок оптимальний бленд дає
    1/MSE = 1/MSE_A + 1/MSE_B = C_A/k_A + C_B/k_B
— це ЛІНІЙНА функція від `(C_A, C_B)` на відрізку `C_A + C_B = C`, тож її
максимум завжди в КУТІ. ⟹ **увесь бюджет у метод із меншим `k`, ділити НІКОЛИ
не вигідно.** Рівність лише при `k_A = k_B`.

🚨 Але теорія спирається на «чиста дисперсія». У `sub87` є підлога зміщення
(родини `λ` прив'язані до ЗАМИКАННЯ, а не до істини; пілот 128 пар; гейт 1024),
а `v2` незміщена. Якщо зміщення важить на робочій точці, домішування `v2` могло б
його розбавити. Це вже НЕ теорія — це вимір.

📣 ПЕРЕДБАЧЕНО: оптимум у куті `α=1` (тільки `sub87`), програш від ділення
   монотонний; на `α=0.5` буде ~0.85–0.90× від найкращого.
🟢 ЖИВЕ, якщо якесь `α<1` дає ≥1.05× проти `α=1` (означало б, що зміщення
   `sub87` на робочій точці вже важить більше за втрату семплів).
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
import jitanim                                                    # noqa: E402
import curvegui                                                   # noqa: E402
from base_real import cut_and_budget, closure_parts, W_           # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 24
ALPHA = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)

TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
IDX = {int(v): i for i, v in enumerate(TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
HAVE = {int(v): j for j, v in enumerate(SD)}

print(__doc__.split('📣')[1], flush=True)
acc = {a: [] for a in ALPHA}
for n, s0 in enumerate(list(IDX)[:NETS]):
    W = [torch.tensor(np.asarray(WZ[HAVE[s0]][k], np.float32), device=DEV)
         for k in range(32)]
    t1 = TT['T1'][IDX[s0]]
    t2 = TT['T2'][IDX[s0]]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        cl = closure_parts(wc, 8, DEV, extras=True)
        f_cut = sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
        f_full = float(len(W)) * W_ * W_
        # ✅ ЦІНА, а не семпли: v2 дорожчий на семпл у f_full/f_cut разів
        for a in ALPHA:
            n87 = max(64, int(round(a * npair)))
            nv2 = max(64, int(round((1 - a) * npair * f_cut / f_full)))
            se = []
            for r in range(DRAWS):
                parts, ws = [], []
                if a > 0:
                    g = torch.Generator(device=DEV).manual_seed(61_000 + 977 * r + n)
                    z = torch.randn(n87, W_, device=DEV, generator=g)
                    parts.append(jitanim._run_sub87(wc, n87, cl, DEV, 0,
                                                    z_ext=z).cpu().numpy())
                    ws.append(a)
                if a < 1:
                    # 🚨 НЕЗАЛЕЖНИЙ потік (інший seed) — у цьому вся суть питання
                    g2 = torch.Generator(device=DEV).manual_seed(85_000 + 613 * r + n)
                    z2 = torch.randn(nv2, W_, device=DEV, generator=g2)
                    parts.append(curvegui._v2(W, nv2, DEV, z2).double().cpu().numpy())
                    ws.append(1 - a)
                if len(parts) == 1:
                    se.append(parts[0])
                else:
                    se.append((parts, ))
            if len(parts) == 1:
                P = np.array(se)
                acc[a].append(float(np.mean((P - t1) * (P - t2))))
            else:
                A = np.array([s[0][0] for s in se])
                Bv = np.array([s[0][1] for s in se])
                # ОРАКУЛЬНА вага (підбір на тих самих даних ⟹ стеля)
                ws_ = np.linspace(0.0, 1.0, 101)
                m = [float(np.mean(((w * A + (1 - w) * Bv) - t1)
                                   * ((w * A + (1 - w) * Bv) - t2))) for w in ws_]
                acc[a].append(min(m))
    del W
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS} {s0}', flush=True)

base = sum(acc[1.0])
print(f'\n{NETS} мереж × {DRAWS} жеребів, крос-ціль, РІВНА ЗАГАЛЬНА ЦІНА, '
      f'НЕЗАЛЕЖНІ потоки, вага ОРАКУЛЬНА')
print(f'{"α (частка ціни на sub87)":>26s} {"MSE":>12s} {"×проти α=1":>12s}')
print('-' * 54)
for a in ALPHA:
    s = sum(acc[a])
    print(f'{a:26.2f} {s/NETS:12.4e} {base/s:11.4f}×')
best = max(ALPHA, key=lambda a: base / sum(acc[a]))
g = base / sum(acc[best])
print(f'\n🔑 найкраще α = {best:.2f}, виграш {g:.4f}× проти «увесь бюджет у sub87»')
print(f'🔑 ВИРОК: {"🟢 ЖИВЕ" if g >= 1.05 else "⛔ МЕРТВО — ділити бюджет не варто"}'
      f'  (гейт 1.05×; вага ОРАКУЛЬНА, реальність гірша)')
print(f'📊 БАЛ: 1.954e-07 → {1.954e-07/max(g,1e-9):.3e} у найкращому оракульному разі')
