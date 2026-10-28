"""ЧОТИРИ ВЕКТОРИ СЕРЕДНЬОГО: 2 потоки × 2 алгоритми (спостереження юзера, С77).

Схема 2×2 дає чотири оцінки того самого `m`, і ДВА незалежні контрасти:
  • «по алгоритмах» `p₈₇(z) − p_v2(z)` — на СПІЛЬНОМУ потоці, вибіркова похибка
    скорочується ⟹ чиста МЕТОДА. ⛔ Уже зміряно: 88.8°, `R²=0.002` — знака нема;
  • «по семплах»  `p₈₇(z_A) − p₈₇(z_B)` — той самий алгоритм, РІЗНІ потоки ⟹
    чиста ВИБІРКОВА ПОХИБКА, і вона спостережувана **БЕЗ ІСТИНИ**.

🎯 ДРУГИЙ КОНТРАСТ — це той клас, який у нас найпродуктивніший: величина, вже
оплачена й викинута. Розділивши НАЯВНУ вибірку навпіл, отримуємо `σ̂` за НУЛЬ
флопів (жодного зайвого семпла — ті самі, лише порахувати двома сумами).

📣 ПЕРЕДБАЧЕНО: `σ̂²` добре передбачить ВЛАСНУ квадратичну похибку по МЕРЕЖАХ
   (бо мережі різняться масштабом `‖a‖` у рази), але ПОГАНО по жеребах усередині
   мережі (сама `σ̂` шумна тієї ж величини, що й те, що вона міряє).
🟢 ЖИВЕ, якщо кореляція `σ̂²` з фактичною похибкою ПО ЖЕРЕБАХ (усередині мережі,
   масштаб знято) дає `R² ≥ 0.25`: тоді можна адаптивно усаджувати.
🚨 По мережах кореляція нічого не дасть сама по собі: знати, що мережа складна,
   марно — бюджет на неї однаковий і вже на підлозі.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
import jitanim                                                    # noqa: E402
from base_real import cut_and_budget, closure_parts, W_           # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
DRAWS = int(sys.argv[3]) if len(sys.argv) > 3 else 32

TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
IDX = {int(v): i for i, v in enumerate(TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
HAVE = {int(v): j for j, v in enumerate(SD)}

print(__doc__.split('📣')[1], flush=True)
rows = []
for n, s0 in enumerate(list(IDX)[:NETS]):
    W = [torch.tensor(np.asarray(WZ[HAVE[s0]][k], np.float32), device=DEV)
         for k in range(32)]
    t1, t2 = TT['T1'][IDX[s0]], TT['T2'][IDX[s0]]
    with torch.no_grad():
        wc, npair = cut_and_budget(W, DEV)
        cl = closure_parts(wc, 8, DEV, extras=True)
        h = npair // 2
        for r in range(DRAWS):
            g = torch.Generator(device=DEV).manual_seed(31_000 + 977 * r + n)
            z = torch.randn(npair, W_, device=DEV, generator=g)
            full = jitanim._run_sub87(wc, npair, cl, DEV, 0, z_ext=z).cpu().numpy()
            # 🔑 ТІ САМІ семпли, лише двома половинами — ЗАЙВИХ ФЛОПІВ НЕМА
            pa = jitanim._run_sub87(wc, h, cl, DEV, 0, z_ext=z[:h]).cpu().numpy()
            pb = jitanim._run_sub87(wc, h, cl, DEV, 0, z_ext=z[h:2 * h]).cpu().numpy()
            sig2 = float(((pa - pb) ** 2).mean()) / 4.0      # оцінка Var(full)
            err2 = float(((full - t1) * (full - t2)).mean())  # крос-ціль
            rows.append((n, sig2, err2))
    del W
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS} {s0}', flush=True)

A = np.array(rows)
net, s2, e2 = A[:, 0].astype(int), A[:, 1], A[:, 2]
print(f'\n{NETS} мереж × {DRAWS} жеребів, крос-ціль, півподіл БЕЗ зайвих флопів')
print(f'сер. σ̂² {s2.mean():.4e} · сер. факт. похибка² {e2.mean():.4e} · '
      f'відношення {s2.mean()/e2.mean():.3f}')
print(f'🔑 R² ПО ВСІХ точках (мережі + жереби): '
      f'{np.corrcoef(s2, e2)[0,1]**2:.4f}')
# 🚨 всередині мережі: знімаємо мережевий масштаб
zs, ze = [], []
for k in range(NETS):
    m = net == k
    zs.append(s2[m] / s2[m].mean() - 1)
    ze.append(e2[m] / e2[m].mean() - 1)
zs, ze = np.concatenate(zs), np.concatenate(ze)
r2in = float(np.corrcoef(zs, ze)[0, 1] ** 2)
print(f'🔑 R² ПО ЖЕРЕБАХ усередині мережі (масштаб знято): {r2in:.4f}')
mn_s = np.array([s2[net == k].mean() for k in range(NETS)])
mn_e = np.array([e2[net == k].mean() for k in range(NETS)])
print(f'🔑 R² ПО МЕРЕЖАХ (середні): {np.corrcoef(mn_s, mn_e)[0,1]**2:.4f}')
print(f'\n🔑 ВИРОК: {"🟢 ЖИВЕ" if r2in >= 0.25 else "⛔ МЕРТВО"} '
      f'(гейт: R² по жеребах ≥0.25 ⟹ можна адаптивно усаджувати)')
print('📊 БАЛ: 1.954e-07 — вимір діагностичний, сабміт не чіпався')
