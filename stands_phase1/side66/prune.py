"""АНАЛІТИЧНЕ ОБРІЗАННЯ ЗАМІСТЬ ПІЛОТА: чи бачить формула тих самих мертвих? (С80)

## ЗВІДКИ
Сабміт витрачає **128 пар крізь ПОВНУ мережу** (пілот) лише щоб дізнатись, які
нейрони ніколи не відкриваються, і викинути їх. Це ~5% усього, що є на вибірку.
🔑 АЛЕ ця сама величина вже є у формулі задарма: `f_cl = Φ(m/s)` з
`closure_parts(extras=True)` — це і є аналітична ЙМОВІРНІСТЬ, що нейрон
відкриється. Якщо формула бачить тих самих мертвих, пілот можна прибрати,
а звільнені ~5% віддати вибірці ⟹ ~4.8% балу (для порівняння: увесь приріст
`sub91` = 1.1%).

## ⛔ ЧОГО НЕ ВИЙДЕ (перевірено думкою, щоб не витрачати прогін)
Точний алгебраїчний тест «усі вхідні ваги ≤0» марний: після ReLU `a≥0`, тож
умова достатня, але її ймовірність `2^−256`. Мертвість породжує НЕ знак ваг,
а те, що досяжний конус вужчий за додатний ортант ⟹ тест мусить бути
статистичним.

## ЩО МІРЯЄМО (24 мережі, шари 1..31)
Три маски «живий/мертвий» на кожен нейрон:
  1. 🟠**ПІЛОТ** — як у сабміті: `max(p) > 0` на 128 антитетичних парах;
  2. 🟢**ФОРМУЛА** — `Φ(m/s) > t`, поріг `t` свіпається;
  3. ✅**ІСТИНА** — великий МК: нейрон живий, якщо `E[a] > 0`.
Далі:
  * скільки кожен метод обрізає (від цього залежить ціна семпла);
  * 🚨**НЕБЕЗПЕЧНА ПОМИЛКА**: обрізали живого. Міряємо не штуки, а ВТРАЧЕНУ
    МАСУ `‖E[a] обрізаних живих‖ / ‖E[a]‖` — саме вона псує прогноз;
  * при якому `t` формула дає ту саму частку обрізання, що й пілот.

⛔ ПЛОМБА: `t=0` ⟹ формула не обрізає нікого, втрачена маса РІВНО 0.
📣 ПЕРЕДБАЧЕНО: формула БАЧИТЬ мертвих (це різкий поділ, а не тонкий хвіст),
   але поріг доведеться брати консервативнішим за пілотів, бо `Φ` спирається
   на гаусову форму, а мертвість живе САМЕ у хвості, де гаус найгірший.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/prune.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import closure_parts                              # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, W_, KS = 32, 256, 31
PILOT, TRUTH, CH = 128, 1 << 18, 8192
TH = [0.0, 1e-12, 1e-9, 1e-6, 1e-5, 1e-4, 1e-3, 3.9e-3, 1e-2]
print(__doc__.split('Запуск')[0].split('## ЩО')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
fs = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

pil_cut, tru_dead = [], []
an_cut = {t: [] for t in TH}
an_lost = {t: [] for t in TH}          # 🚨 втрачена маса живих
pil_lost = []
for n, f in enumerate(fs):
    W = [torch.tensor(np.asarray(WZ[have[int(np.load(f)['seed'])]][q],
                                 np.float32), device=DEV) for q in range(L_)]
    with torch.no_grad():
        # 🟢 ФОРМУЛА на ПОВНИХ вагах: f_cl[i-1] = Φ(m/s) шару i
        _, _, f_cl, _ = closure_parts(W, KS, DEV, extras=True)

        # 🟠 ПІЛОТ як у сабміті
        g = torch.Generator(device=DEV).manual_seed(31_000 + 17 * n)
        zp = torch.randn(PILOT, W_, device=DEV, generator=g)
        a = torch.cat([zp, -zp], 0)
        pil = []
        for i in range(L_):
            p = a @ W[i]
            if i < L_ - 1:
                pil.append((p.max(0).values > 0.0))
            a = torch.relu(p)

        # ✅ ІСТИНА: E[a] по великому МК
        acc = [torch.zeros(W_, dtype=torch.float64, device=DEV)
               for _ in range(L_ - 1)]
        done = 0
        while done < TRUTH:
            k = min(CH, TRUTH - done)
            z = torch.randn(k, W_, device=DEV, generator=g)
            x = torch.cat([z, -z], 0)
            for i in range(L_ - 1):
                x = torch.relu(x @ W[i])
                acc[i] += x.double().sum(0)
            done += k
        E = [acc[i] / (2.0 * done) for i in range(L_ - 1)]
        live = [E[i] > 0 for i in range(L_ - 1)]
        tot = float(sum(float(e.norm() ** 2) for e in E)) ** 0.5

        tru_dead.append(float(np.mean([1.0 - float(l.double().mean())
                                       for l in live])))
        pil_cut.append(float(np.mean([1.0 - float(p.double().mean())
                                      for p in pil])))
        # 🚨 маса, яку пілот зрізав помилково
        ml = sum(float((E[i] * (~pil[i] & live[i])).norm() ** 2)
                 for i in range(L_ - 1)) ** 0.5
        pil_lost.append(ml / max(tot, 1e-30))

        for t in TH:                     # 🟢 формула, свіп порога
            am = [f_cl[i] > t for i in range(KS)]
            an_cut[t].append(float(np.mean([1.0 - float(m.double().mean())
                                            for m in am])))
            ml = sum(float((E[i] * (~am[i] & live[i])).norm() ** 2)
                     for i in range(KS)) ** 0.5
            an_lost[t].append(ml / max(tot, 1e-30))
    del W
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж, шари 1..{KS}, істина {2*TRUTH} семплів\n')
print(f'✅ІСТИННО МЕРТВИХ: {100*np.mean(tru_dead):.2f}%')
print(f'🟠ПІЛОТ обрізає:   {100*np.mean(pil_cut):.2f}%   '
      f'🚨втрачена маса живих {100*np.mean(pil_lost):.4f}%\n')
print(f'{"поріг t":>10s} {"🟢формула обрізає":>18s} {"🚨втрачена маса":>16s} '
      f'{"проти пілота":>13s}')
print('-' * 62)
for t in TH:
    print(f'{t:10.1e} {100*np.mean(an_cut[t]):17.2f}% '
          f'{100*np.mean(an_lost[t]):15.4f}% '
          f'{np.mean(an_cut[t])/max(np.mean(pil_cut),1e-30):12.3f}×')
print(f'\n⛔ПЛОМБА t=0: втрачена маса {100*np.mean(an_lost[0.0]):.6f}% (0 = ок)')
print('📊 БАЛ 1.954e-07 — ворота ідеї; виграш дасть лише повний прогін')
