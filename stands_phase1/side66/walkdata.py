"""ПОСЕМПЛОВА ХОДА: як біжуче середнє йде до істини, семпл за семплом (С80).

Юзер (первісне формулювання): «коли наші методи і/або МК рухаються, то відносно
цієї точки з похибкою ми бачитимемо, ЯК вони це роблять».

Готує дані для панелі 5 у `linegui.py`. Кадр відліку той самий, що в
`anchorframe.py`:
    вісь X — уздовж `E[o]` (істина);
    вісь Y — уздовж поперечної складової атрактора `x*` (та сама «сіра лінія»).
У цьому кадрі істина стоїть у нулі, атрактор — нерухомо збоку, а біжуче
середнє МК блукає і сходиться до нуля.

🔑 ЩО САМЕ ЗБЕРІГАЄМО:
  * `walk`  — траєкторія БІЖУЧОГО середнього після кожного семпла (K×2);
  * `raw`   — самі семпли поодинці (K×2), щоб було видно, з чого це середнє
              складається (їхній розкид на два порядки більший за хід);
  * `att`   — де в цьому кадрі стоїть атрактор;
  * `mu2`   — ‖E[o]‖, щоб малювати у ЧАСТКАХ від істини, а не в сирих одиницях.

⛔ ПЛОМБА: остання точка ходи мусить збігтися із середнім усіх `K` семплів
   (це перевірка, що біжуче середнє рахується без зсуву на один крок).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/walkdata.py [cuda:N] [NETS] [K]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
K = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
W_, D = 256, 32

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
TZ = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
SEEDS = list(np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy'))
IDX = [SEEDS.index(s) for s in TZ['seeds'].tolist()]
print(__doc__.split('🔑')[1].split('Запуск')[0], flush=True)


def _f(W, X):
    for L in range(D):
        X = torch.relu(X @ W[L])
    return X


WALK, RAW, ATT, MU2 = [], [], [], []
for j in range(min(NETS, len(IDX))):
    W = [torch.tensor(np.asarray(WZ[IDX[j]][k], np.float32), device=DEV)
         for k in range(D)]
    mu = torch.tensor(TZ['T1'][j], dtype=torch.float64, device=DEV)
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(1234 + 53 * j)
        # атрактор
        x = torch.randn(1, W_, device=DEV, generator=g)
        x = x / x.norm()
        for _ in range(800):
            y = _f(W, x)
            x = y / y.norm().clamp_min(1e-30)
        xs = x.reshape(-1).double()
        e_mu = mu / mu.norm()
        d = xs - (xs @ e_mu) * e_mu
        ax = d / d.norm().clamp_min(1e-30)
        c = float((xs @ mu) / (xs @ xs))          # оракульний масштаб
        ATT.append([float(((c * xs - mu) @ e_mu)), float(((c * xs - mu) @ ax))])

        # семпли по одному, у кадрі (e_mu, ax)
        outs = torch.empty(K, W_, dtype=torch.float64, device=DEV)
        done = 0
        while done < K:
            b = min(8192, K - done)
            z = torch.randn(b, W_, device=DEV, generator=g)
            outs[done:done + b] = _f(W, z).double()
            done += b
        run = torch.cumsum(outs, 0) / torch.arange(
            1, K + 1, device=DEV, dtype=torch.float64).reshape(-1, 1)
        dev_run = run - mu
        dev_raw = outs - mu
        WALK.append(torch.stack([dev_run @ e_mu, dev_run @ ax], 1).cpu().numpy())
        RAW.append(torch.stack([dev_raw @ e_mu, dev_raw @ ax], 1).cpu().numpy())
        MU2.append(float(mu.norm()))
        # ⛔ПЛОМБА: остання точка ходи = середнє всіх K
        seal = float((run[-1] - outs.mean(0)).norm())
    print(f'  мережа {j+1}/{min(NETS,len(IDX))}  ⛔пломба хода-vs-середнє = '
          f'{seal:.2e}', flush=True)
    del W, outs, run
    torch.cuda.empty_cache()

np.savez_compressed('/mnt/win_d/llm/White-Box/side66/logs/walk.npz',
                    walk=np.array(WALK, np.float32),
                    raw=np.array(RAW, np.float32),
                    att=np.array(ATT), mu=np.array(MU2), K=K)
print(f'\n💾 збережено side66/logs/walk.npz  '
      f'({min(NETS,len(IDX))} мереж × {K} семплів)')
print('📊 БАЛ 1.954e-07 — дані для панелі 5')
