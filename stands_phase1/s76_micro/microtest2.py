"""№5б: ПРИТЯЖКА ВИХОДУ СУРОГАТА ДО ПОДОБИ x₂₈ ІЗ ТИХ ЖЕ 8к. (С76)

Продовження microtest.py за ідеєю юзера: згенерований x̂₂₈ підтягуємо до
емпіричного розподілу СПРАВЖНІХ x₂₈ тренувальної пачки:
  M1 — зшити СЕРЕДНЄ: x̂ + (mean(x₂₈ трен.) − mean(x̂));
  M2 — зшити середнє І розкид покомпонентно: стандартизувати x̂,
       одягнути емпіричні (μ, σ) кожного нейрона з 8к.
Міряємо ті самі метрики: bias сирої версії (велика пачка, крос-ціль),
R²cv (β=1) і R²cv з ОПТИМАЛЬНИМ скаляром β (інваріант афінних правок).

📣 ПЕРЕДБАЧЕННЯ:
  1) bias впаде сильно (перший момент зшито), але НЕ до нуля — хвіст
     нелінійний, форма розподілу лишається брехлива; і НЕ нижче шуму
     самих 8к (еталон подоби = ті самі семпли, циркулярність);
  2) R²cv(опт.β) НЕ зміниться ВЗАГАЛІ — афінна правка не рухає кореляцію;
  ⟹ двохрівневий виграш лишиться <1. ГЕЙТ ЖИТТЯ: bias/MSE < 1 І R²cv > 0.7.
"""
import glob
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, W_                                    # noqa: E402
from biassplit import truth_full                                         # noqa: E402
sys.path.insert(0, '/mnt/win_d/llm/White-Box/s76_micro')
from microtest import head_fwd, tail_fwd, draw, fit_lin, L_              # noqa: E402

DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:1'


def main(nets=8, dev=DEV):
    t0 = time.time()
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'№5б ПРИТЯЖКА СУРОГАТА ДО ПОДОБИ 8к · {nets} мереж, {dev}')
    print('📣 bias впаде, але не <1 MSE; R²cv(опт.β) не зміниться\n', flush=True)

    acc = {k: [] for k in ('RAW', 'M1', 'M2')}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            T1 = truth_full(W, dev, 810_000 + 7919 * n).float()
            T2 = truth_full(W, dev, 910_000 + 7919 * n).float()
            xa = draw(npair, 500_000 + n, dev)
            xb = draw(npair, 600_000 + n, dev)
            ya = head_fwd(wc, xa)
            yb = head_fwd(wc, xb)
            fb = tail_fwd(wc, yb)
            xbar = fb.double().mean(0).float()
            mse0 = float(torch.dot(xbar - T1, xbar - T2))
            mu_t, sd_t = ya.mean(0), ya.std(0).clamp_min(1e-12)
            x_big = draw(4 * npair, 700_000 + n, dev)
            g = fit_lin(xa, ya)
            yh_b, yh_big = g(xb), g(x_big)
            for name in ('RAW', 'M1', 'M2'):
                if name == 'RAW':
                    hb, hbig = yh_b, yh_big
                elif name == 'M1':
                    off = mu_t - yh_big.mean(0)
                    hb, hbig = yh_b + off, yh_big + off
                else:
                    mu_g = yh_big.mean(0)
                    sd_g = yh_big.std(0).clamp_min(1e-12)
                    hb = (yh_b - mu_g) / sd_g * sd_t + mu_t
                    hbig = (yh_big - mu_g) / sd_g * sd_t + mu_t
                tb = tail_fwd(wc, torch.relu(hb))
                tbig = tail_fwd(wc, torch.relu(hbig))
                mb = tbig.double().mean(0).float()
                bias_r = float(torch.dot(mb - T1, mb - T2)) / max(mse0, 1e-30)
                fc = fb - fb.mean(0)
                tc = tb - tb.mean(0)
                vf = float((fc ** 2).sum())
                # β=1 і оптимальний скаляр β
                r2_b1 = 1.0 - float(((fc - tc) ** 2).sum()) / vf
                cov = float((fc * tc).sum())
                vt = float((tc ** 2).sum())
                r2_op = cov * cov / max(vt * vf, 1e-30)
                acc[name].append((bias_r, r2_b1, r2_op))
        del W, wc, xa, xb, ya, yb, x_big
        torch.cuda.empty_cache()
        print(f'  {n+1}/{nets} ok  [{time.time()-t0:.0f}s]', flush=True)

    print(f'\n{"варіант":>8} {"bias/MSE":>10} {"R²cv(β=1)":>10} {"R²cv(опт)":>10}')
    for name in ('RAW', 'M1', 'M2'):
        a = np.array(acc[name])
        print(f'{name:>8} {np.mean(a[:,0]):>10.2f} {np.mean(a[:,1]):>10.3f} '
              f'{np.mean(a[:,2]):>10.3f}')
    print(f'   час {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
