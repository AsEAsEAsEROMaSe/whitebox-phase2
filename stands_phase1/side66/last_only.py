"""ЧИ ПОТРІБЕН v-ЛАНЦЮГ ВЗАГАЛІ, ЯКЩО Є МОМЕНТИ ОСТАННЬОГО ШАРУ. (мета 1e-8)

ЗВІДКИ — аномалія в `last_ceiling.py`. Там міряно `(raw − v) − Φδm − φδs` = 29.3×.
Але за гаусовою формулою `Φ(t)δm + φ(t)δs` — це ПОВНА похибка `raw`. Тоді
віднімання ще й `v` — ЗАЙВЕ, і 29.3× — це те, що лишилось ПІСЛЯ зайвого
віднімання. Отже без `v` може бути кратно краще, а весь 31-шаровий ланцюг
взагалі не потрібен.

ЩО МІРЯЄМО (усе оракульно ⟹ стелі)
    raw − v                       база
    raw − Φδm − φδs               БЕЗ ланцюга, лише моменти шару 31  ★
    raw − v − Φδm − φδs           те, що міряв last_ceiling (29.3×)
    raw − Φδm                     лише середнє
Якщо «без ланцюга» ≫ 29.3×, то конструкція має бути принципово іншою:
не нести похибку через 31 шар, а ОДИН РАЗ виправити моменти на виході.
Тоді вся задача зводиться до точності замикання на ОСТАННЬОМУ шарі, і
арифметика вже порахована: покращення замикання у 10× дає 7.3×, у 30× — 16.6×.

КОНТРОЛЬ
  * друкуємо ‖v‖² окремо: якщо 29.3× ≈ база/‖v‖², то гіпотеза підтверджена
    і саме `v` є тим, що лишилось;
  * істина власна, 4М свіжих семплів.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (GATE_N, INV_SQRT_2PI, MIN_PAIRS,     # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments, _Phi, _phi             # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)


def run_one(wc, seed, dev, mu_t, sg_t):
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
    npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used) / (2.0 * per_cut)))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    x = torch.cat([z, -z], 0)
    n2 = float(x.shape[0])
    cov = (x.T @ x) / n2
    ev, evec = torch.linalg.eigh(cov.double())
    wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
    x = torch.relu(x @ (wh @ wc[0]))
    sg0 = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = (x.double().sum(0) / n2 - (sg0 * INV_SQRT_2PI).double())
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        v = (v @ wc[i].double()) * g
        if i == L_ - 1:
            pd = p.double()
            m = pd.sum(0) / n2
            s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
    mt, st = mu_t[L_ - 2].double(), sg_t[L_ - 2].double()
    t = mt / st
    return (x.double().sum(0) / n2, v,
            _Phi(t) * (m - mt), _phi(t) * (s - st))


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keys = ('raw − v (база)', 'raw − Φδm − φδs ★', 'raw − v − Φδm − φδs',
            'raw − Φδm', 'raw (без нічого)')
    E = {k: [] for k in keys}
    vn = []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            for sd in SEEDS:
                raw, v, dm, ds = run_one(wc, sd, dev, mu_t, sg_t)
                E['raw − v (база)'].append((raw - v - t_last).cpu().numpy())
                E['raw − Φδm − φδs ★'].append((raw - dm - ds - t_last).cpu().numpy())
                E['raw − v − Φδm − φδs'].append(
                    (raw - v - dm - ds - t_last).cpu().numpy())
                E['raw − Φδm'].append((raw - dm - t_last).cpu().numpy())
                E['raw (без нічого)'].append((raw - t_last).cpu().numpy())
                vn.append(float((v ** 2).mean()))
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    base = float((np.array(E['raw − v (база)']) ** 2).mean())
    print(f'\n{len(fs)} мереж × {len(SEEDS)} жеребів, істина {TRUTH_N}')
    print(f'{"варіант":26s} {"MSE":>11s} {"×":>9s}')
    print('-' * 50)
    for k in keys:
        m = float((np.array(E[k]) ** 2).mean())
        print(f'{k:26s} {m:11.4e} {base/m:9.3f}')
    print(f'\n‖v‖² (сер) = {np.mean(vn):.4e}')
    print('Якщо «raw − v − Φδm − φδs» ≈ ‖v‖², то v і є тим, що лишилось,')
    print('і правильна конструкція — БЕЗ ланцюга, лише моменти шару 31.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
