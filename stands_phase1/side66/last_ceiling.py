"""СТЕЛЯ ВСЬОГО КЛАСУ: ЩО ЛИШИТЬСЯ, ЯКЩО ЗНАТИ m І s ОСТАННЬОГО ШАРУ ТОЧНО.

ЗВІДКИ. Уся конструкція σ-поправки несе нев'язку через 32 шари і впирається
в те, що нев'язка народжується заново. Але фінальна похибка в гаусовій моделі
виражається ЛИШЕ через моменти ОСТАННЬОГО шару:
    mean(relu(p₃₁)) − E[relu(p₃₁)] ≈ Φ(t)·δm₃₁ + φ(t)·δs₃₁
Нести через шари нічого не треба — треба знати два вектори на шарі 31.

ЩО МІРЯЄМО (усе оракульно ⟹ це СТЕЛІ, не методи)
    база            sub66
    −φ·δs           знята лише похибка σ останнього шару
    −Φ·δm           знята лише похибка середнього останнього шару
    −Φ·δm − φ·δs    знято обидва ⟹ ЛИШАЄТЬСЯ ТІЛЬКИ НЕГАУСОВІСТЬ
Останній рядок — тверда стеля всього класу «гаусова поправка до вибірки».
Якщо він дає, скажімо, 50×, шлях до 1e-8 у класі Є (питання лише як дізнатись
моменти). Якщо 3×, клас вичерпано і 1e-8 треба шукати деінде.

КОНТРОЛЬ
  * «raw − Φδm − φδs» проти ІСТИНИ, і окремо друкуємо, наскільки гаусова
    формула G(m,s) з ТОЧНИМИ моментами відтворює істину — це та сама
    негаусовість, виміряна незалежним способом; два числа мусять зійтись.
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
    raw = x.double().sum(0) / n2
    mt, st = mu_t[L_ - 2].double(), sg_t[L_ - 2].double()
    t = mt / st
    return raw, v, _Phi(t) * (m - mt), _phi(t) * (s - st), (mt * _Phi(t)
                                                           + st * _phi(t))


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keys = ('база', '−φδs', '−Φδm', '−Φδm−φδs')
    E = {k: [] for k in keys}
    gap = []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            for sd in SEEDS:
                raw, v, dm, ds, gform = run_one(wc, sd, dev, mu_t, sg_t)
                b = raw - v - t_last
                E['база'].append(b.cpu().numpy())
                E['−φδs'].append((b - ds).cpu().numpy())
                E['−Φδm'].append((b - dm).cpu().numpy())
                E['−Φδm−φδs'].append((b - dm - ds).cpu().numpy())
            gap.append(float(((gform - t_last) ** 2).mean()))
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    base = float((np.array(E['база']) ** 2).mean())
    print(f'\n{len(fs)} мереж × {len(SEEDS)} жеребів, істина {TRUTH_N}')
    print(f'{"варіант":28s} {"MSE":>11s} {"×":>9s}')
    print('-' * 50)
    for k in keys:
        m = float((np.array(E[k]) ** 2).mean())
        print(f'{k:28s} {m:11.4e} {base/m:9.3f}')
    print(f'\nконтроль негаусовості: MSE гаусової формули G(m_іст,s_іст) проти '
          f'істини = {np.mean(gap):.4e}')
    print('Якщо він близький до «−Φδm−φδs» — це та сама негаусовість, зміряна двічі.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
