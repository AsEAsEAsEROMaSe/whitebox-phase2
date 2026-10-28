"""СТЕЛЯ κ₃: чи вилікує треті кумулянти похибку замикання. (мета 1e-8)

ЗВІДКИ. Уся ціль зведена до одного числа: `r` (похибка замикання / шум вибірки)
з ~5 до 0.2. Замикання робить РІВНО ОДНЕ припущення — що `p` на кожному шарі
гаусів. Отже його похибка = негаусовість. Найнижчий її порядок — третій.

ПИТАННЯ, НА ЯКЕ ЦЕЙ СТЕНД ВІДПОВІДАЄ ОСТАТОЧНО. Якщо дати замиканню ІСТИННІ
треті (і четверті) кумулянти — наскільки впаде `r`? Це СТЕЛЯ: питання доставки
(як дізнатись κ₃ без істини) свідомо винесено за дужки. Якщо стеля <2× — уся
κ₃-лінія мертва незалежно від доставки, і ми більше про неї не думаємо.

ПОПРАВКИ ЕДЖВОРТА (виведені, не вгадані). Для X із (m, s, κ₃, κ₄), Y~N(m,s²):
    E[g(X)] ≈ E[g(Y)] + (κ₃/6)E[g'''(Y)] + (κ₄/24)E[g''''(Y)]
для g=relu:   g'''=δ',  g''''=δ''  ⟹  −κ₃·t·φ(t)/(6s²) + κ₄·(t²−1)·φ(t)/(24s³)
для g=relu²:  g'''=2δ,  g''''=2δ'  ⟹  +κ₃·φ(t)/(3s)    − κ₄·t·φ(t)/(12s²)
Позадіагональ лишається гаусовою (для неї потрібні СПІЛЬНІ κ₃ — тензор 256³).
Тобто це стеля саме МАРГІНАЛЬНОЇ поправки — найдешевшої з можливих.

ДРУГЕ ПИТАННЯ, БЕЗКОШТОВНО (ідея юзера про ударну хвилю / в'язкісні розв'язки).
Якщо злам ЛОКАЛЬНИЙ у глибині — «є шар, де відображення вироджується» — то
профіль негаусовості по шарах мусить мати ПІК. Якщо він гладкий, то дорогу
поправку немає куди поставити одну, і її доведеться розмазувати. Друкуємо
асиметрію й ексцес кожного шару.

КОНТРОЛІ
  * ✅ κ₃=κ₄=0 мусить відтворити гаусове замикання ДО ОСТАННЬОЇ ЦИФРИ;
  * одношаговий тест окремо від рекурсії: з ІСТИННИМИ (m,s,κ₃,κ₄) наскільки
    точніше передбачається E[relu(p_i)] — якщо тут виграшу нема, далі й дивитись
    нема сенсу;
  * істина власна, 4М свіжих семплів, накопичення у fp64 (скорочення порядків);
  * `r` міряється проти РЕАЛЬНОГО шуму вибірки на тих самих мережах.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import CHUNK, cut_net                       # noqa: E402
from born_source import _Phi, _phi                            # noqa: E402
from closure_precision import NQ, relu_cov0, relu_cov_gauss   # noqa: E402
from caustic_dirs import run_one                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
UPTO = 31
NSEED = 8
SEED0 = 400_000


def truth_cumulants(wc, dev, seed, n):
    """Істина: (m, s, κ₃, κ₄) кожного `p` і справжнє E[relu(p)]. fp64."""
    depth = len(wc)
    S = [[torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
          for w in wc[1:]] for _ in range(5)]
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < n:
        mch = min(CHUNK, n - done)
        z = torch.tensor(rng.standard_normal((mch // 2, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        x = torch.relu(x @ wc[0])
        acc[0] += x.sum(0).double()
        for i in range(1, depth):
            p = x @ wc[i]
            pd = p.double()
            q = pd
            for k in range(4):
                S[k][i - 1] += q.sum(0)
                q = q * pd
            x = torch.relu(p)
            xd = x.double()
            acc[i] += xd.sum(0)
            S[4][i - 1] += (xd * xd).sum(0)
        done += mch
    out = {}
    for i in range(1, depth):
        a1, a2, a3, a4 = (S[k][i - 1] / done for k in range(4))
        m = a1
        v = (a2 - m * m).clamp_min(1e-30)
        mu3 = a3 - 3.0 * m * a2 + 2.0 * m ** 3
        mu4 = a4 - 4.0 * m * a3 + 6.0 * m * m * a2 - 3.0 * m ** 4
        out[i] = (m, v.sqrt(), mu3, mu4 - 3.0 * v * v,
                  acc[i] / done, S[4][i - 1] / done)
    return out, [a / done for a in acc]


def relu_moments_edge(m, s, k3, k4):
    """(E[relu], E[relu²]) з поправками Еджворта по маргінальних κ₃, κ₄."""
    t = m / s
    Ph, ph = _Phi(t), _phi(t)
    e1 = m * Ph + s * ph
    e2 = (m * m + s * s) * Ph + m * s * ph
    if k3 is not None:
        e1 = e1 - k3 * t * ph / (6.0 * s * s)
        e2 = e2 + k3 * ph / (3.0 * s)
    if k4 is not None:
        e1 = e1 + k4 * (t * t - 1.0) * ph / (24.0 * s ** 3)
        e2 = e2 - k4 * t * ph / (12.0 * s * s)
    return e1, e2


def closure_edge(wc, nodes, wts, upto, cum, mode, prof=None):
    """Замикання з маргінальною поправкою Еджворта; повертає (m, s) шару upto.

    mode: 'gauss' — чиста гаусова гілка; 'zero' — гілка Еджворта з НУЛЬОВИМИ
    кумулянтами (контроль: мусить збігтися з 'gauss'); 'k3'; 'k34';
    'mean3' — κ₃+κ₄ ЛИШЕ в середнє, коваріація гаусова;
    'oracle' — ІСТИННІ E[relu] і Var[relu] на кожному шарі (стеля всього
    маргінального: лишається тільки позадіагональна негаусовість).
    """
    w0 = wc[0].double()
    m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    C_est = relu_cov0(wc[0])
    for i in range(1, upto + 1):
        Wd = wc[i].double()
        C_p = Wd.T @ (C_est @ Wd)
        m_p = m_est @ Wd
        s_p = C_p.diagonal().clamp_min(1e-30).sqrt()
        if i == upto:
            return m_p, s_p
        C_g, m_g = relu_cov_gauss(m_p, C_p, nodes, wts)
        if mode == 'gauss':
            C_est, m_est = C_g, m_g
            continue
        if mode == 'zero':
            k3 = k4 = torch.zeros_like(m_p)
        elif mode in ('prof', 'profshuf'):
            # κ₃ = γ₃(шар)·s³, κ₄ = γ₄(шар)·s⁴ — ЖОДНОЇ істини цієї мережі
            g3, g4 = prof[i]
            k3, k4 = g3 * s_p ** 3, g4 * s_p ** 4
        else:
            k3 = cum[i][2]
            k4 = cum[i][3] if mode != 'k3' else None
        e1, e2 = relu_moments_edge(m_p, s_p, k3, k4)
        if mode in ('mean3', 'zero', 'prof', 'profshuf'):
            C_est, m_est = C_g, e1
            continue
        d_g = C_g.diagonal().clamp_min(1e-30)
        d_n = (e2 - e1 * e1).clamp_min(1e-30)
        sc = (d_n / d_g).sqrt()
        C_est = (sc.unsqueeze(1) * C_g) * sc.unsqueeze(0)
        m_est = e1
    raise AssertionError


def main(nets=6, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    LS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 31)
    prof = {k: {i: [] for i in LS} for k in ('skew', 'kurt', 'e0', 'e3', 'e34')}
    KEYS = ('гаусове', 'контроль κ=0', '+κ₃', '+κ₃+κ₄',
            '+κ₃+κ₄ лише в середнє (ОРАКУЛ)',
            '★ПРОФІЛЬ γ(шар), LOO по мережах', '⛔профіль із перемішаними шарами')
    res = {k: [] for k in KEYS}
    noise, keep, spread = [], [], {i: [] for i in LS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            cum, _ = truth_cumulants(wc, dev, 90001 + n, TRUTH_N)
            g3 = {i: float((cum[i][2] / cum[i][1] ** 3).mean())
                  for i in range(1, UPTO)}
            g4 = {i: float((cum[i][3] / cum[i][1] ** 4).mean())
                  for i in range(1, UPTO)}
            keep.append((wc, cum, g3, g4))

            for i in LS:
                m, s, k3, k4, tru, _ = cum[i]
                sk = k3 / s ** 3
                prof['skew'][i].append(float(sk.mean()))
                prof['kurt'][i].append(float((k4 / s ** 4).mean()))
                spread[i].append(float(sk.std() / sk.mean().abs().clamp_min(1e-30)))
                sc = float(tru.abs().mean())
                for key, a, b in (('e0', None, None), ('e3', k3, None),
                                  ('e34', k3, k4)):
                    e1, _ = relu_moments_edge(m, s, a, b)
                    prof[key][i].append(float((e1 - tru).abs().mean()) / sc)

            mt = cum[UPTO][0]
            nz = []
            for j in range(NSEED):
                _, m_h, _ = run_one(wc, SEED0 + 1009 * j, dev)
                nz.append(float(((m_h - mt).abs() / cum[UPTO][1]).mean()))
            noise.append(float(np.mean(nz)))
        print(f'  істина мережі {n+1}/{len(fs)}, шум {noise[-1]:.3e}', flush=True)

    for n, (wc, cum, _, _) in enumerate(keep):
        oth = [k for j, k in enumerate(keep) if j != n]
        pr = {i: (float(np.mean([o[2][i] for o in oth])),
                  float(np.mean([o[3][i] for o in oth]))) for i in range(1, UPTO)}
        sh = {i: pr[UPTO - i] for i in range(1, UPTO)}
        mt, st = cum[UPTO][0], cum[UPTO][1]
        with torch.no_grad():
            for key, mode, p in (('гаусове', 'gauss', None),
                                 ('контроль κ=0', 'zero', None),
                                 ('+κ₃', 'k3', None), ('+κ₃+κ₄', 'k34', None),
                                 ('+κ₃+κ₄ лише в середнє (ОРАКУЛ)', 'mean3', None),
                                 ('★ПРОФІЛЬ γ(шар), LOO по мережах', 'prof', pr),
                                 ('⛔профіль із перемішаними шарами', 'profshuf', sh)):
                mc, _ = closure_edge(wc, nodes, wts, UPTO, cum, mode, p)
                res[key].append(float(((mc - mt).abs() / st).mean()))
        print(f'  замикання {n+1}/{len(keep)}  гаус {res["гаусове"][-1]:.3e} → '
              f'оракул {res[KEYS[4]][-1]:.3e} → профіль {res[KEYS[5]][-1]:.3e}',
              flush=True)

    print(f'\n{len(fs)} мереж, істина {TRUTH_N}, шум по {NSEED} жеребах')
    print('\n1. ЧИ Є ЗЛАМ У ГЛИБИНІ (профіль негаусовості `p` по шарах)')
    print(f'{"шар":>4s} {"асиметрія γ₃":>13s} {"розкид/мереж":>13s} '
          f'{"розкид/нейрон":>14s} {"ексцес γ₄":>10s} '
          f'{"похиб.E[relu] гаус":>19s} {"+κ₃":>9s} {"+κ₃+κ₄":>9s}')
    for i in LS:
        s3 = np.array(prof['skew'][i])
        print(f'{i:4d} {s3.mean():13.4f} {s3.std()/abs(s3.mean()):13.3f} '
              f'{np.mean(spread[i]):14.3f} {np.mean(prof["kurt"][i]):10.4f} '
              f'{np.mean(prof["e0"][i]):19.3e} {np.mean(prof["e3"][i]):9.3e} '
              f'{np.mean(prof["e34"][i]):9.3e}')

    nz = float(np.mean(noise))
    print(f'\n2. СТЕЛЯ: похибка замикання на шарі {UPTO} (частка істинного s), '
          f'шум вибірки {nz:.3e}')
    print(f'{"варіант":33s} {"похибка":>11s} {"r":>7s} {"β":>7s} {"виграш":>8s}')
    print('-' * 72)
    d = abs(np.mean(res['контроль κ=0']) / np.mean(res['гаусове']) - 1.0)
    print(f'✅ контроль κ=0 проти гаусового: розбіжність {d:.2e} (має бути ~0)')
    for k in KEYS:
        if k == 'контроль κ=0':
            continue
        e = float(np.mean(res[k]))
        r = e / nz
        b = 1.0 / (1.0 + r * r)
        print(f'{k:33s} {e:11.3e} {r:7.2f} {b:7.3f} {1.0 + 1.0/(r*r):8.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
