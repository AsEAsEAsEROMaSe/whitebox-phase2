"""ЧИ Є ПОХИБКА ЗАМИКАННЯ СКІНЧЕННОРОЗМІРНИМ ЕФЕКТОМ ~1/√N. (ідея юзера: TAP)

ТВЕРДЖЕННЯ. Для ПРЯМОЇ мережі з незалежними по шарах вагами гаусове замикання
ТОЧНЕ в границі нескінченної ширини (гаусова еквівалентність / DMFT). Член
Онсаґера виправляє вплив вузла на себе через сусідів — а в прямому проході
кожна `W` вживається рівно раз, зворотного шляху нема ⟹ ми ВЖЕ на тому
порядку, і вся негаусовість є ефектом скінченної ширини.

ЯКЩО ЦЕ ТАК, то похибка мусить спадати як `N^(−1/2)` (або `N^(−1)`), і це
ОДНИМ фактом пояснює три сьогоднішні виміри:
  * чому ряд Мелера K=4 не змінив нічого (внесок не в довжині ряду);
  * чому коефіцієнти нев'язки залежать від РЕАЛІЗАЦІЇ ваг, а спільного базису
    між мережами нема (`nwidth.py`, моди класу по ~1/6);
  * чому `r_s` не зсунути: якщо похибка замикання ~`N^(−α)`, а шум вибірки
    ~`n^(−1/2)`, їх відношення задане ЗАДАЧЕЮ, а не якістю методу.

ЩО РАХУЄМО. Синтетичні мережі He-ініціалізації різної ширини, глибина 16.
Пошарово:
    асиметрія `p`                                     (міра негаусовості)
    одношагова похибка `|E_гаус[relu] − E_іст[relu]| / s`  (з ІСТИННИХ моментів)
    накопичена похибка замикання по `s`               (уся рекурсія)
і підганяємо показник `α` у `похибка ≈ c·N^(−α)`.

КОНТРОЛІ
  * ⛔ мережа з ОДНОГО шару: там `p = zW₀` СТРОГО гаусів ⟹ похибка мусить бути
    на рівні шуму істини, а не степеневого закону;
  * похибка істини друкується окремо (дві половини) — щоб не сплутати спад
    сигналу зі спадом шуму;
  * дві незалежні мережі на ширину, друкуємо розкид.
"""
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from closure_precision import NQ, relu_cov0, relu_cov_gauss   # noqa: E402
from born_source import _Phi, _phi                            # noqa: E402
from sub66_torch import INV_SQRT_2PI                          # noqa: E402

DEV = 'cuda:3'
WIDTHS = (48, 96, 192, 384, 768)
DEPTH = 16
NSAMP = 1_048_576
CH = 32768
NREP = 2


def make_net(width, depth, seed, dev):
    g = torch.Generator(device='cpu').manual_seed(seed)
    sc = math.sqrt(2.0 / width)
    return [(torch.randn(width, width, generator=g) * sc).to(dev)
            for _ in range(depth)]


def truth(wc, dev, seed, n, half):
    """(m, s, асиметрія) `p` кожного шару і справжнє E[relu(p)]."""
    d = len(wc)
    S = [[torch.zeros(wc[0].shape[1], dtype=torch.float64, device=dev)
          for _ in range(1, d)] for _ in range(4)]
    rng = np.random.default_rng(int(seed) + 7919 * half)
    done = 0
    while done < n:
        mch = min(CH, n - done)
        z = torch.randn(mch // 2, wc[0].shape[0], device=dev,
                        generator=torch.Generator(device=dev).manual_seed(
                            int(rng.integers(1 << 30))))
        x = torch.relu(torch.cat([z, -z], 0) @ wc[0])
        for i in range(1, d):
            p = (x @ wc[i]).double()
            S[0][i - 1] += p.sum(0)
            S[1][i - 1] += (p * p).sum(0)
            S[2][i - 1] += (p * p * p).sum(0)
            S[3][i - 1] += torch.relu(p).sum(0)
            x = torch.relu(p.float())
        done += mch
    out = {}
    for i in range(1, d):
        a1, a2, a3, r = (S[k][i - 1] / done for k in range(4))
        v = (a2 - a1 * a1).clamp_min(1e-30)
        mu3 = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
        out[i] = (a1, v.sqrt(), mu3 / v.pow(1.5), r)
    return out


def main(dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)
    LS = (1, 2, 4, 8, 15)
    res = {w: {i: [[], [], [], []] for i in LS} for w in WIDTHS}

    for width in WIDTHS:
        for rep in range(NREP):
            with torch.no_grad():
                wc = make_net(width, DEPTH, 1000 + 31 * rep + width, dev)
                T = [truth(wc, dev, 5000 + rep, NSAMP // 2, h) for h in (0, 1)]
                # замикання по всій глибині
                m_e = (wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
                       ) * INV_SQRT_2PI
                C_e = relu_cov0(wc[0])
                sclo = {}
                for i in range(1, DEPTH):
                    Wd = wc[i].double()
                    C_p = Wd.T @ (C_e @ Wd)
                    m_p = m_e @ Wd
                    sclo[i] = (m_p, C_p.diagonal().clamp_min(1e-30).sqrt())
                    if i < DEPTH - 1:
                        C_e, m_e = relu_cov_gauss(m_p, C_p, nodes, wts)
                for i in LS:
                    mt = 0.5 * (T[0][i][0] + T[1][i][0])
                    st = 0.5 * (T[0][i][1] + T[1][i][1])
                    sk = 0.5 * (T[0][i][2] + T[1][i][2])
                    rt = 0.5 * (T[0][i][3] + T[1][i][3])
                    t = mt / st
                    e1 = mt * _Phi(t) + st * _phi(t)
                    res[width][i][0].append(float(sk.abs().mean()))
                    res[width][i][1].append(float(((e1 - rt).abs() / st).mean()))
                    res[width][i][2].append(
                        float(((sclo[i][1] - st).abs() / st).mean()))
                    res[width][i][3].append(float(
                        ((T[0][i][3] - T[1][i][3]).abs() / st).mean()))
        print(f'  ширина {width}: асим(15)={np.mean(res[width][15][0]):.4f} '
              f'одношаг={np.mean(res[width][15][1]):.3e} '
              f'накопич={np.mean(res[width][15][2]):.3e} '
              f'✅шум={np.mean(res[width][15][3]):.3e}', flush=True)

    for nm, idx in (('АСИМЕТРІЯ p', 0), ('ОДНОШАГОВА похибка E[relu]', 1),
                    ('НАКОПИЧЕНА похибка замикання по s', 2)):
        print(f'\n{nm}')
        print(f'{"шар":>4s}' + ''.join(f'{"N="+str(w):>11s}' for w in WIDTHS)
              + f'{"α":>8s}')
        for i in LS:
            v = np.array([np.mean(res[w][i][idx]) for w in WIDTHS])
            al = -np.polyfit(np.log(WIDTHS), np.log(np.maximum(v, 1e-12)), 1)[0]
            print(f'{i:4d}' + ''.join(f'{x:11.3e}' for x in v) + f'{al:8.2f}')
    print('\n✅ШУМ ІСТИНИ (дві половини) — щоб не сплутати спад сигналу зі спадом шуму')
    print(f'{"шар":>4s}' + ''.join(f'{"N="+str(w):>11s}' for w in WIDTHS))
    for i in LS:
        v = [np.mean(res[w][i][3]) for w in WIDTHS]
        print(f'{i:4d}' + ''.join(f'{x:11.3e}' for x in v))


if __name__ == '__main__':
    main()
