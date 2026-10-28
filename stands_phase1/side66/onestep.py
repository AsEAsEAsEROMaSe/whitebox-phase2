"""ОДНОКРОКОВЕ ЗАМИКАННЯ — ЧИ ПРИБИРАЄ ВОНО НАКОПИЧЕННЯ ПОХИБКИ. (мета 1e-8)

ЗВІДКИ. `closure_precision.py`: рекурсивне замикання точне на шарі 1
(зам/виб 0.13, ρ(δs)=0.9917), але його похибка НАКОПИЧУЄТЬСЯ — 3.7e-3 на
шарі 2 і 2.0e-2 на шарі 8, бо гаусове припущення годує саме себе 31 раз.
`stage_gain.py`: через це замикання дає 1.157× (шари 1..3) і починає ШКОДИТИ
з шару 4, тоді як оракульна стеля 2.855×.

ІДЕЯ. Не годувати рекурсію собою. На кожному шарі брати моменти `p_{i-1}`
З ВИБІРКИ і робити РІВНО ОДИН крок гаусового замикання:
    C_1s(x_{i-1}) = коваріація relu(p) для гаусового p ~ N(m̂_{i-1}, Ĉ_{i-1})
    s_1s(p_i)     = √diag(Wᵀ C_1s W)
Тоді похибка = похибка ОДНОГО кроку (≈3.7e-3, стала по глибині), а не
накопичена. Проти вибіркового шуму 5.7e-3 це дало б r≈0.65 ⟹ ρ≈0.84.

🚨 ПАСТКА, ЯКУ ОБХОДИМО ЯВНО. Підставити вибіркову коваріацію ТОГО САМОГО
шару — тавтологія: вибіркова Ĉ(p_i) = Wᵀ Ĉ(x_{i-1}) W ТОЧНО, тож δs вийшла б
тотожним нулем. Тому беремо шар i−1 і робимо крок через relu, де гаусове
припущення нетривіальне.

⚠️ ЧОГО БОЇМОСЯ. `s_1s` тепер ЗАЛЕЖИТЬ від вибірки ⟹ частина вибіркового шуму
просочується в оцінку, і δs може «з'їсти сама себе». Саме це й міряємо:
ρ(δs_однокрок, δs_істинна) проти ρ(δs_рекурсія, δs_істинна) на тих самих шарах.

КОНТРОЛІ
  * друкуємо ОБИДВА варіанти поруч — рекурсію і однокрок, на тих самих мережах;
  * тавтологічний контроль: `s` з ВИБІРКОВОЇ Ĉ того самого шару мусить дати
    рівно δs = 0 (стовпець «тавт») — якщо не нуль, вада реалізації;
  * істина власна, 4М свіжих семплів.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (INV_SQRT_2PI, MIN_PAIRS, PILOT_PAIRS,  # noqa: E402
                         TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments                         # noqa: E402
from closure_precision import relu_cov0, relu_cov_gauss, NQ   # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:2'
TRUTH_N = 4_194_304
SHOW = (1, 2, 3, 4, 6, 8, 12, 16, 24, 31)


def main(nets=4, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    tab = {}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            _, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)

            rng = np.random.default_rng(12345)
            rng.standard_normal((PILOT_PAIRS, W_))
            per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
            used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
            npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used)
                                       / (2.0 * per_cut)))
            z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                             device=dev)
            x = torch.cat([z, -z], 0)
            n2 = float(x.shape[0])
            cov = (x.T @ x) / n2
            ev, evec = torch.linalg.eigh(cov.double())
            wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
            x = torch.relu(x @ (wh @ wc[0]))

            # рекурсивне замикання (як у closure_precision) — для порівняння
            w0 = wc[0].double()
            m_rec = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
            C_rec = relu_cov0(wc[0])

            Cx_prev = None                 # вибіркова коваріація x_{i-1}
            mp_prev = Cp_prev = None       # вибіркові моменти p_{i-1}
            for i in range(1, L_):
                Wd = wc[i].double()
                xd = x.double()
                mx = xd.sum(0) / n2
                Cx = (xd.T @ xd) / n2 - torch.outer(mx, mx)

                p = x @ wc[i]
                pd = p.double()
                mp = pd.sum(0) / n2
                Cp = (pd.T @ pd) / n2 - torch.outer(mp, mp)
                sh = Cp.diagonal().clamp_min(1e-30).sqrt()
                st = sg_t[i - 1].double()

                # 1) рекурсія
                C_p_rec = Wd.T @ (C_rec @ Wd)
                s_rec = C_p_rec.diagonal().clamp_min(1e-30).sqrt()
                # 2) однокрок: гаусове замикання від ВИБІРКОВИХ моментів p_{i-1}
                if Cp_prev is None:
                    s_1s = relu_cov0(wc[0])       # шар 1: те саме, точна формула
                    s_1s = (Wd.T @ (s_1s @ Wd)).diagonal().clamp_min(1e-30).sqrt()
                else:
                    C_1s, _ = relu_cov_gauss(mp_prev, Cp_prev, nodes, wts)
                    s_1s = (Wd.T @ (C_1s @ Wd)).diagonal().clamp_min(1e-30).sqrt()
                # 3) тавтологічний контроль: вибіркова Ĉ(x_{i-1}) ⟹ мусить дати 0
                s_tau = (Wd.T @ (Cx @ Wd)).diagonal().clamp_min(1e-30).sqrt()

                d_tru = sh - st
                def rho(d):
                    return float((d * d_tru).sum()
                                 / (d.norm() * d_tru.norm()).clamp_min(1e-300))
                tab.setdefault(i, []).append((
                    float(((s_rec - st).abs() / st).mean()),
                    float(((s_1s - st).abs() / st).mean()),
                    float(((sh - st).abs() / st).mean()),
                    rho(sh - s_rec), rho(sh - s_1s),
                    float((sh - s_tau).abs().max())))

                C_rec, m_rec = relu_cov_gauss(m_rec @ Wd, C_p_rec, nodes, wts)
                Cx_prev, mp_prev, Cp_prev = Cx, mp, Cp
                x = torch.relu(p)
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    print(f'\n{len(fs)} мереж, істина {TRUTH_N}')
    print(f'{"шар":>4s} {"рекурсія":>10s} {"однокрок":>10s} {"вибірка":>10s}'
          f' {"ρ рек":>7s} {"ρ 1крок":>8s} {"тавт":>9s}')
    print('-' * 64)
    for i in SHOW:
        if i not in tab:
            continue
        a = np.mean(np.array(tab[i]), 0)
        print(f'{i:4d} {a[0]:10.3e} {a[1]:10.3e} {a[2]:10.3e}'
              f' {a[3]:+7.4f} {a[4]:+8.4f} {a[5]:9.1e}')
    print('\nЧИТАННЯ: якщо «однокрок» не росте з глибиною, а ρ 1крок ≫ ρ рек —')
    print('накопичення прибрано і δs стає доступною на глибині. «тавт» = 0 завжди.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
