"""ЗАМИКАННЯ З ОПТИМАЛЬНИМИ ВАГАМИ — ФІКСУЄМО ЗДОБУТОК ЧЕСНО.

ЗВІДКИ. `stage_gain.py`: замикання шари 1..3 дає 1.157×, з шару 4 ШКОДИТЬ
(1..6 → 0.730×), бо там його похибка більша за вибірковий шум.
`closure_precision.py`: виміряно відношення r_k = похибка_замикання/шум_вибірки
на всіх 31 шарі.

ІДЕЯ. Поправку не вмикати/вимикати, а ЗВАЖИТИ. Якщо оцінка s_est має похибку η,
незалежну від вибірки, то оптимальний коефіцієнт при δs_est — це
    β_k = 1/(1 + r_k²)
(звичайна регресія: cov/var). При r→∞ β→0 ⟹ зашкодити НЕ МОЖЕ за побудовою.
Модель ρ = 1/√(1+r²) вже підтверджена виміром до 3-го знака на шарах 1-2 і
тримається для рекурсії на глибині (шар 8: 0.268 проти 0.296).

🚨 ЧЕСНІСТЬ. r_k ВИМІРЯНІ на мережах 0..5 (`clo_full.log`), а бал міряється на
мережах 8..15 — ІНШИХ. Інакше це була б підгонка на тих самих даних.

КОНТРОЛІ
  * β≡1 на шарах 1..8 (тобто без зважування) мусить ПРОГРАТИ — це відтворення
    ефекту «1..6 → 0.730×»; якщо не програє, вада стенда;
  * замикання 1..3 без ваг — відоме 1.157×, як опорна точка;
  * оракул усі шари — стеля 2.855×;
  * парне порівняння, медіана поруч із середнім.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (GATE_N, INV_SQRT_2PI, MIN_PAIRS,     # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments, _phi                   # noqa: E402
from closure_precision import relu_cov0, relu_cov_gauss, NQ   # noqa: E402
from stage_gain import closure_s                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)
SKIP = 8                     # мережі 0..5 пішли на вимір r_k ⟹ беремо з 8-ї

# r_k = похибка_замикання / шум_вибірки, ВИМІРЯНІ на мережах 0..5 (clo_full.log)
R_K = {1: 0.131, 2: 0.947, 3: 1.352, 4: 1.933, 5: 2.426, 6: 2.744,
       7: 2.998, 8: 3.605, 9: 4.082, 10: 4.561, 11: 5.140, 12: 5.618}
BETA = {k: 1.0 / (1.0 + r * r) for k, r in R_K.items()}


def run_one(wc, seed, dev, sg_t, s_clo, variants):
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
    V = {q: v.clone() for q in variants}

    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        F = _phi(m / s)
        Wd = wc[i].double()
        v = (v @ Wd) * g
        ds_c = (s - s_clo[i]) if i in s_clo else None
        ds_o = s - sg_t[i - 1].double()
        for q in variants:
            a = 0.0
            if q == 'oracle':
                a = F * ds_o
            elif ds_c is not None:
                kind, upto = q
                if i <= upto:
                    a = F * ds_c * (BETA[i] if kind == 'beta' else 1.0)
            V[q] = (V[q] @ Wd) * g + a
        x = torch.relu(p)
    return x.double().sum(0) / n2, v, V


def main(nets=8, dev=DEV):
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][SKIP:SKIP + nets]

    variants = ['oracle', ('flat', 3), ('flat', 8), ('beta', 8), ('beta', 12)]
    E_b, VAR = [], {q: [] for q in variants}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 70001 + k, TRUTH_N)
            s_clo = closure_s(wc, nodes, wts, 12)
            t_last = tr[-1]
            eb = []
            ev = {q: [] for q in variants}
            for sd in SEEDS:
                raw, v, V = run_one(wc, sd, dev, sg_t, s_clo, variants)
                eb.append((raw - v - t_last).cpu().numpy())
                for q in variants:
                    ev[q].append((raw - V[q] - t_last).cpu().numpy())
            E_b.append(eb)
            for q in variants:
                VAR[q].append(ev[q])
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    E_b = np.array(E_b)
    base = float((E_b ** 2).mean())

    def med(E):
        return float(np.median((E_b ** 2).mean(axis=(1, 2))
                               / (E ** 2).mean(axis=(1, 2))))

    print(f'\n{len(fs)} мереж (з {SKIP}-ї, r_k міряні на 0..5) × {len(SEEDS)} жеребів')
    print(f'база MSE {base:.4e}')
    print(f'β: ' + ' '.join(f'{k}:{BETA[k]:.3f}' for k in (1, 2, 3, 4, 6, 8, 12)))
    print(f'\n{"варіант":34s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}')
    print('-' * 68)
    names = {'oracle': 'ОРАКУЛ усі шари (стеля)',
             ('flat', 3): 'замикання 1..3, без ваг (опора)',
             ('flat', 8): 'замикання 1..8, без ваг (контроль)',
             ('beta', 8): 'замикання 1..8, β-ваги ★',
             ('beta', 12): 'замикання 1..12, β-ваги ★'}
    for q in variants:
        E = np.array(VAR[q])
        m = float((E ** 2).mean())
        print(f'{names[q]:34s} {m:11.4e} {base/m:9.3f} {med(E):10.3f}')
    print('\nКОНТРОЛЬ: «1..8 без ваг» мусить програти (<1.0), «β-ваги» — ні.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
