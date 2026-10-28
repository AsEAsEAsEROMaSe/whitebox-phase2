"""μ-ПОПРАВКА: КОРИГУЄМО СЕРЕДНЄ, А НЕ ЛИШЕ σ. (мета 1e-8)

ЗВІДКИ — два виміри, що разом перевертають пріоритет.
1. `last_ceiling.py`: на останньому шарі −φ·δs дає 1.022×, а −Φ·δm дає 19.9×.
   Тобто 96.6% похибки — це похибка СЕРЕДНЬОГО, а не σ.
2. `closure_precision.py`, стовпець «зам.по m»: похибка замикання ПО СЕРЕДНЬОМУ
   менша за шум вибірки (s/√N ≈ 1.27e-2·s) аж до шару ~8, тоді як по σ
   замикання програє вже з шару 3.
⟹ ми коригували менш цінну величину гіршим боком замикання.

ЩО ПРОБУЄМО (усе з β-вагами, β = 1/(1+r²), r рахується ПО ХОДУ з відомого
шуму вибірки і виміряної на банку похибки замикання)
    σ-поправка          F·δs                       ← нинішнє, 1.264×
    μ-поправка          Φ(t)·δm                    ← нове
    μ+σ разом           Φ(t)·δm + φ(t)·δs
    μ ПЕРЕЗАПУСКОМ      V ← Φ(t)·δm + φ(t)·δs      ← не додаємо, а ЗАМІНЮЄМО
Останній варіант принципово інший: якщо `m_замикання` точна, то δm містить УСЮ
накопичену похибку, і її не треба нести — ланцюг можна перезапустити.

🚨 РИЗИК ПОДВІЙНОГО ОБЛІКУ. `v` уже несе накопичену похибку середнього. Додавати
до неї ще й Φ·δm — це порахувати те саме двічі. Саме тому поруч стоїть варіант
із ЗАМІНОЮ, і саме тому дивимось на всі чотири в одній таблиці.

КОНТРОЛІ
  * база і ОРАКУЛ усі шари — опорні точки (1.000 і ~2.6-2.9×);
  * σ-поправка мусить відтворити ~1.25× з `beta_gain.py`;
  * мережі беруться з 8-ї (r міряні на 0..5) — без підгонки на тих самих;
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
from born_source import truth_moments, _Phi, _phi             # noqa: E402
from closure_precision import relu_cov0, relu_cov_gauss, NQ   # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)
SKIP = 8
UPTO = 12

# похибки замикання (частки істинного s), виміряні на мережах 0..5 — clo_full.log
E_S = {1: 2.97e-4, 2: 3.67e-3, 3: 6.36e-3, 4: 9.63e-3, 5: 1.24e-2, 6: 1.46e-2,
       7: 1.69e-2, 8: 2.06e-2, 9: 2.39e-2, 10: 2.67e-2, 11: 3.10e-2, 12: 3.37e-2}
E_M = {1: 2.81e-4, 2: 1.73e-3, 3: 3.42e-3, 4: 5.25e-3, 5: 6.78e-3, 6: 7.89e-3,
       7: 8.79e-3, 8: 1.02e-2, 9: 1.17e-2, 10: 1.28e-2, 11: 1.45e-2, 12: 1.55e-2}


def closure_ms(wc, nodes, wts, upto):
    """(m, s) доактиваційних `p` шарів 1..upto із самих ваг."""
    w0 = wc[0].double()
    m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    C_est = relu_cov0(wc[0])
    out = {}
    for i in range(1, upto + 1):
        Wd = wc[i].double()
        C_p = Wd.T @ (C_est @ Wd)
        m_p = m_est @ Wd
        out[i] = (m_p, C_p.diagonal().clamp_min(1e-30).sqrt())
        if i < upto:
            C_est, m_est = relu_cov_gauss(m_p, C_p, nodes, wts)
    return out


def run_one(wc, seed, dev, sg_t, mu_t, clo, variants, noise_s, noise_m):
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
        t = m / s
        Fs, Fm = _phi(t), _Phi(t)
        Wd = wc[i].double()
        v = (v @ Wd) * g

        add_s = add_m = None
        if i in clo:
            m_c, s_c = clo[i]
            bs = 1.0 / (1.0 + (E_S[i] / noise_s) ** 2)
            bm = 1.0 / (1.0 + (E_M[i] / noise_m) ** 2)
            add_s = Fs * (s - s_c) * bs
            add_m = Fm * (m - m_c) * bm
        ds_o = Fs * (s - sg_t[i - 1].double())
        dm_o = Fm * (m - mu_t[i - 1].double())

        for q in variants:
            base_carry = (V[q] @ Wd) * g
            if q == 'oracle':
                V[q] = base_carry + ds_o
            elif q == 'oracle_ms':
                V[q] = base_carry + ds_o + dm_o
            elif add_s is None:
                V[q] = base_carry
            elif q == 'sigma':
                V[q] = base_carry + add_s
            elif q == 'mu':
                V[q] = base_carry + add_m
            elif q == 'mu_sigma':
                V[q] = base_carry + add_m + add_s
            elif q == 'restart':
                V[q] = add_m + add_s + base_carry * (1.0 - min(1.0, float(
                    1.0 / (1.0 + (E_M[i] / noise_m) ** 2))))
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

    variants = ['oracle', 'oracle_ms', 'sigma', 'mu', 'mu_sigma', 'restart']
    E_b, VAR = [], {q: [] for q in variants}
    noise_s = noise_m = None
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 70001 + k, TRUTH_N)
            clo = closure_ms(wc, nodes, wts, UPTO)
            if noise_s is None:
                per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
                usd = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
                np2 = 2 * max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - usd)
                                             / (2.0 * per_cut)))
                noise_s = 1.0 / math.sqrt(2.0 * np2)   # відн. шум ŝ
                noise_m = 1.0 / math.sqrt(np2)         # відн. шум m̂ (в одиницях s)
            t_last = tr[-1]
            eb = []
            ev = {q: [] for q in variants}
            for sd in SEEDS:
                raw, v, V = run_one(wc, sd, dev, sg_t, mu_t, clo, variants,
                                    noise_s, noise_m)
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

    print(f'\n{len(fs)} мереж (з {SKIP}-ї) × {len(SEEDS)} жеребів')
    print(f'база MSE {base:.4e}; шум ŝ {noise_s:.3e}, шум m̂ {noise_m:.3e}')
    names = {'oracle': 'ОРАКУЛ σ усі шари', 'oracle_ms': 'ОРАКУЛ μ+σ усі шари',
             'sigma': 'замикання σ, β (опора ~1.25)', 'mu': 'замикання μ, β ★',
             'mu_sigma': 'замикання μ+σ, β ★', 'restart': 'замикання μ+σ, перезапуск'}
    print(f'\n{"варіант":32s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}')
    print('-' * 66)
    for q in variants:
        E = np.array(VAR[q])
        m = float((E ** 2).mean())
        print(f'{names[q]:32s} {m:11.4e} {base/m:9.3f} {med(E):10.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
