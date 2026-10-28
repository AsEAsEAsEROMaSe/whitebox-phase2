"""НОВА КОНСТРУКЦІЯ З РЕАЛЬНИМ ЗАМИКАННЯМ (без оракула). (мета 1e-8)

ЗВІДКИ. `last_only.py`: `raw − Φδm − φδs` на ОСТАННЬОМУ шарі дає 144×, тоді як
наш v-ланцюг знімає лише 3.6% похибки (raw без поправок = 0.964× бази).
Стеля нової конструкції 144× ⟹ бал 1.5e-09, з великим запасом над ціллю 1e-8.

АЛЕ 144× — ОРАКУЛЬНІ. Тут той самий доданок береться з РЕАЛЬНОГО замикання:
    δm = m̂₃₁ − m_замикання,   δs = ŝ₃₁ − s_замикання
з β-вагами β = 1/(1+r²), r = похибка_замикання / шум_вибірки, обидва ВИМІРЯНІ.
Очікування з арифметики: r_m≈2.9 ⟹ β_m≈0.11, r_s≈12 ⟹ β_s≈0.007, тобто
приблизно 1.12×. Якщо вийде помітно більше — модель хибна; менше — теж.

ЩО ПОРІВНЮЄМО
    база                 raw − v
    ОРАКУЛ на шарі 31    стеля 144×
    замикання на шарі 31 ★  нова конструкція, реальна
    наша β-σ на всіх шарах  опора 1.28×
    комбінація              чи додаються вони

КОНТРОЛІ
  * оракульний рядок мусить відтворити ~144× — позитивний контроль стенда;
  * друкуємо виміряні r_m, r_s на шарі 31 — щоб бачити, чи збігаються
    з передбаченням 2.9 і 12;
  * істина власна, 4М свіжих семплів.
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
from closure_precision import NQ                              # noqa: E402
from mu_gain import closure_ms                                # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613)
UPTO = 31
# β для σ-поправки на всіх шарах (наша опора, beta_gain.py)
R_S = {1: 0.131, 2: 0.947, 3: 1.352, 4: 1.933, 5: 2.426, 6: 2.744, 7: 2.998,
       8: 3.605, 9: 4.082, 10: 4.561, 11: 5.140, 12: 5.618}


def run_one(wc, seed, dev, mu_t, sg_t, clo):
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
    vsig = v.clone()
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        Wd = wc[i].double()
        v = (v @ Wd) * g
        add = 0.0
        if i in R_S:
            b = 1.0 / (1.0 + R_S[i] ** 2)
            add = _phi(m / s) * (s - clo[i][1]) * b
        vsig = (vsig @ Wd) * g + add
        x = torch.relu(p)
    mt, st = mu_t[L_ - 2].double(), sg_t[L_ - 2].double()
    mc, sc = clo[L_ - 1]
    t_o, t_c = mt / st, m / s
    return (x.double().sum(0) / n2, v, vsig,
            _Phi(t_o) * (m - mt), _phi(t_o) * (s - st),      # оракульні
            _Phi(t_c) * (m - mc), _phi(t_c) * (s - sc),      # із замикання
            float(((mc - mt).abs() / st).mean()),
            float(((sc - st).abs() / st).mean()),
            float(((m - mt).abs() / st).mean()),
            float(((s - st).abs() / st).mean()))


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

    keys = ('база raw−v', 'ОРАКУЛ шар31 (стеля)', 'замикання шар31 ★',
            'наша β-σ (опора)', 'β-σ + замикання шар31')
    E = {k: [] for k in keys}
    st_m, st_s, nz_m, nz_s = [], [], [], []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            clo = closure_ms(wc, nodes, wts, UPTO)
            t_last = tr[-1]
            for sd in SEEDS:
                (raw, v, vsig, dmo, dso, dmc, dsc,
                 em, es, nm, ns) = run_one(wc, sd, dev, mu_t, sg_t, clo)
                st_m.append(em); st_s.append(es); nz_m.append(nm); nz_s.append(ns)
                bm = 1.0 / (1.0 + (em / nm) ** 2)
                bs = 1.0 / (1.0 + (es / ns) ** 2)
                E['база raw−v'].append((raw - v - t_last).cpu().numpy())
                E['ОРАКУЛ шар31 (стеля)'].append((raw - dmo - dso - t_last).cpu().numpy())
                E['замикання шар31 ★'].append(
                    (raw - bm * dmc - bs * dsc - t_last).cpu().numpy())
                E['наша β-σ (опора)'].append((raw - vsig - t_last).cpu().numpy())
                E['β-σ + замикання шар31'].append(
                    (raw - vsig - bm * dmc - bs * dsc - t_last).cpu().numpy())
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    base = float((np.array(E['база raw−v']) ** 2).mean())
    rm = np.mean(st_m) / np.mean(nz_m)
    rs = np.mean(st_s) / np.mean(nz_s)
    print(f'\n{len(fs)} мереж × {len(SEEDS)} жеребів, істина {TRUTH_N}')
    print(f'шар 31: похибка замикання m {np.mean(st_m):.3e} / шум m̂ '
          f'{np.mean(nz_m):.3e} ⟹ r_m = {rm:.2f}, β_m = {1/(1+rm*rm):.3f}')
    print(f'шар 31: похибка замикання s {np.mean(st_s):.3e} / шум ŝ '
          f'{np.mean(nz_s):.3e} ⟹ r_s = {rs:.2f}, β_s = {1/(1+rs*rs):.3f}')
    print(f'\n{"варіант":26s} {"MSE":>11s} {"×":>9s}')
    print('-' * 50)
    for k in keys:
        m = float((np.array(E[k]) ** 2).mean())
        print(f'{k:26s} {m:11.4e} {base/m:9.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
