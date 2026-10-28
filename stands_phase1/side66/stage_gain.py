"""СКІЛЬКИ ДАЄ σ-ПОПРАВКА, ОБМЕЖЕНА ПЕРШИМИ ШАРАМИ — ОРАКУЛ vs РЕАЛЬНЕ ЗАМИКАННЯ.

ЗВІДКИ (`closure_precision.py`): на шарі 1 замикання дає `s` у 7.7× точніше за
вибірку, і δs із кореляцією **0.9917** проти істинної. На шарі 2 паритет,
з шару 3 замикання програє (його похибка накопичується рекурсивно).
Оракульна σ-поправка на ВСІХ шарах дає 3.0×.

ПИТАННЯ. Скільки з тих 3.0× сидить у перших шарах — і чи бере це замикання
БЕЗ оракула. Якщо шар 1 дає помітну частку, це готовий виграш: δs там
рахується з ваг майже точно, а коштує ~0.1% бюджету.

ЩО ПОРІВНЮЄМО (усе на ТИХ САМИХ семплах, парно)
    база              sub66, несемо лише μ
    ОРАКУЛ до k       δs = ŝ − s_істинна на шарах 1..k        ← стеля
    ЗАМИКАННЯ до k    δs = ŝ − s_замикання на шарах 1..k      ← метод
Замикання рахується ОДИН раз на мережу (не залежить від жеребу):
    C₀ = арккосинус (точно) → C_p = WᵀCW → C наступного = моменти relu(гаусa).

КОНТРОЛІ
  * «ЗАМИКАННЯ до 1» мусить майже збігтися з «ОРАКУЛ до 1» (бо ρ=0.9917) —
    якщо ні, вада в реалізації, а не в ідеї;
  * істина власна, 4М свіжих семплів; парне порівняння; медіана поруч із середнім;
  * ОРАКУЛ усі шари відтворює відомі 3.0× — позитивний контроль стенда.
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
from closure_precision import (relu_cov0, relu_cov_gauss,     # noqa: E402
                               relu_m, NQ)

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)
KS = (1, 2, 3, 6)


def closure_s(wc, nodes, wts, upto):
    """s_замикання доактиваційних `p` шарів 1..upto — з самих ваг, без вибірки."""
    w0 = wc[0].double()
    m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    C_est = relu_cov0(wc[0])
    out = {}
    for i in range(1, upto + 1):
        Wd = wc[i].double()
        C_p = Wd.T @ (C_est @ Wd)
        out[i] = C_p.diagonal().clamp_min(1e-30).sqrt()
        if i < upto:
            C_est, m_est = relu_cov_gauss(m_est @ Wd, C_p, nodes, wts)
    return out


def run_one(wc, seed, dev, sg_t, s_clo, ks):
    """Прогін sub66 + σ-поправки, обмежені шарами 1..k, оракульні й замиканням."""
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
    keys = [('o', k) for k in ks] + [('c', k) for k in ks]
    V = {q: v.clone() for q in keys}

    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        F = _phi(m / s)
        Wd = wc[i].double()
        v = (v @ Wd) * g
        add_o = F * (s - sg_t[i - 1].double())
        add_c = F * (s - s_clo[i]) if i in s_clo else None
        for (tag, k) in keys:
            a = 0.0
            if i <= k:
                a = add_o if tag == 'o' else (add_c if add_c is not None else 0.0)
            V[(tag, k)] = (V[(tag, k)] @ Wd) * g + a
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
        if int(np.load(f)['seed']) in have][:nets]

    ks = list(KS)
    keys = [('o', k) for k in ks] + [('c', k) for k in ks] + [('o', L_)]
    E_b, VAR = [], {q: [] for q in keys}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            s_clo = closure_s(wc, nodes, wts, max(ks))
            t_last = tr[-1]
            eb = []
            ev = {q: [] for q in keys}
            for sd in SEEDS:
                raw, v, V = run_one(wc, sd, dev, sg_t, s_clo, ks + [L_])
                eb.append((raw - v - t_last).cpu().numpy())
                for q in keys:
                    ev[q].append((raw - V[q] - t_last).cpu().numpy())
            E_b.append(eb)
            for q in keys:
                VAR[q].append(ev[q])
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    E_b = np.array(E_b)
    base = float((E_b ** 2).mean())

    def med(E):
        return float(np.median((E_b ** 2).mean(axis=(1, 2))
                               / (E ** 2).mean(axis=(1, 2))))

    print(f'\n{len(fs)} мереж × {len(SEEDS)} жеребів, істина {TRUTH_N}')
    print(f'база MSE {base:.4e}')
    print(f'\n{"варіант":32s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}')
    print('-' * 66)
    for q in keys:
        E = np.array(VAR[q])
        m = float((E ** 2).mean())
        tag, k = q
        nm = (f'ОРАКУЛ шари 1..{k}' if tag == 'o' else f'ЗАМИКАННЯ шари 1..{k} ★')
        if k == L_:
            nm = 'ОРАКУЛ усі шари (стеля)'
        print(f'{nm:32s} {m:11.4e} {base/m:9.3f} {med(E):10.3f}')
    print('\nКОНТРОЛЬ: «ЗАМИКАННЯ 1..1» ≈ «ОРАКУЛ 1..1» (ρ(δs)=0.9917).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
