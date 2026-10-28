"""ЧИ ДАЄ ПРАВИЛО ПРАЙСА БАЛ. (бічний стенд до С66, парне порівняння)

`price_rule.py` міряє R² самого правила переносу. Цей стенд міряє те, що
вирішує: MSE фіналу. Усе на ТИХ САМИХ семплах, різниця ЛИШЕ в множнику
переносу коваріаційної нев'язки:
    A (нинішнє)  S ← (g⊗g) ⊙ (Wᵀ S W)      g_j  = P(p_j > 0)
    B (Прайс)    S ← P     ⊙ (Wᵀ S W)      P_ij = P(p_i>0, p_j>0)
На діагоналі A бере `g²≈0.25` там, де правильно `g≈0.5`.

КОНТРОЛІ
  * парне: той самий жереб, той самий прогін, ті самі `g` — різниця в правилі;
  * істина рахується САМОСТІЙНО на свіжому жеребі (не мітка банку);
  * λ підганяється на ІНШИХ мережах (5 згорток по мережах), окремо для A і B;
  * друкуємо і МЕДІАНУ помережного відношення — на грейдер іде вона;
  * позитивний контроль: `+σ ОРАКУЛ` (δs виміряна проти істини) — стеля.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from sub66_torch import (GATE_N, INV_SQRT_2PI, MIN_PAIRS,     # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments, _phi                   # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331)


def relu_cov_true(w0):
    S = (w0.T @ w0).double()
    s = S.diagonal().clamp_min(1e-30).sqrt()
    rho = (S / torch.outer(s, s)).clamp(-1.0, 1.0)
    e2 = torch.outer(s, s) / (2.0 * math.pi) * (
        (1.0 - rho * rho).clamp_min(0.0).sqrt() + rho * (math.pi / 2 + torch.asin(rho)))
    return e2 - torch.outer(s, s) / (2.0 * math.pi)


def run_one(wc, seed, dev, mu_t, sg_t, gate_n=GATE_N):
    """Один прогін; НЕСЕМО ДВІ нев'язки паралельно (правила A і B) на тих семплах."""
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
    V = {k: v.clone() for k in ('A', 'B', 'D', 'O')}
    U = {k: torch.zeros_like(v) for k in ('A', 'B', 'D', 'O')}  # σ-доданок (для λ)

    xd = x.double()
    m0 = xd.sum(0) / n2
    S = {k: (xd.T @ xd) / n2 - torch.outer(m0, m0) - relu_cov_true(wc[0])
         for k in ('A', 'B', 'D')}

    for i in range(1, L_):
        p = x @ wc[i]
        M = (p[:gate_n] > 0).to(p.dtype)
        ng = float(M.shape[0])          # 🚨 НЕ gate_n: зріз може бути коротший за нього
        g = (M.sum(0) / ng).double()
        P = ((M.T @ M) / ng).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        F = _phi(m / s)
        Wd = wc[i].double()

        v = (v @ Wd) * g
        gg = torch.outer(g, g)
        gd = gg.clone()
        gd.diagonal().copy_(g)      # D: точна діагональ P_ii = g_i, БЕЗ шуму й БЕЗ флопів
        add = {}
        for k, mult in (('A', gg), ('B', P), ('D', gd)):
            SW = Wd.T @ (S[k] @ Wd)
            add[k] = F * (SW.diagonal() / (2.0 * s))
            S[k] = SW * mult
        add['O'] = _phi(mu_t[i - 1] / sg_t[i - 1]) * (s - sg_t[i - 1])
        for k in V:
            V[k] = (V[k] @ Wd) * g + add[k]
            U[k] = (U[k] @ Wd) * g + add[k]
        x = torch.relu(p)

    return x.double().sum(0) / n2, v, V, U


def main(nets=20, dev=DEV, gate_n=GATE_N):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    E_b, VAR, USG = [], {}, {}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            eb = []
            ev = {q: [] for q in ('A', 'B', 'D', 'O')}
            us = {q: [] for q in ('A', 'B', 'D', 'O')}
            for sd in SEEDS:
                raw, v, V, U = run_one(wc, sd, dev, mu_t, sg_t, gate_n)
                eb.append((raw - v - t_last).cpu().numpy())
                for q in ev:
                    ev[q].append((raw - V[q] - t_last).cpu().numpy())
                    us[q].append(U[q].cpu().numpy())
            E_b.append(eb)
            for q in ev:
                VAR.setdefault(q, []).append(ev[q])
                USG.setdefault(q, []).append(us[q])
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    E_b = np.array(E_b)
    for q in VAR:
        VAR[q] = np.array(VAR[q])
        USG[q] = np.array(USG[q])
    base = float((E_b ** 2).mean())
    n = E_b.shape[0]
    fold = np.arange(n) % 5

    def med(E):
        r = (E_b ** 2).mean(axis=(1, 2)) / (E ** 2).mean(axis=(1, 2))
        return float(np.median(r))

    print(f'\n{n} мереж × {len(SEEDS)} жеребів, власна істина {TRUTH_N}, GATE_N={gate_n}')
    print(f'{"варіант":34s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}')
    print('-' * 68)
    print(f'{"sub66 (несемо лише μ)":34s} {base:11.4e} {1.0:9.3f} {1.0:10.3f}')
    for q, nm in (('O', '+σ ОРАКУЛ (стеля)'),
                  ('A', '+σ доставлена, A: g⊗g'),
                  ('B', '+σ доставлена, B: Прайс повний'),
                  ('D', '+σ доставлена, D: лише діаг. ★')):
        m = float((VAR[q] ** 2).mean())
        print(f'{nm:34s} {m:11.4e} {base/m:9.3f} {med(VAR[q]):10.3f}')
        pred = np.zeros_like(E_b)
        for c in range(5):
            tr_, te = fold != c, fold == c
            lam = float((E_b[tr_] * USG[q][tr_]).sum() / (USG[q][tr_] ** 2).sum())
            pred[te] = E_b[te] - lam * USG[q][te]
        lam_all = float((E_b * USG[q]).sum() / (USG[q] ** 2).sum())
        mp = float((pred ** 2).mean())
        print(f'{"   те саме, λ=%.3f (чесна)" % lam_all:34s} {mp:11.4e} '
              f'{base/mp:9.3f} {med(pred):10.3f}')
        c = float((E_b * USG[q]).sum() / math.sqrt(float((E_b ** 2).sum())
                                                   * float((USG[q] ** 2).sum())))
        print(f'{"   кореляція з похибкою бази":34s} {c:+11.4f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         gate_n=int(sys.argv[2]) if len(sys.argv) > 2 else GATE_N)
