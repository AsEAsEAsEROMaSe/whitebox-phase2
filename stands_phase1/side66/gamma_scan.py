"""СТАЛА УСАДКИ γ У ПЕРЕНОСІ НЕВ'ЯЗКИ. (бічний стенд до С66)

ЗВІДКИ. `carry_rule.py` показав НЕОЧІКУВАНЕ: математично ПРАВИЛЬНИЙ множник
(діагональ `P_ii = g_i`, теорема Прайса) дає 0.825×, а «хибний» `g_i²` — 1.055×.
Кореляція з похибкою при цьому падає 0.228 → 0.098.

МЕХАНІЗМ. `price_rule.py` виміряв, що нев'язка коваріації НАРОДЖУЄТЬСЯ на
кожному шарі заново (на шарі 1 перенос пояснює лише 22% дисперсії). Отже
накопичена «стара» нев'язка — шум, а корисна лише свіжа, з останніх шарів.
Зайвий множник `g≈0.5` на шар діє як ЕКСПОНЕНЦІЙНЕ ВІКНО, що її забуває.
⟹ оптимальна усадка — величина, яку ПІДБИРАЮТЬ, а не виводять.

ЩО МІРЯЄМО. Той самий перенос із множником `γ·(g⊗g)`, γ по сітці.
γ=1.0 — нинішній код. γ<1 — сильніша усадка (коротше вікно).
Ціна: РІВНО НУЛЬ додаткових флопів (та сама поелементна операція).

КОНТРОЛІ
  * парне: усі γ на ОДНОМУ прогоні, ті самі семпли, ті самі затвори;
  * λ для кожного γ підганяється на ІНШИХ мережах (5 згорток) ⟹ виграш не
    від підгонки масштабу, а від форми вікна;
  * друкуємо медіану (на грейдер іде вона) і кореляцію ρ: за моделлю
    виграш = 1/(1−ρ²), тож ρ мусить рухатись узгоджено з балом;
  * позитивний контроль: γ=1.0 мусить відтворити 1.055× із `carry_rule.py`.
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
GAMMAS = (0.25, 0.40, 0.55, 0.70, 0.85, 1.00, 1.20, 1.50)


def relu_cov_true(w0):
    S = (w0.T @ w0).double()
    s = S.diagonal().clamp_min(1e-30).sqrt()
    rho = (S / torch.outer(s, s)).clamp(-1.0, 1.0)
    e2 = torch.outer(s, s) / (2.0 * math.pi) * (
        (1.0 - rho * rho).clamp_min(0.0).sqrt() + rho * (math.pi / 2 + torch.asin(rho)))
    return e2 - torch.outer(s, s) / (2.0 * math.pi)


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
    keys = [f'{g:.2f}' for g in GAMMAS] + ['O']
    V = {k: v.clone() for k in keys}
    U = {k: torch.zeros_like(v) for k in keys}

    xd = x.double()
    m0 = xd.sum(0) / n2
    S0 = (xd.T @ xd) / n2 - torch.outer(m0, m0) - relu_cov_true(wc[0])
    S = {f'{g:.2f}': S0.clone() for g in GAMMAS}

    for i in range(1, L_):
        p = x @ wc[i]
        M = (p[:GATE_N] > 0).to(p.dtype)
        g = (M.sum(0) / float(M.shape[0])).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        F = _phi(m / s)
        Wd = wc[i].double()
        gg = torch.outer(g, g)

        v = (v @ Wd) * g
        add = {}
        for gam in GAMMAS:
            k = f'{gam:.2f}'
            SW = Wd.T @ (S[k] @ Wd)
            add[k] = F * (SW.diagonal() / (2.0 * s))
            S[k] = SW * (gg * gam)
        add['O'] = _phi(mu_t[i - 1] / sg_t[i - 1]) * (s - sg_t[i - 1])
        for k in V:
            V[k] = (V[k] @ Wd) * g + add[k]
            U[k] = (U[k] @ Wd) * g + add[k]
        x = torch.relu(p)

    return x.double().sum(0) / n2, v, V, U


def main(nets=12, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keys = [f'{g:.2f}' for g in GAMMAS] + ['O']
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
            ev = {q: [] for q in keys}
            us = {q: [] for q in keys}
            for sd in SEEDS:
                raw, v, V, U = run_one(wc, sd, dev, mu_t, sg_t)
                eb.append((raw - v - t_last).cpu().numpy())
                for q in keys:
                    ev[q].append((raw - V[q] - t_last).cpu().numpy())
                    us[q].append(U[q].cpu().numpy())
            E_b.append(eb)
            for q in keys:
                VAR.setdefault(q, []).append(ev[q])
                USG.setdefault(q, []).append(us[q])
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    E_b = np.array(E_b)
    for q in keys:
        VAR[q] = np.array(VAR[q])
        USG[q] = np.array(USG[q])
    base = float((E_b ** 2).mean())
    n = E_b.shape[0]
    fold = np.arange(n) % 5

    def med(E):
        return float(np.median((E_b ** 2).mean(axis=(1, 2)) / (E ** 2).mean(axis=(1, 2))))

    print(f'\n{n} мереж × {len(SEEDS)} жеребів, істина {TRUTH_N}. база MSE {base:.4e}')
    print(f'{"γ":>6s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}'
          f' {"ρ":>8s} {"1/(1-ρ²)":>9s} {"λ":>7s} {"× з λ":>7s}')
    print('-' * 76)
    for q in keys:
        E = VAR[q]
        m = float((E ** 2).mean())
        r = float((E_b * USG[q]).sum() / math.sqrt(float((E_b ** 2).sum())
                                                   * float((USG[q] ** 2).sum())))
        pred = np.zeros_like(E_b)
        for c in range(5):
            tr_, te = fold != c, fold == c
            lam = float((E_b[tr_] * USG[q][tr_]).sum() / (USG[q][tr_] ** 2).sum())
            pred[te] = E_b[te] - lam * USG[q][te]
        lam_all = float((E_b * USG[q]).sum() / (USG[q] ** 2).sum())
        mp = float((pred ** 2).mean())
        nm = 'ОРАКУЛ' if q == 'O' else q
        print(f'{nm:>6s} {m:11.4e} {base/m:9.3f} {med(E):10.3f} {r:+8.4f}'
              f' {1.0/(1.0-r*r):9.3f} {lam_all:7.3f} {base/mp:7.3f}')
    print('\nγ=1.00 — нинішній код (позитивний контроль: мусить дати ~1.055).')
    print('Ціна будь-якого γ — РІВНО НУЛЬ додаткових флопів.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 12)
