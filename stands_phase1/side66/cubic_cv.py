"""КУБІЧНІ МАРГІНАЛЬНІ КЗ — ОСТАННЄ, ЩО ДЕШЕВЕ І НЕ МІРЯНЕ. (мета 1e-8)

ЗВІДКИ. Межа квадратичного класу порахована з наших чисел: лінійна частина
2.9× + квадратична стеля 2.6× = 7.5× (бал ~8.5e-08). Щоб іти далі, потрібні
моменти ТРЕТЬОГО порядку від x₀ — а це тензор 256³ (трійки = 1.14 бюджету).

АЛЕ МАРГІНАЛЬНІ треті моменти відомі ТОЧНО і коштують один прохід:
    p₀ⱼ = z·W₀[:,j] ~ N(0, σⱼ²),  σⱼ = ‖W₀[:,j]‖
    E|p|³      = σ³·2√(2/π)
    E[relu(p)³] = σ³·√(2/π)
Тензор не потрібен — це 256 незалежних КЗ по одній координаті.

ЧОМУ САМЕ |p|³. У проєкті вже закриті: p² (прибите вибілюванням ТОЧНО),
p³ (прибите антитетикою ТОЧНО), relu(p)² (прибите обома), пари через арксинус
(нуль), p⁴ (виміряний нуль на 20 мережах). Але `|p|³` антитетикою НЕ
прибивається: пара (z, −z) дає рівно |p|³, яке має власний розкид.

ЩО МІРЯЄМО. КЗ переносимо тим самим v-ланцюгом (щоб вона потрапила «в кінець»)
і віднімаємо з регресійним коефіцієнтом, підігнаним на ІНШИХ МЕРЕЖАХ.

КОНТРОЛІ
  * p³ окремим рядком: мусить дати РІВНО нуль (антитетика прибиває тотожно) —
    позитивний контроль, що стенд бачить прибиті величини як прибиті;
  * SHUFFLE коефіцієнта по мережах ⟹ виграш мусить зникнути;
  * коефіцієнт з ІНШИХ мереж (5 згорток), не з тієї, де міряємо.
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
from born_source import truth_moments                         # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613)
S2PI = math.sqrt(2.0 / math.pi)


def run_one(wc, seed, dev):
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
    npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used) / (2.0 * per_cut)))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    xin = torch.cat([z, -z], 0)
    n2 = float(xin.shape[0])
    cov = (xin.T @ xin) / n2
    ev, evec = torch.linalg.eigh(cov.double())
    wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()

    w0 = wh @ wc[0]
    p0 = (xin @ w0).double()
    sg = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt().double()

    # КЗ з ТОЧНО відомими середніми (для p ~ N(0, σ²))
    cvs = {
        '|p|³': (p0.abs() ** 3).sum(0) / n2 - sg ** 3 * 2.0 * S2PI,
        'relu³': (torch.relu(p0) ** 3).sum(0) / n2 - sg ** 3 * S2PI,
        'p³ (контроль=0)': (p0 ** 3).sum(0) / n2,
    }
    x = torch.relu(p0.float())
    u0 = x.double().sum(0) / n2 - (sg * INV_SQRT_2PI)

    v = u0.clone()
    carried = {k: val.clone() for k, val in cvs.items()}
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        Wd = wc[i].double()
        v = (v @ Wd) * g
        for k in carried:
            carried[k] = (carried[k] @ Wd) * g
        x = torch.relu(p)
    return x.double().sum(0) / n2, v, carried


def main(nets=10, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keys = ('|p|³', 'relu³', 'p³ (контроль=0)')
    E, C = [], {k: [] for k in keys}
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, _, _ = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            eb = []
            cc = {q: [] for q in keys}
            for sd in SEEDS:
                raw, v, car = run_one(wc, sd, dev)
                eb.append((raw - v - t_last).cpu().numpy())
                for q in keys:
                    cc[q].append(car[q].cpu().numpy())
            E.append(eb)
            for q in keys:
                C[q].append(cc[q])
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    E = np.array(E)
    base = float((E ** 2).mean())
    n = E.shape[0]
    fold = np.arange(n) % 5
    print(f'\n{n} мереж × {len(SEEDS)} жеребів; база MSE {base:.4e}')
    print(f'\n{"КЗ":22s} {"MSE":>11s} {"×":>8s} {"λ":>10s} {"ρ":>9s}')
    print('-' * 62)
    rs = np.random.default_rng(9)
    for q in keys + ('SHUFFLE',):
        U = np.array(C['|p|³' if q == 'SHUFFLE' else q])
        if q == 'SHUFFLE':
            U = U[rs.permutation(n)]
        pred = np.zeros_like(E)
        for c in range(5):
            tr_, te = fold != c, fold == c
            lam = float((E[tr_] * U[tr_]).sum() / max((U[tr_] ** 2).sum(), 1e-300))
            pred[te] = E[te] - lam * U[te]
        lam_all = float((E * U).sum() / max((U ** 2).sum(), 1e-300))
        rho = float((E * U).sum() / math.sqrt(max(float((E ** 2).sum())
                                                  * float((U ** 2).sum()), 1e-300)))
        m = float((pred ** 2).mean())
        print(f'{q:22s} {m:11.4e} {base/m:8.3f} {lam_all:10.3e} {rho:+9.4f}')
    print('\nКОНТРОЛЬ: «p³» мусить дати рівно 1.000 (антитетика прибиває тотожно).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 10)
