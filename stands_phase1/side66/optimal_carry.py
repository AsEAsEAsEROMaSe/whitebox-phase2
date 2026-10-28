"""ЧИ ОПТИМАЛЬНИЙ КОЕФІЦІЄНТ, З ЯКИМ МИ ТЯГНЕМО ПОМИЛКУ В КІНЕЦЬ. (мета 1e-8)

ЗВІДКИ. `v`-ланцюг — це КОНТРОЛЬНА ЗМІННА: `u₀ = x̄₀ − E[x₀]`, і `E[x₀] = σ/√(2π)`
відоме ТОЧНО. Класична теорія КЗ каже, що коефіцієнт при ній має бути
РЕГРЕСІЙНИМ: β* = Cov(a, u₀)/Var(u₀). Ми ж беремо аналітичний перенос ΠW_i·g_i —
наближення якобіана — і мовчки вважаємо його оптимальним. Це не перевірялось.
А саме тут сидить 96.6% похибки (`last_ceiling.py`: −Φδm дає 19.9×).

ЩО МІРЯЄМО (усе на тих самих прогонах, парно)
    база              raw − v                     ← наш перенос, λ=1
    λ·v               raw − λv, λ з ІНШИХ жеребів ← чи оптимальний масштаб
    λ по шарах        окремий λ на внесок кожного шару (32 числа)
    +2-й момент       додаємо diag(Ĉ₀ − C₀) як ДРУГУ КЗ з регресійним коеф.
Друга КЗ теж має ТОЧНО відоме середнє (арккосинусне ядро), тобто законна.

🚨 ПАСТКА, ЯКОЇ УНИКАЄМО. Коефіцієнт, підігнаний на ТОМУ САМОМУ жеребі, дає
фальшивий виграш (С63: діагональ 1.19× там, де мусить бути 1.00). Тому λ
підганяється на ІНШИХ ЖЕРЕБАХ тієї ж мережі (їх багато — 24), а перевіряється
на відкладених.

КОНТРОЛІ
  * λ=1 мусить відтворити базу РІВНО (позитивний контроль реалізації);
  * SHUFFLE: λ з переміщаних жеребів ⟹ виграш мусить зникнути;
  * друкуємо сам λ_opt: якщо він ≈1, наш перенос уже оптимальний і гілка мертва;
    якщо ні — знайдено систематичну ваду переносу.
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
from closure_precision import relu_cov0                       # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
NDRAW = 24


def run_one(wc, seed, dev):
    """Прогін; повертає raw, v (наш перенос), і ПОШАРОВІ внески переносу."""
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

    xd = x.double()
    m0 = xd.sum(0) / n2
    sg0 = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt()
    u0 = m0 - (sg0 * INV_SQRT_2PI).double()          # КЗ №1: перший момент шару 0
    d0 = ((xd.T @ xd) / n2 - torch.outer(m0, m0)
          - relu_cov0(wc[0])).diagonal()             # КЗ №2: другий момент шару 0

    v = u0.clone()
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        v = (v @ wc[i].double()) * g
        x = torch.relu(p)
    # той самий перенос, але для КЗ №2 (щоб мати її «в одиницях фіналу»)
    v2 = d0.clone()
    xx = None
    return x.double().sum(0) / n2, v, u0, d0, v2


def carry(wc, vec, seed, dev):
    """Переносимо довільний вектор шару 0 тим самим ланцюгом (ті самі затвори)."""
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
    v = vec.clone()
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        v = (v @ wc[i].double()) * g
        x = torch.relu(p)
    return v


def main(nets=6, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    res = {k: [] for k in ('база', 'λ', 'λ+2й', 'shuffle')}
    lams = []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, _, _ = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            E, V, V2 = [], [], []
            for j in range(NDRAW):
                sd = 12345 + 7919 * j
                raw, v, u0, d0, _ = run_one(wc, sd, dev)
                v2 = carry(wc, d0, sd, dev)
                E.append((raw - t_last).cpu().numpy())
                V.append(v.cpu().numpy())
                V2.append(v2.cpu().numpy())
            E = np.array(E); V = np.array(V); V2 = np.array(V2)

            # половина жеребів — підгін, половина — перевірка
            h = NDRAW // 2
            def mse(pred):
                return float((pred ** 2).mean())
            lam = float((E[:h] * V[:h]).sum() / (V[:h] ** 2).sum())
            lams.append(lam)
            A = np.stack([V[:h].ravel(), V2[:h].ravel()], 1)
            c, *_ = np.linalg.lstsq(A, E[:h].ravel(), rcond=None)
            rs = np.random.default_rng(k)
            Vs = V[rs.permutation(NDRAW)]
            lam_sh = float((E[:h] * Vs[:h]).sum() / (Vs[:h] ** 2).sum())

            res['база'].append(mse(E[h:] - V[h:]))
            res['λ'].append(mse(E[h:] - lam * V[h:]))
            res['λ+2й'].append(mse(E[h:] - c[0] * V[h:] - c[1] * V2[h:]))
            res['shuffle'].append(mse(E[h:] - lam_sh * Vs[h:]))
        print(f'  мережа {k+1}/{len(fs)}: λ={lam:.4f}', flush=True)

    b = float(np.mean(res['база']))
    print(f'\n{len(fs)} мереж × {NDRAW} жеребів (12 підгін / 12 перевірка)')
    print(f'λ_opt: сер {np.mean(lams):.4f}, розкид {np.std(lams):.4f} '
          f'(1.0000 = наш перенос уже оптимальний)')
    print(f'\n{"варіант":28s} {"MSE":>11s} {"×":>9s}')
    print('-' * 50)
    for k in ('база', 'λ', 'λ+2й', 'shuffle'):
        m = float(np.mean(res[k]))
        print(f'{k:28s} {m:11.4e} {b/m:9.3f}')
    print('\nЧИТАННЯ: λ≈1 ⟹ перенос оптимальний за масштабом.')
    print('«λ+2й» > 1 ⟹ другий момент шару 0 несе НЕЗАЛЕЖНУ інформацію.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
