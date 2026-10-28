"""ЧИ Є МИГАННЯ ВАЖЕЛЕМ ПОВЕРХ v2 — З ОРАКУЛЬНИМ E[f]. (ідея юзера, вирішальний вимір)

ЗВІДКИ. `blink_info.py`: частки відкритих `f₁..f₃₁` пояснюють R²=0.169
дисперсії цільового скаляра (shuffle −0.008). Але то R² щодо СИРОГО внеску,
а v2 уже знімає 2.9× — сигнал може бути тим самим.

ЩО МІРЯЄМО. Контрольну змінну поверх ПОВНОЇ оцінки sub66:
    â_CV = (raw − v) − β·(f̄ − E[f])
`E[f]` беремо ОРАКУЛЬНО з 4М свіжих семплів ⟹ це СТЕЛЯ гілки, не метод.
Якщо навіть зі стелею виграш ~1.0 — мигання не важіль, і питання «як дізнатись
E[f] без істини» ставити нема сенсу.

🚨 ЧОМУ САМЕ ОРАКУЛЬНЕ E[f]. Якби E[f] бралося з тієї ж вибірки, поправка була б
ТОТОЖНО нуль (f̄ − f̄ = 0) — це пастка С63, на якій діагональ дала фальшиві 1.19×.
А для шару 0 E[f₀] = 0.5 ТОЧНО, але антитетика робить f̄₀ = 0.5 тотожно
⟹ f₀ як КЗ дає рівно нуль ЗА ПОБУДОВОЮ. Друкуємо його окремим рядком як
позитивний контроль стенда: там мусить бути рівно 1.000.

КОНТРОЛІ
  * β підганяється на ІНШИХ мережах (5 згорток) — не на тих, де міряється;
  * парне: та сама вибірка, той самий прогін, різниця лише в поправці;
  * медіана помережного відношення (на грейдер іде вона);
  * shuffle: β застосована до переміщаних по мережах f̄ ⟹ мусить дати ~1.000.
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

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331)


def oracle(wc, dev, seed, n_tot, chunk=65536):
    """Істина фіналу + ОРАКУЛЬНІ E[f_i] на свіжому жеребі, чистий N(0,I)."""
    gen = torch.Generator(device=dev)
    gen.manual_seed(int(seed))
    acc_a = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
    acc_f = torch.zeros(L_, dtype=torch.float64, device=dev)
    done = 0
    while done < n_tot:
        n = min(chunk, n_tot - done)
        x = torch.randn(n, W_, generator=gen, device=dev)
        for i in range(L_):
            p = x @ wc[i]
            acc_f[i] += (p > 0).to(p.dtype).mean(1).double().sum()
            x = torch.relu(p)
        acc_a += x.double().sum(0)
        done += n
    return acc_a / n_tot, acc_f / n_tot


def run_one(wc, seed, dev):
    """Прогін sub66; повертає (raw, v, f̄) — f̄ рахується на ТИХ САМИХ семплах."""
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

    h = x @ (wh @ wc[0])
    fbar = [(h > 0).to(h.dtype).mean().double()]
    x = torch.relu(h)
    sg0 = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = (x.double().sum(0) / n2 - (sg0 * INV_SQRT_2PI).double())
    for i in range(1, L_):
        p = x @ wc[i]
        fbar.append((p > 0).to(p.dtype).mean().double())
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        v = (v @ wc[i].double()) * g
        x = torch.relu(p)
    return x.double().sum(0) / n2, v, torch.stack(fbar)


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    E, D = [], []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            a_true, f_true = oracle(wc, dev, 90001 + k, TRUTH_N)
            eb, df = [], []
            for sd in SEEDS:
                raw, v, fbar = run_one(wc, sd, dev)
                eb.append((raw - v - a_true).cpu().numpy())
                df.append((fbar - f_true).cpu().numpy())
            E.append(eb); D.append(df)
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    E = np.array(E)                       # (мережі, жереби, нейрони)
    D = np.array(D)                       # (мережі, жереби, шари)
    base = float((E ** 2).mean())
    n = E.shape[0]
    fold = np.arange(n) % 5

    def apply_cv(cols, Dsrc):
        """β на ІНШИХ мережах; повертає MSE і медіанне відношення."""
        pred = np.zeros_like(E)
        for c in range(5):
            tr, te = fold != c, fold == c
            X = Dsrc[tr][:, :, cols].reshape(-1, len(cols))
            Y = E[tr].reshape(-1, E.shape[2])
            B, *_ = np.linalg.lstsq(X, Y, rcond=None)
            pred[te] = E[te] - Dsrc[te][:, :, cols] @ B
        r = (E ** 2).mean(axis=(1, 2)) / (pred ** 2).mean(axis=(1, 2))
        return float((pred ** 2).mean()), float(np.median(r))

    rs = np.random.default_rng(11)
    Dsh = D[rs.permutation(n)]
    print(f'\n{n} мереж × {len(SEEDS)} жеребів, оракульне E[f] на {TRUTH_N}')
    print(f'{"КЗ на":34s} {"MSE":>11s} {"× серед.":>9s} {"× медіана":>10s}')
    print('-' * 68)
    print(f'{"sub66 (без поправки)":34s} {base:11.4e} {1.0:9.3f} {1.0:10.3f}')
    for nm, cols, src in (('f₀ (позит. контроль: рівно 1.000)', [0], D),
                          ('f₁..f₃₁ (оракульне E[f]) ★', list(range(1, L_)), D),
                          ('усі f₀..f₃₁', list(range(L_)), D),
                          ('SHUFFLE по мережах', list(range(1, L_)), Dsh)):
        m, md = apply_cv(cols, src)
        print(f'{nm:34s} {m:11.4e} {base/m:9.3f} {md:10.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
