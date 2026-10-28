"""ПЕРЕЗВАЖУВАННЯ СЕМПЛІВ ПІД ТОЧНІ МОМЕНТИ ШАРУ 0. (мета 1e-8)

ЗВІДКИ. `v`-ланцюг віднімає похибку шару 0 у ЛІНІЙНОМУ наближенні, і несе її
через 31 шар наближеним якобіаном `g`, який сам шумний. Але цю похибку можна
прибрати ТОЧНО — не віднімаючи, а ПЕРЕЗВАЖУЮЧИ семпли:
    w_k ∝ exp(λ·x₀(z_k)),   підбираємо λ так, щоб  Σ w_k x₀(z_k) = E[x₀]
`E[x₀] = σ_j/√(2π)` відоме ТОЧНО ⟹ 256 законних умов. Задача опукла
(entropy balancing), розв'язується кількома кроками Ньютона:
    градієнт = Σw·x₀ − ціль,   гессіан = Σw·(x₀−m)(x₀−m)ᵀ

ЧОМУ ЦЕ МОЖЕ БУТИ СИЛЬНІШЕ. Перенос прибирає похибку шару 0 лише в першому
порядку і лише вздовж напрямку, який задає наближений якобіан. Перезважування
робить моменти шару 0 РІВНИМИ істинним тотожно — а отже і для ВСІХ наступних
шарів одразу, без жодного якобіана.

ЦІНА ~НУЛЬ: ваги залежать лише від `x₀`, тож рахуються одразу після першого
шару, і далі несемо зважені суми. Один прохід, нічого зберігати не треба.
Ньютон: 3-5 ітерацій по 256³ + N·256 ≈ 1e8 флоп проти бюджету 2.6e10.

⚠️ ЧОГО БОЇМОСЯ. Ваги залежать від даних ⟹ оцінка стає ЗМІЩЕНОЮ на O(1/N).
Класичний trade-off калібрувальних оцінювачів. Саме тому міряємо MSE, а не
дисперсію, і друкуємо зсув окремо.

КОНТРОЛІ
  * λ=0 (рівномірні ваги) мусить дати РІВНО `raw` — позитивний контроль;
  * друкуємо ‖Σw·x₀ − ціль‖ ПІСЛЯ Ньютона — мусить бути ~1e-15 (умови виконані);
  * розкид ваг (max/min) — якщо він великий, оцінка нестійка;
  * порівняння парне: та сама вибірка, ті самі семпли.
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
DEV = 'cuda:2'
TRUTH_N = 4_194_304
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)


def balance(x0, target, iters=12, ridge=1e-9):
    """Ваги w ∝ exp(λ·x₀) з умовою Σw·x₀ = target. Ньютон в опуклій задачі."""
    n, d = x0.shape
    lam = torch.zeros(d, dtype=torch.float64, device=x0.device)
    sc = x0.std(0).clamp_min(1e-12)
    xs = (x0 - x0.mean(0)) / sc                      # масштабуємо для стійкості
    tgt = (target - x0.mean(0)) / sc
    for _ in range(iters):
        e = (xs @ lam)
        e = e - e.max()
        w = torch.exp(e)
        w = w / w.sum()
        m = w @ xs
        r = m - tgt
        if float(r.abs().max()) < 1e-13:
            break
        dxs = xs - m
        H = (dxs * w.unsqueeze(1)).T @ dxs
        H = H + ridge * torch.eye(d, dtype=H.dtype, device=H.device) * H.diagonal().mean()
        lam = lam - torch.linalg.solve(H, r)
    e = (xs @ lam)
    w = torch.exp(e - e.max())
    return w / w.sum()


def run_one(wc, seed, dev, use_w):
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
    sg0 = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt()
    tgt = (sg0 * INV_SQRT_2PI).double()
    v = xd.sum(0) / n2 - tgt

    diag = None
    if use_w:
        w = balance(xd, tgt)
        diag = (float(((w @ xd) - tgt).abs().max()),
                float(w.max() / w.min()))
    else:
        w = torch.full((int(n2),), 1.0 / n2, dtype=torch.float64, device=dev)

    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        v = (v @ wc[i].double()) * g
        x = torch.relu(p)
    return (w @ x.double()), x.double().sum(0) / n2, v, diag


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keys = ('база raw−v', 'перезважування', 'перезваж.−v', 'контроль λ=0')
    E = {k: [] for k in keys}
    dg = []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, _, _ = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            for sd in SEEDS:
                aw, raw, v, d = run_one(wc, sd, dev, True)
                a0, raw0, _, _ = run_one(wc, sd, dev, False)
                dg.append(d)
                E['база raw−v'].append((raw - v - t_last).cpu().numpy())
                E['перезважування'].append((aw - t_last).cpu().numpy())
                E['перезваж.−v'].append((aw - v - t_last).cpu().numpy())
                E['контроль λ=0'].append((a0 - raw0).cpu().numpy())
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    b = float((np.array(E['база raw−v']) ** 2).mean())
    dg = np.array(dg)
    print(f'\n{len(fs)} мереж × {len(SEEDS)} жеребів')
    print(f'умови Ньютона ‖Σw·x₀ − ціль‖max = {dg[:,0].max():.2e} (мусить ~1e-13)')
    print(f'розкид ваг max/min = {dg[:,1].mean():.2f}')
    print(f'\n{"варіант":24s} {"MSE":>11s} {"×":>9s} {"|зсув|":>11s}')
    print('-' * 60)
    for k in keys:
        A = np.array(E[k])
        m = float((A ** 2).mean())
        print(f'{k:24s} {m:11.4e} {b/m:9.3f} {np.abs(A.mean(0)).mean():11.3e}')
    print('\nКОНТРОЛЬ λ=0 мусить бути РІВНО 0 (рівномірні ваги = звичайне середнє).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
