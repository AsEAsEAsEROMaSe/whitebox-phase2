"""🚨 ЧИ ДОДАЄТЬСЯ НАХИЛ ДО ВИБІЛЮВАННЯ, ЧИ ВОНИ ПРО ОДНЕ Й ТЕ САМЕ?

ЗВІДКИ. `tilt_wide.py` дав 1.1037× на 40 мережах — але стенд рахував ψ БЕЗ
ВИБІЛЮВАННЯ, а реальний `estimator_side1.py` вибілює:
    cov = xᵀx/(2n);  whiten = cov^(−1/2);  w0 = whiten @ w_cut[0]
Вибілювання примусово робить ЕМПІРИЧНИЙ другий момент вибірки ТОЧНО
одиничним (С55: 2.67× → 1.36×). А нахил `q ∝ φ·exp(zᵀCz)` — це теж
втручання у ДРУГИЙ МОМЕНТ. Отже вони можуть бити в ту саму ступінь свободи,
і 1.10× просто зникне.

🚨 Це рівно та пастка, від якої застерігає пам'ять: «ізольований стенд бреше —
питати, ЩО стенд прибрав». Мій стенд прибрав вибілювання.

ЩО ПОРІВНЮЄМО (усе на продакшн-npair, істина 1М):
  1. **база+вибіл.** — рівно те, що робить sub66;
  2. **нахил, без вибіл.** — те, що міряв `tilt_wide` (для звірки);
  3. **нахил + ЗВАЖЕНЕ вибілювання** — `Ĉ = Σw_i z_iz_iᵀ/Σw_i` замість
     звичайного, бо саме зважений другий момент прямує до `I` під `q`;
     оцінка незміщена `(1/n)Σw_iψ_i` і самонормована `Σw_iψ_i/Σw_i`.
🔑 Нахил ПАРНИЙ (`w(z)=w(−z)`), тож антитетика з ним сумісна без застережень.

КОНТРОЛІ
  * ✅ `c=0` у зваженому вибілюванні мусить ТОЧНО відтворити «база+вибіл.»;
  * ✅ «нахил без вибіл.» мусить відтворити ~1.10× із `tilt_wide`;
  * друкуємо ВСІ чотири MSE в одних одиницях, щоб бачити, що з чим змагається.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import cut_net                              # noqa: E402
from impsamp import gates_and_G                               # noqa: E402
from tilt_end2end import dirs_from_G                          # noqa: E402
from tilt_wide import truth_last                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NDRAW = 64
MTILT = 64
CBEST = 0.004


def run_draw(wc, G, m0, dev, seed, npair, U=None, c=0.0, white=False,
             selfnorm=False):
    g = torch.Generator(device=dev).manual_seed(seed)
    xi = torch.randn(npair, W_, device=dev, generator=g)
    if U is None or c == 0.0:
        z = xi
        w = torch.ones(npair, dtype=torch.float64, device=dev)
    else:
        al = (1.0 - 2.0 * c) ** -0.5
        Um = U[:, :MTILT]
        pr = (xi @ Um)
        z = xi + pr * (al - 1.0) @ Um.T
        q = (z @ Um).double() ** 2
        w = torch.exp(MTILT * float(np.log(al)) - c * q.sum(1))
    zb = torch.cat([z, -z], 0)
    wb = torch.cat([w, w], 0)
    w0 = wc[0]
    if white:
        # зважений другий момент: E_q[w·zzᵀ] = E_φ[zzᵀ] = I
        C = (zb * wb.unsqueeze(1).float()).T @ zb / float(wb.sum())
        ev, V = torch.linalg.eigh(C.double())
        wh = (V * ev.clamp_min(1e-8).rsqrt()) @ V.T
        w0 = wh.float() @ wc[0]
    x = torch.relu(zb @ w0)
    x0 = x.double()
    for w_ in wc[1:]:
        x = torch.relu(x @ w_)
    psi = x.double() - (x0 - m0) @ G
    num = (wb.unsqueeze(1) * psi).sum(0)
    return num / (wb.sum() if selfnorm else float(2 * npair))


def main(nets=20, dev=DEV):
    torch.manual_seed(1717)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    KEYS = ['база', 'база+вибіл', 'нахил', 'нахил+вибіл', 'нахил+вибіл+СН',
            '✅c=0+вибіл']
    tot = {k: 0.0 for k in KEYS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_net(W, dev)
            G = gates_and_G(wc, dev, 4242 + n)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() * INV_SQRT_2PI
            T = truth_last(wc, dev, 410_000 + 7919 * n)
            U = dirs_from_G(wc, G, dev, 66_000 + n)
            for d in range(NDRAW):
                sd = 33_000_000 + 811 * n + d
                cfg = {
                    'база': dict(),
                    'база+вибіл': dict(white=True),
                    'нахил': dict(U=U, c=CBEST),
                    'нахил+вибіл': dict(U=U, c=CBEST, white=True),
                    'нахил+вибіл+СН': dict(U=U, c=CBEST, white=True,
                                           selfnorm=True),
                    '✅c=0+вибіл': dict(U=U, c=0.0, white=True),
                }
                for k, kw in cfg.items():
                    mm = run_draw(wc, G, m0, dev, sd, npair, **kw)
                    tot[k] += float(((mm - T) ** 2).sum())
        del W, wc, G
        torch.cuda.empty_cache()
        if (n + 1) % 5 == 0:
            print(f'  … {n+1}/{nets}  нахил/база={tot["база"]/tot["нахил"]:.3f}× '
                  f' вибіл/база={tot["база"]/tot["база+вибіл"]:.3f}×'
                  f'  нахил+вибіл/вибіл='
                  f'{tot["база+вибіл"]/tot["нахил+вибіл"]:.3f}×', flush=True)

    b = tot['база']
    print(f'\n{nets} мереж × {NDRAW} жеребів продакшн-розміру\n')
    print(f'{"варіант":>16s} {"MSE відносно бази":>18s}')
    for k in KEYS:
        print(f'{k:>16s} {b/max(tot[k],1e-300):17.4f}×')
    print(f'\n🎯 ГОЛОВНЕ ЧИСЛО — приріст нахилу ПОВЕРХ вибілювання: '
          f'{tot["база+вибіл"]/max(tot["нахил+вибіл"],1e-300):.4f}×  '
          f'(самонормований {tot["база+вибіл"]/max(tot["нахил+вибіл+СН"],1e-300):.4f}×)')
    print(f'✅ КОНТРОЛЬ «c=0+вибіл» проти «база+вибіл»: '
          f'{tot["база+вибіл"]/max(tot["✅c=0+вибіл"],1e-300):.6f}× '
          f'(мусить бути 1.000000)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 20)
