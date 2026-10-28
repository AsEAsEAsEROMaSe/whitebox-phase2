"""ШИРОКЕ ПІДТВЕРДЖЕННЯ ПОПРАВКИ G — ЯК У САБМІТІ. (мета 1e-8)

ЗВІДКИ. `sigma_with_g.py` на 6 мережах: σ-гілка 1.287× → **1.389×** (+7.9%).
Але наше власне правило: **малі прогони не розсуджують** (С66: та сама
конфігурація дала 1.140× на одних 40 мережах і 1.004× на інших).

ЯК ЦЕЙ СТЕНД ЧЕСНІШИЙ. У `sigma_with_g` і `G`, і `β` міряні НА КОЖНІЙ мережі
(β з її ж істини). У сабміті так не буває — там КОНСТАНТИ. Тому тут:
    ПУЛ (8 мереж)   → `G[шар]` (21 число) і `β[шар]` для ОБОХ варіантів,
                      усереднені, далі ЗАМОРОЖЕНІ;
    ТЕСТ (32 інші)  → жодної істини, крім фінальної для MSE.
Це і ширший прогін, і ближче до продакшену.

КОНТРОЛІ
  * β обох варіантів виводяться з ПУЛУ однаково ⟹ жодного підгону на користь;
  * ⛔ варіант із `G` з ПЕРЕМІШАНИМИ шарами — має бути гірший за базовий;
  * парне порівняння: ті самі мережі й жереби;
  * друкуємо МЕДІАНУ по мережах поруч із середнім (С65: переноситься медіана)
    і частку мереж, де поправка ШКОДИТЬ.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402
from closure_precision import NQ, relu_cov_gauss              # noqa: E402
from resid_coeffs import basis_from_closure                   # noqa: E402
from closure_g import truth_all, closure_g, ALL               # noqa: E402
from sigma_with_g import final_truth, run_one, UPTO, KMAX     # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
POOL_N = 1_048_576
TEST_N = 2_097_152
NPOOL, NTEST = 8, 32
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)


def main(dev=DEV):
    torch.manual_seed(606)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:NPOOL + NTEST]

    def load(f):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        return cut_net(W, dev)[0]

    # ── ПУЛ: G і β, далі заморожені
    Gs, r0s, rGs = [], {i: [] for i in range(1, KMAX + 1)}, {i: [] for i in
                                                             range(1, KMAX + 1)}
    for k, f in enumerate(fs[:NPOOL]):
        with torch.no_grad():
            wc = load(f)
            T, mlast, slast = truth_all(wc, dev, 70001 + k, POOL_N)
            G, st = {}, {}
            for i in ALL:
                mp, Cpp, Crr = T[i]
                sp = Cpp.diagonal().clamp_min(1e-30).sqrt()
                Cg, _ = relu_cov_gauss(mp, Cpp, nodes, wts)
                Q = basis_from_closure(mp, sp, dev)
                G[i] = (Q.T @ (Crr - Cg) @ Q) / float(Cg.diagonal().mean())
                st[i] = sp
            st[UPTO] = slast
            Gs.append(G)
        print(f'  пул {k+1}/{NPOOL}', flush=True)
        Gs[-1] = {i: G[i] for i in ALL}
        # r_s обох варіантів на цій мережі пулу
        with torch.no_grad():
            noise = {i: [] for i in range(1, KMAX + 1)}
            for sd in SEEDS[:4]:
                _, _, _, sh = run_one(wc, sd, dev, {})
                for i in range(1, KMAX + 1):
                    noise[i].append(float(((sh[i] - st[i]).abs() / st[i]).mean()))
            for tag, g, dst in (('0', None, r0s), ('G', G, rGs)):
                tr = {}
                closure_g(wc, nodes, wts, g, dev, tr)
                for i in range(1, KMAX + 1):
                    e = float(((tr[i][1] - st[i]).abs() / st[i]).mean())
                    dst[i].append(e / float(np.mean(noise[i])))

    Gbar = {i: torch.stack([g[i] for g in Gs]).mean(0) for i in ALL}
    Gshuf = {i: Gbar[max(1, UPTO - i)] for i in ALL}
    b0 = {i: 1.0 / (1.0 + np.mean(r0s[i]) ** 2) for i in range(1, KMAX + 1)}
    bG = {i: 1.0 / (1.0 + np.mean(rGs[i]) ** 2) for i in range(1, KMAX + 1)}
    print('\nЗАМОРОЖЕНІ константи з пулу:')
    for i in (1, 2, 4, 8, 12, 16, 20):
        print(f'  шар {i:2d}: r_s {np.mean(r0s[i]):5.2f}→{np.mean(rGs[i]):5.2f}, '
              f'β {b0[i]:.3f}→{bG[i]:.3f}')

    # ── ТЕСТ: жодної істини, крім фінальної
    keys = ('база raw−v', 'σ-гілка ЯК ЗАРАЗ', '★σ-гілка З G',
            '★G зі СТАРИМИ консерв. β', '⛔G з перемішаними шарами')
    per = {k: [] for k in keys}
    for k, f in enumerate(fs[NPOOL:]):
        with torch.no_grad():
            wc = load(f)
            tl = final_truth(wc, dev, 91001 + k, TEST_N)
            sc = {}
            for tag, g in (('0', None), ('G', Gbar), ('S', Gshuf)):
                tr = {}
                closure_g(wc, nodes, wts, g, dev, tr)
                sc[tag] = {i: tr[i][1] for i in range(1, UPTO + 1)}
            var = {keys[1]: (sc['0'], b0), keys[2]: (sc['G'], bG),
                   keys[3]: (sc['G'], b0), keys[4]: (sc['S'], bG)}
            acc = {kk: [] for kk in keys}
            for sd in SEEDS:
                raw, v, vs, _ = run_one(wc, sd, dev, var)
                acc[keys[0]].append(float(((raw - v - tl) ** 2).mean()))
                for kk in keys[1:]:
                    acc[kk].append(float(((raw - vs[kk] - tl) ** 2).mean()))
            for kk in keys:
                per[kk].append(float(np.mean(acc[kk])))
        if (k + 1) % 8 == 0:
            b = np.array(per[keys[0]]); g = np.array(per[keys[2]])
            print(f'  тест {k+1}/{NTEST}: поточне × = {b.mean()/g.mean():.3f}',
                  flush=True)

    B = np.array(per[keys[0]])
    print(f'\nПУЛ {NPOOL} мереж (лише для констант) · ТЕСТ {NTEST} мереж '
          f'× {len(SEEDS)} жеребів, істина {TEST_N}')
    print(f'\n{"варіант":28s} {"MSE":>11s} {"× сер.":>8s} {"× медіана":>10s} '
          f'{"шкодить":>9s}')
    print('-' * 70)
    for kk in keys:
        A = np.array(per[kk])
        print(f'{kk:28s} {A.mean():11.4e} {B.mean()/A.mean():8.3f} '
              f'{np.median(B/A):10.3f} {float((A > B).mean()):8.1%}')


if __name__ == '__main__':
    main()
