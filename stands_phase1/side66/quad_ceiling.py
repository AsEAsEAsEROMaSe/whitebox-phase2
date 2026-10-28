"""СТЕЛЯ ПОВНОГО КВАДРАТИЧНОГО КЛАСУ — ПРЯМИЙ ВИМІР. (мета 1e-8)

ЗВІДКИ. Я стверджував, що квадратичні КЗ дають 2.6×, спираючись на стелю
σ-поправки. Але σ-поправка бере з нев'язки `S₀ = Ĉ(x₀) − C₀` лише ДІАГОНАЛЬ
ПІСЛЯ ПЕРЕНОСУ — один скаляр на нейрон на шар. Повний квадратичний простір —
це ВСІ 32 896 компонент `S₀`, кожна з ТОЧНО відомим середнім (арккосинус).
Стеля цього простору прямо не мірялась, а від неї залежить, чи існує шлях
до 1e-8 у нашому класі взагалі.

ЩО МІРЯЄМО. Багато НЕЗАЛЕЖНИХ жеребів на одній мережі ⟹ можна чесно
регресувати похибку фіналу на головні компоненти `S₀`:
    e(жереб) ≈ Σ_k c_k · PC_k(S₀)
R² на ВІДКЛАДЕНИХ жеребах ⟹ стеля = 1/(1−R²).
Порівнюємо з тим, що вже беремо (v-ланцюг + σ-поправка).

🚨 ПАСТКА, ЯКОЇ УНИКАЄМО. Регресія на 32 896 регресорів по 200 жеребах —
чисте перенавчання. Тому: PCA до K компонент, K ≪ жеребів, і R² рахується
ЛИШЕ на відкладених жеребах. Плюс SHUFFLE-контроль.

КОНТРОЛІ
  * перший стовпець — R² на НАВЧАЛЬНИХ жеребах: розрив із тестом показує,
    скільки з'їдає перенавчання;
  * SHUFFLE: ті самі компоненти, переміщані по жеребах ⟹ R²_тест ≈ 0;
  * свіп по K: якщо R² росте з K і не насичується — простір справді багатий;
    якщо насичується на малому K — уся інформація в кількох напрямках;
  * окремо: R² самої лише ДІАГОНАЛІ S₀ (те, що бере σ-поправка).
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
TRUTH_N = 2_097_152
NDRAW = 240
KS = (2, 5, 10, 20, 40, 80)


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
    x = torch.relu(xin @ (wh @ wc[0]))

    xd = x.double()
    m0 = xd.sum(0) / n2
    sg0 = wc[0].pow(2).sum(0).clamp_min(1e-30).sqrt().double()
    S0 = (xd.T @ xd) / n2 - torch.outer(m0, m0) - relu_cov0(wc[0])
    v = m0 - sg0 * INV_SQRT_2PI
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        v = (v @ wc[i].double()) * g
        x = torch.relu(p)
    iu = torch.triu_indices(S0.shape[0], S0.shape[1], device=dev)
    return x.double().sum(0) / n2, v, S0[iu[0], iu[1]], S0.diagonal()


def r2(Y, X, ntr):
    """R² на відкладених рядках; коефіцієнти — з перших ntr."""
    A = np.concatenate([X[:ntr], np.ones((ntr, 1))], 1)
    B = np.concatenate([X[ntr:], np.ones((len(X) - ntr, 1))], 1)
    w, *_ = np.linalg.lstsq(A, Y[:ntr], rcond=None)
    def sc(P, T):
        return 1.0 - float(((T - P) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())
    return sc(B @ w, Y[ntr:]), sc(A @ w, Y[:ntr])


def main(nets=2, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, _, _ = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            t_last = tr[-1]
            E, U, D = [], [], []
            for j in range(NDRAW):
                raw, v, su, dg = run_one(wc, 1000003 + 7919 * j, dev)
                E.append((raw - v - t_last).cpu().numpy())
                U.append(su.cpu().numpy().astype(np.float32))
                D.append(dg.cpu().numpy())
                if (j + 1) % 60 == 0:
                    print(f'    жереб {j+1}/{NDRAW}', flush=True)
            E = np.array(E); U = np.array(U); D = np.array(D)
            base = float((E ** 2).mean())

            ntr = NDRAW // 2
            Uc = U - U[:ntr].mean(0)
            _, S, Vt = np.linalg.svd(Uc[:ntr], full_matrices=False)
            PC = Uc @ Vt.T
            Dc = D - D[:ntr].mean(0)

            print(f'\nмережа {k+1}: база MSE {base:.4e}, {NDRAW} жеребів '
                  f'({ntr} навч / {NDRAW-ntr} тест)')
            print(f'{"регресори":26s} {"R² тест":>9s} {"R² навч":>9s} {"стеля":>8s}')
            print('-' * 56)
            for K in KS:
                te, trn = r2(E, PC[:, :K], ntr)
                print(f'{"PCA(S₀), K=%d" % K:26s} {te:+9.4f} {trn:+9.4f} '
                      f'{(1/(1-te) if te < 0.99 else 99):8.2f}')
            te, trn = r2(E, Dc[:, :80], ntr)
            print(f'{"лише діагональ S₀ (80)":26s} {te:+9.4f} {trn:+9.4f} '
                  f'{(1/(1-te) if te < 0.99 else 99):8.2f}')
            rs = np.random.default_rng(5)
            te, trn = r2(E, PC[rs.permutation(NDRAW)][:, :40], ntr)
            print(f'{"SHUFFLE K=40":26s} {te:+9.4f} {trn:+9.4f}')
    print('\nЧИТАННЯ: «стеля» = 1/(1−R²) — скільки дала б ІДЕАЛЬНА квадратична КЗ.')
    print('Якщо вона ~2-3× — межа класу підтверджена. Якщо ≫10× — шлях є.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 2)
