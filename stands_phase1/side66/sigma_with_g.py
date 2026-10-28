"""σ-ГІЛКА З ВИПРАВЛЕНИМ ЗАМИКАННЯМ — КІНЕЦЬ-У-КІНЕЦЬ. (мета 1e-8)

ЗВІДКИ. `closure_g.py` (пошарово): поправка `G` (21 універсальне число на шар,
базис із замикання, коефіцієнти з ІНШИХ мереж) робить σ-ланцюг замикання
точнішим у **1.9–2.7×** по всій глибині:
    шар  8: 2.041e-02 → 1.090e-02
    шар 16: 4.578e-02 → 1.860e-02
    шар 31: 9.448e-02 → 3.464e-02
🚨 Попереднє прочитання було ПОМИЛКОВЕ: я міряв похибку СЕРЕДНЬОГО, а поправка
лагодить ДИСПЕРСІЮ. Помилку спіймав юзер.

ЧОМУ ЦЕ МАЄ ЗНАЧЕННЯ. Сабміт (`estimator_side1/sub72`) несе саме σ-поправку
    vsig ← (vsig·W)·g + φ(t)·(ŝ − s_замикання)·β,   β = 1/(1+r_s²)
де `r_s` = похибка_замикання_по_s / шум_вибірки_по_s. Виміряні `r_s`: 1.93 на
шарі 4, 3.61 на 8-му ⟹ `β` = 0.21 і 0.071. Якщо похибка замикання падає вдвічі,
`r_s` падає вдвічі, а `β` на шарі 8 росте з 0.071 до ~0.21 — ВТРИЧІ.

ЩО МІРЯЄМО — не проміжні величини, а MSE фіналу:
    база                raw − v
    σ-гілка, замикання ЯК ЗАРАЗ
    σ-гілка, замикання З ПОПРАВКОЮ G (LOO по мережах)   ★
і поруч — виміряні `r_s`, `β` обох варіантів по шарах.

КОНТРОЛІ
  * `β` РАХУЮТЬСЯ з виміряних `r_s` цього ж прогону, однаково для обох ⟹
    порівняння чесне, жодного підгону на одну користь;
  * `G` беруться з ІНШИХ мереж (LOO) ⟹ жодної істини цієї мережі;
  * ⛔ варіант із `G` з ІНШОГО шару — має бути гірший;
  * істина власна; парне порівняння на тих самих жеребах.
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
from transport_r2 import CHUNK, cut_net                       # noqa: E402
from born_source import _Phi, _phi                            # noqa: E402
from closure_precision import NQ, relu_cov_gauss              # noqa: E402
from resid_coeffs import basis_from_closure                   # noqa: E402
from closure_g import truth_all, closure_g, ALL               # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
UPTO = 31
TRUTH_N = 1_048_576
SEEDS = (12345, 777, 24680, 999331, 5150, 60613, 727272, 8181)
KMAX = 20


def final_truth(wc, dev, seed, n):
    """E[relu(p₃₁)] — 256 чисел, яких truth_all не дає."""
    acc = torch.zeros(wc[UPTO].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < n:
        mch = min(CHUNK, n - done)
        z = torch.tensor(rng.standard_normal((mch // 2, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        x = torch.relu(x @ wc[0])
        for i in range(1, L_):
            x = torch.relu(x @ wc[i])
        acc += x.sum(0).double()
        done += mch
    return acc / done


def run_one(wc, seed, dev, variants):
    """Один жереб. variants: назва → (s_замикання по шарах, β по шарах)."""
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
    vs = {k: v.clone() for k in variants}
    shat = {}
    for i in range(1, L_):
        p = x @ wc[i]
        g = ((p[:GATE_N] > 0).to(p.dtype).sum(0) / GATE_N).double()
        pd = p.double()
        m = pd.sum(0) / n2
        s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        shat[i] = s
        Wd = wc[i].double()
        v = (v @ Wd) * g
        for k, (sc, beta) in variants.items():
            add = 0.0
            if i in beta:
                add = _phi(m / s) * (s - sc[i]) * beta[i]
            vs[k] = (vs[k] @ Wd) * g + add
        x = torch.relu(p)
    return x.double().sum(0) / n2, v, vs, shat


def main(nets=6, dev=DEV):
    torch.manual_seed(1010)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    keep = []
    for k, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            T, mlast, slast = truth_all(wc, dev, 70001 + k, TRUTH_N)
            tl = final_truth(wc, dev, 91001 + k, TRUTH_N)
            G, st = {}, {}
            for i in ALL:
                mp, Cpp, Crr = T[i]
                sp = Cpp.diagonal().clamp_min(1e-30).sqrt()
                Cg, _ = relu_cov_gauss(mp, Cpp, nodes, wts)
                Q = basis_from_closure(mp, sp, dev)
                G[i] = (Q.T @ (Crr - Cg) @ Q) / float(Cg.diagonal().mean())
                st[i] = sp
            st[UPTO] = slast
            keep.append((wc, G, st, tl))
        print(f'  істина мережі {k+1}/{len(fs)}', flush=True)

    keys = ('база raw−v', 'σ-гілка, замикання ЯК ЗАРАЗ',
            '★σ-гілка, замикання З G (LOO)')
    E = {k: [] for k in keys}
    rs = {'ЯК ЗАРАЗ': {i: [] for i in range(1, KMAX + 1)},
          'З G': {i: [] for i in range(1, KMAX + 1)}}
    for k, (wc, G, st, tl) in enumerate(keep):
        Gl = {i: torch.stack([o[1][i] for j, o in enumerate(keep) if j != k]
                             ).mean(0) for i in ALL}
        with torch.no_grad():
            s0 = {}
            sG = {}
            for name, g, dst in (('0', None, s0), ('G', Gl, sG)):
                tr = {}
                closure_g(wc, nodes, wts, g, dev, tr)
                for i in range(1, UPTO + 1):
                    dst[i] = tr[i][1]
            # шум вибірки по s і виміряні r_s (спільні жереби для обох)
            noise = {i: [] for i in range(1, KMAX + 1)}
            for sd in SEEDS[:4]:
                _, _, _, sh = run_one(wc, sd, dev, {})
                for i in range(1, KMAX + 1):
                    noise[i].append(float(((sh[i] - st[i]).abs() / st[i]).mean()))
            beta = {}
            for nm, sc in (('ЯК ЗАРАЗ', s0), ('З G', sG)):
                b = {}
                for i in range(1, KMAX + 1):
                    e = float(((sc[i] - st[i]).abs() / st[i]).mean())
                    r = e / float(np.mean(noise[i]))
                    rs[nm][i].append(r)
                    b[i] = 1.0 / (1.0 + r * r)
                beta[nm] = b
            var = {keys[1]: (s0, beta['ЯК ЗАРАЗ']), keys[2]: (sG, beta['З G'])}
            for sd in SEEDS:
                raw, v, vs, _ = run_one(wc, sd, dev, var)
                E[keys[0]].append((raw - v - tl).cpu().numpy())
                for kk in keys[1:]:
                    E[kk].append((raw - vs[kk] - tl).cpu().numpy())
        print(f'  мережа {k+1}/{len(keep)} готова', flush=True)

    print(f'\n{len(keep)} мереж × {len(SEEDS)} жеребів, істина {TRUTH_N}')
    print(f'\n{"шар":>4s} {"r_s ЯК ЗАРАЗ":>13s} {"r_s З G":>10s} '
          f'{"β ЯК ЗАРАЗ":>12s} {"β З G":>9s}')
    for i in (1, 2, 4, 8, 12, 16, 20):
        r0, rg = np.mean(rs['ЯК ЗАРАЗ'][i]), np.mean(rs['З G'][i])
        print(f'{i:4d} {r0:13.2f} {rg:10.2f} {1/(1+r0*r0):12.3f} '
              f'{1/(1+rg*rg):9.3f}')

    base = float((np.array(E[keys[0]]) ** 2).mean())
    print(f'\n{"варіант":32s} {"MSE":>11s} {"×":>8s}')
    print('-' * 54)
    for kk in keys:
        m = float((np.array(E[kk]) ** 2).mean())
        print(f'{kk:32s} {m:11.4e} {base/m:8.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
