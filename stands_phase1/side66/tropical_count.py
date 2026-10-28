"""ЧИ Є КОМБІНАТОРИКА РОЗРІЗУ МАЛОЮ. (ідея юзера: тропічна геометрія)

ТВЕРДЖЕННЯ, ЯКЕ ПЕРЕВІРЯЄМО. ReLU-мережа — тропічний поліном; простір різано
гіперплощинами перемикання на лінійні регіони (камери). Якщо «шість напрямків»
мають комбінаторне тлумачення — кілька домінантних камер/граней — то розріз
має бути МАЛИМ: більшість затворів замерзлі, перемикається жменя нейронів,
і камери мають вимірну масу.

ПРОТИЛЕЖНА ГІПОТЕЗА (арифметична). С62 виміряла ентропію маски шару 0 = 218.5
біта ⟹ ~2²¹⁸ візерунків, кожен семпл у власній камері, домінантних камер нема.
Тоді тропічна мова описує об'єкт правильно, але важеля не дає.

ЩО МІРЯЄМО, ПОШАРОВО (жодної істини, один прогін)
  * `g_j = P(p_j > 0)` — частка семплів, де нейрон відкритий;
  * скільки нейронів СПРАВДІ перемикаються: `0.05 < g < 0.95`;
  * ентропія маски шару в бітах: `Σ_j H(g_j)` (верхня оцінка при незалежності)
    і МАКСИМАЛЬНА маса однієї камери серед семплів;
  * ⟹ ефективне число камер `2^H`.

ГОЛОВНЕ ПИТАННЯ — ЧИ КОМБІНАТОРНІ 6 НАПРЯМКІВ. Беремо топ-6 ляпуновських
напрямків (з ваг, `lyapunov_dirs`) і питаємо, на чому вони живуть:
    перекриття з підпростором k НАЙБІЛЬШ перемикних нейронів
    перекриття з підпростором k НАЙБІЛЬШ замерзлих
    ⛔ перекриття з k ВИПАДКОВИМИ нейронами          ← контроль
Якщо напрямки сидять на жмені перемикних — теза юзера жива й указує, де
рахувати точно. Якщо вони розмазані як випадкові — тропіка важеля не дає.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (INV_SQRT_2PI, MIN_PAIRS, PILOT_PAIRS,  # noqa: E402
                         TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                                # noqa: E402
from born_source import _Phi                                    # noqa: E402
from closure_precision import NQ, relu_cov0, relu_cov_gauss     # noqa: E402
from lyapunov_dirs import closure_chain, transport, top_right, overlap  # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
UPTO = 31
LS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 31)


def gates(wc, seed, dev):
    """`g_j = P(p_j>0)` кожного шару на повній вибірці оцінювача."""
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
    out, cell = {}, {}
    for i in range(1, L_):
        p = x @ wc[i]
        M = (p > 0)
        out[i] = M.to(torch.float64).mean(0)
        if i in LS:
            # маса найбільшої камери шару: найчастіший повний візерунок
            key = (M[:, :62].to(torch.int64) *
                   (2 ** torch.arange(62, device=p.device))).sum(1)
            cell[i] = float(torch.bincount(
                torch.unique(key, return_inverse=True)[1]).max()) / n2
        x = torch.relu(p)
    return out, cell, n2


def sub_of(idx, n, dev):
    P = torch.zeros(n, idx.shape[0], dtype=torch.float64, device=dev)
    P[idx, torch.arange(idx.shape[0], device=dev)] = 1.0
    return P


def main(nets=8, dev=DEV, ktop=6):
    torch.manual_seed(7)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    sw = {i: [] for i in LS}
    ent = {i: [] for i in LS}
    cl = {i: [] for i in LS}
    ov = {k: [] for k in ('перемикні', 'замерзлі', '⛔випадкові')}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            g, cell, n2 = gates(wc, 400_000, dev)
            for i in LS:
                q = g[i].clamp(1e-12, 1 - 1e-12)
                sw[i].append(float(((q > 0.05) & (q < 0.95)).sum()))
                ent[i].append(float((-q * q.log2()
                                     - (1 - q) * (1 - q).log2()).sum()))
                cl[i].append(cell[i])

            clo = closure_chain(wc, nodes, wts, UPTO)
            gg = {i: _Phi(clo[i][0] / clo[i][1]) for i in range(1, L_ - 1)}
            phi_last = _Phi(clo[UPTO][0] / clo[UPTO][1])
            P1 = top_right(transport(wc, gg, 1, phi_last), ktop)
            nw = phi_last.shape[0]
            gl = g[UPTO][:nw]
            hh = -(gl.clamp(1e-12, 1 - 1e-12) * gl.clamp(1e-12, 1 - 1e-12).log2())
            order = torch.argsort(hh, descending=True)
            k = 32
            ov['перемикні'].append(overlap(P1, sub_of(order[:k], nw, dev)))
            ov['замерзлі'].append(overlap(P1, sub_of(order[-k:], nw, dev)))
            ov['⛔випадкові'].append(overlap(
                P1, sub_of(torch.randperm(nw, device=dev)[:k], nw, dev)))
        print(f'  мережа {n+1}/{len(fs)}  перемикних(шар31)={sw[31][-1]:.0f} '
              f'ентропія={ent[31][-1]:.0f} біт', flush=True)

    print(f'\n{len(fs)} мереж, вибірка {n2:.0f} семплів')
    print('\n1. НАСКІЛЬКИ ВЕЛИКИЙ РОЗРІЗ')
    print(f'{"шар":>4s} {"перемикних із 256":>18s} {"ентропія маски, біт":>20s} '
          f'{"камер 2^H":>12s} {"макс.маса камери":>17s}')
    for i in LS:
        e = np.mean(ent[i])
        print(f'{i:4d} {np.mean(sw[i]):18.1f} {e:20.1f} {2.0**min(e,300):12.2e} '
              f'{np.mean(cl[i]):17.2e}')

    print(f'\n2. ЧИ КОМБІНАТОРНІ 6 НАПРЯМКІВ (перекриття топ-{ktop} ляпуновських '
          f'із підпростором 32 нейронів; випадкове = {32/256:.3f})')
    for k in ('перемикні', 'замерзлі', '⛔випадкові'):
        print(f'   {k:14s}: {np.mean(ov[k]):.3f} ± {np.std(ov[k]):.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
