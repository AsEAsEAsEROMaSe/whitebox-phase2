"""ЧИ Є ВИДІЛЕНИЙ ШАР ВИРОДЖЕННЯ. (ідея юзера: поля Якобі / спряжені точки)

ТВЕРДЖЕННЯ. Промінь у неоднорідному середовищі — геодезична; чи розбіжаться
сусідні промені, каже рівняння Якобі, а каустика — СПРЯЖЕНА ТОЧКА, де поле
Якобі обертається в нуль. Рівняння Ріккаті на матрицю розбігання ЛОКАЛЬНЕ по
глибині ⟹ мало б указати шар виродження, не проганяючи всю мережу. Емпіричний
кандидат у нас є — восьмий шар (С65). Збіг двох незалежних шляхів був би
сильнішим свідченням, ніж будь-який окремий вимір.

ЩО РАХУЄМО (з самих ваг, затвори із замикання; жодної істини, жодного семпла)
    ЛОКАЛЬНЕ (аналог Ріккаті — один крок):  N_k = G_k W_{k+1}
        PR(N_k), σ₁/σ₂(N_k), ln σ₁(N_k)      ← де саме крок вироджується
    ГЛОБАЛЬНЕ (накопичене):  M_k = W_{k+1} G_{k+1} … W_31
        PR(M_k) і його ПРИРІСТ по k          ← де саме накопичується колапс
Спряжена точка мала б дати ПІК приросту на одному шарі. Гладкий профіль
означає, що вироджується скрізь потроху й ставити одну поправку нікуди.

КОНТРОЛІ
  * ⛔ де максимум приросту НАСПРАВДІ — друкуємо аргмакс по кожній мережі
    окремо; якщо він стрибає між мережами, це шум, а не спряжена точка;
  * ⛔ затвори, перемішані ВСЕРЕДИНІ шару — нульова гіпотеза;
  * поруч — скільки сингулярних чисел ВІДІРВАНО від суцільної маси
    (σ_i > 3×медіани) — питання «закону краю» з того ж спектра.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                               # noqa: E402
from born_source import _Phi                                   # noqa: E402
from closure_precision import NQ                               # noqa: E402
from lyapunov_dirs import closure_chain, transport             # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
UPTO = 31


def pr_of(sv):
    e = (sv * sv).double()
    return float(e.sum() ** 2 / (e * e).sum())


def main(nets=8, dev=DEV):
    torch.manual_seed(31337)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    KS = list(range(1, L_ - 1))
    loc = {k: [] for k in KS}
    gap = {k: [] for k in KS}
    glob_pr = {k: [] for k in KS}
    edge = {k: [] for k in KS}
    dpr, dpr_sh, arg, arg_sh = {k: [] for k in KS}, {k: [] for k in KS}, [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            clo = closure_chain(wc, nodes, wts, UPTO)
            g = {i: _Phi(clo[i][0] / clo[i][1]) for i in range(1, L_ - 1)}
            gs = {i: g[i][torch.randperm(g[i].shape[0], device=dev)]
                  for i in range(1, L_ - 1)}
            phi_last = _Phi(clo[UPTO][0] / clo[UPTO][1])

            for k in KS:
                N = g[k].unsqueeze(1) * wc[k + 1].double()
                sv = torch.linalg.svdvals(N)
                loc[k].append(pr_of(sv))
                gap[k].append(float(sv[0] / sv[1]))
                sm = torch.linalg.svdvals(transport(wc, g, k, phi_last))
                glob_pr[k].append(pr_of(sm))
                edge[k].append(float((sm > 3.0 * sm.median()).sum()))
            for d, gg in ((dpr, g), (dpr_sh, gs)):
                p = [pr_of(torch.linalg.svdvals(transport(wc, gg, k, phi_last)))
                     for k in KS]
                for j, k in enumerate(KS[:-1]):
                    d[k].append(p[j + 1] - p[j])
                (arg if gg is g else arg_sh).append(
                    KS[int(np.argmax(np.diff(p)))])
        print(f'  мережа {n+1}/{len(fs)}  аргмакс приросту PR: шар {arg[-1]} '
              f'(⛔перемішані {arg_sh[-1]})', flush=True)

    print(f'\n{len(fs)} мереж, усе з ваг (затвори із замикання)')
    print(f'{"шар":>4s} {"ЛОК PR(N_k)":>12s} {"ЛОК σ₁/σ₂":>11s} '
          f'{"ГЛОБ PR(M_k)":>13s} {"приріст PR":>12s} {"⛔перемішані":>13s} '
          f'{"відірваних σ":>13s}')
    for k in KS:
        d = f'{np.mean(dpr[k]):12.3f}' if dpr[k] else f'{"—":>12s}'
        ds = f'{np.mean(dpr_sh[k]):13.3f}' if dpr_sh[k] else f'{"—":>13s}'
        print(f'{k:4d} {np.mean(loc[k]):12.1f} {np.mean(gap[k]):11.2f} '
              f'{np.mean(glob_pr[k]):13.2f} {d} {ds} {np.mean(edge[k]):13.1f}')

    print(f'\nаргмакс приросту PR по мережах: {sorted(arg)}')
    print(f'⛔ те саме з перемішаними затворами: {sorted(arg_sh)}')
    print(f'розкид аргмаксу: {np.std(arg):.1f} шара (⛔ {np.std(arg_sh):.1f}); '
          f'якщо він великий — виділеного шару НЕМА')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
