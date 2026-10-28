"""ПЕРЕМІР СВІПУ K НА ЗВІРЕНІЙ БАЗІ (`base_real`, повний ланцюг сабміту).

ЗВІДКИ. `k_sweep_refit.py` дав «оптимум K=2–3, 1.405 проти 1.381 при K=8»,
і на цьому я двічі радив сабміт. Але той стенд рахував ланцюг БЕЗ
ВИБІЛЮВАННЯ — та сама вада, що перетворила нахил 1.104× на 1.003×.
Тут — на `base_real.py`, звіреному зі справжнім `estimator_side1.py` до
**1.2e−06** (округлення fp32).

ЩО МІРЯЄМО. Пошаровий μ-якір `λ_k·(x̄_k − m_замик,k)` ПОВЕРХ діючого сабміту
(вибілювання + антитетика + якір-0 + σ-поправка `_LAM_S`). Ланцюг переносу
лінійний за `v`, тож усі якорі несуться одним матричним множенням поруч —
`pred(λ) = pred_база − λ·A`, і λ фітується **LOO ПО МЕРЕЖАХ**.

КОНТРОЛІ
  * ✅ K=0 (λ=0) мусить дати рівно базу — це нинішній сабміт;
  * ✅ ОРАКУЛЬНЕ λ (на тій самій мережі) друкуємо поруч: розрив з LOO = перевчання;
  * ⛔ λ від ПЕРЕМІШАНИХ мереж — мусить бути ≈K=0 або гірше;
  * cond(G) друкуємо: вибух = колінеарність якорів.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts, run_side1, KSIG, W_
from tilt_wide import truth_last

L_, DEV, NDRAW = 32, 'cuda:3', 40
KS = [0, 1, 2, 3, 4, 6, 8]


def main(nets=16, dev=DEV):
    torch.manual_seed(31337)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    B = [np.zeros(KSIG) for _ in range(nets)]
    G = [np.zeros((KSIG, KSIG)) for _ in range(nets)]
    base = np.zeros(nets)
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KSIG, dev)
            T = truth_last(wc, dev, 810_000 + 7919 * n)
            for d in range(NDRAW):
                pr, A = run_side1(wc, npair, s_cl, m_cl, dev,
                                  55_000_000 + 971 * n + d, collect_anchors=True)
                e = pr - T
                base[n] += float((e * e).sum())
                B[n] += (A @ e).cpu().numpy()
                G[n] += (A @ A.T).cpu().numpy()
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets} (npair={npair})', flush=True)

    tot = base.sum()
    perm = np.random.default_rng(11).permutation(nets)
    print(f'\n{nets} мереж × {NDRAW} жеребів на ЗВІРЕНІЙ базі '
          f'(порт ≡ сабміт до 1.2e−06)')
    print(f'{"K":>3s} {"LOO виграш":>11s} {"ОРАКУЛ λ":>10s} '
          f'{"⛔перемішані":>13s} {"cond(G)":>10s}')
    for K in KS:
        if K == 0:
            print(f'{0:3d} {1.0:10.4f}× {1.0:9.4f}× {1.0:12.4f}× '
                  f'{1.0:10.2e}  ✅ = НИНІШНІЙ САБМІТ')
            continue
        s = slice(0, K)
        rl = ro = rs = 0.0
        cond = 0.0
        for n in range(nets):
            Gn, bn = G[n][s, s], B[n][s]
            Go = sum(G[m][s, s] for m in range(nets) if m != n)
            bo = sum(B[m][s] for m in range(nets) if m != n)
            rg = 1e-10 * np.trace(Go) / K * np.eye(K)
            lam = np.linalg.solve(Go + rg, bo)
            rl += base[n] - 2 * lam @ bn + lam @ Gn @ lam
            lo = np.linalg.solve(Gn + 1e-10 * np.trace(Gn) / K * np.eye(K), bn)
            ro += base[n] - 2 * lo @ bn + lo @ Gn @ lo
            m2 = perm[n]
            ls = np.linalg.solve(G[m2][s, s] + 1e-10 * np.trace(G[m2][s, s])
                                 / K * np.eye(K), B[m2][s])
            rs += base[n] - 2 * ls @ bn + ls @ Gn @ ls
            cond = max(cond, float(np.linalg.cond(Gn)))
        print(f'{K:3d} {tot/max(rl,1e-300):10.4f}× {tot/max(ro,1e-300):9.4f}× '
              f'{tot/max(rs,1e-300):12.4f}× {cond:10.2e}')
    print('\n🔑 K=0 — це рівно те, що зараз на лідерборді (2.0605e-07). '
          'Сабмітити варто лише те, де LOO помітно > 1.0')
    print('\nГОТОВІ λ (фіт на ВСІХ мережах — саме їх зашивати в сабміт):')
    for K in (1, 2, 4, 8):
        s2 = slice(0, K)
        Ga = sum(G[m][s2, s2] for m in range(nets))
        ba = sum(B[m][s2] for m in range(nets))
        lam = np.linalg.solve(Ga + 1e-10 * np.trace(Ga) / K * np.eye(K), ba)
        print(f'  K={K}: _LAM_A = ({", ".join(f"{v:.5f}" for v in lam)}'
              + (', ' + ', '.join('0.0' for _ in range(KSIG - K)) if K < KSIG else '')
              + ')')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 16)
