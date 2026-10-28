"""ГІЛКА 1: СКІЛЬКИ ШАРІВ КОРИГУВАТИ — СВІП K З ЧЕСНИМ ПЕРЕФІТОМ λ.

ЗВІДКИ. С66 порахувала: 32 незалежні джерела по 2–4% ⟹ при K=8 стеля 1.60×,
а беремо 1.165× = 38% доступного ⟹ «у вже відкритій гілці ще ~37% балу».
С67 показала, ЧОМУ попередні свіпи нічого не міряли: **свіп K без перефіту λ
не вимірює нічого**, бо λ підігнані під K=8 і компенсують одне одного
(K=6 вийшов ГІРШИМ за K=4). Отже єдиний чесний спосіб — для КОЖНОГО K
підганяти свій власний набір λ, і підганяти його **не на тих самих мережах**.

ЩО РОБИМО. Стек якорів `ℓ = 0…K` рівно як у sub70/72 (шар 0 ТОЧНИЙ `σ/√2π`,
далі `m_ℓ` із гаусового замикання), λ — вектор довжини K+1, підганяється
**LOO по мережах**: для мережі `n` беруться λ, злічені з УСІХ ІНШИХ мереж.
Це знімає головну пастку («підгін і перевірка на тих самих даних»), через яку
«найкраща драбина» в `mlmc_depth.py` дала міраж 1.48× проти чесних 0.96×.

    залишок(n) = ‖e_out‖² − 2λᵀb_n + λᵀG_nλ,   λ = (Σ_{m≠n}G_m)⁻¹(Σ_{m≠n}b_m)
    ВИГРАШ(K) = Σ_n‖e_out‖² / Σ_n залишок(n)

КОНТРОЛІ
  * ✅ K=0 (лише точний шар 0) мусить відтворити базову лінію sub66;
  * ✅ поруч друкуємо ОРАКУЛЬНИЙ λ (підігнаний на тій самій мережі) — розрив
    між ним і LOO показує, скільки з виграшу є перевчанням;
  * ⛔ λ від ПЕРЕМІШАНИХ мереж — мусить бути ≈ K=0 або гірше;
  * вибірка РІВНО продакшн-розміру (npair із `cut_net`), 8 мереж для LOO;
  * друкуємо число обумовленості `G`: якщо якорі майже колінеарні, зростання
    K дає шум, і це видно ДО висновку.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import GATE_N, INV_SQRT_2PI                 # noqa: E402
from transport_r2 import cut_net                             # noqa: E402
from anchor_depth import truth_means, closure_means, one_draw, transport  # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NDRAW = 96
KS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 30]
KMAXA = 30


def main(nets=8, dev=DEV):
    torch.manual_seed(1861)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    nA = KMAXA + 1
    B = [np.zeros(nA) for _ in range(nets)]
    G = [np.zeros((nA, nA)) for _ in range(nets)]
    base = np.zeros(nets)

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_net(W, dev)
            T = truth_means(wc, dev, 610_000 + 7919 * n)
            M = closure_means(wc, dev)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() * INV_SQRT_2PI
            for d in range(NDRAW):
                xb, gates = one_draw(wc, dev, 7_000_000 + 149 * n + d, npair)
                e = xb[-1] - T[-1]
                base[n] += float((e * e).sum())
                cols = []
                for a in range(nA):
                    mm = m0 if a == 0 else M[a]
                    cols.append(transport(xb[a] - mm, gates, wc, a))
                gv = np.array([float((e * c).sum()) for c in cols])
                Gm = np.array([[float((ci * cj).sum()) for cj in cols]
                               for ci in cols])
                B[n] += gv
                G[n] += Gm
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … мережа {n+1}/{nets} (npair={npair})', flush=True)

    tot = base.sum()
    print(f'\n{nets} мереж × {NDRAW} жеребів продакшн-розміру; '
          f'λ підганяються LOO ПО МЕРЕЖАХ')
    print(f'{"K":>4s} {"LOO виграш":>11s} {"ОРАКУЛ λ":>10s} '
          f'{"⛔перемішані λ":>15s} {"cond(G)":>10s}')
    print('  ' + '-' * 56)
    perm = np.random.default_rng(5).permutation(nets)
    for K in KS:
        s = slice(0, K + 1)
        res_loo, res_or, res_sh = 0.0, 0.0, 0.0
        cond = 0.0
        for n in range(nets):
            Gn, bn = G[n][s, s], B[n][s]
            Go = sum(G[m][s, s] for m in range(nets) if m != n)
            bo = sum(B[m][s] for m in range(nets) if m != n)
            rg = 1e-10 * np.trace(Go) / max(K + 1, 1) * np.eye(K + 1)
            lam = np.linalg.solve(Go + rg, bo)
            res_loo += base[n] - 2 * lam @ bn + lam @ Gn @ lam
            lo = np.linalg.solve(Gn + 1e-10 * np.trace(Gn) /
                                 max(K + 1, 1) * np.eye(K + 1), bn)
            res_or += base[n] - 2 * lo @ bn + lo @ Gn @ lo
            m2 = perm[n]
            Gs, bs = G[m2][s, s], B[m2][s]
            ls = np.linalg.solve(Gs + 1e-10 * np.trace(Gs) /
                                 max(K + 1, 1) * np.eye(K + 1), bs)
            res_sh += base[n] - 2 * ls @ bn + ls @ Gn @ ls
            cond = max(cond, float(np.linalg.cond(Gn)))
        print(f'{K:4d} {tot/max(res_loo,1e-300):10.3f}× '
              f'{tot/max(res_or,1e-300):9.3f}× '
              f'{tot/max(res_sh,1e-300):14.3f}× {cond:10.2e}')
    print('\n✅ K=0 = базова лінія sub66 (лише точний якір шару 0)')
    print('🔑 читати так: якщо LOO РОСТЕ з K — у гілці справді є запас; '
          'якщо ОРАКУЛ росте, а LOO ні — то перевчання, і запасу нема')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
