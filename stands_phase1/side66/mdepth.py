"""ДЕ САМЕ ТРЕБА ТОЧНЕ `m`: ПРОФІЛЬ ВНЕСКУ ШАРІВ У ВИГРАШ.

ЗВІДКИ. `ladder31.py`: усе впирається в точність `m` замикання (стовпець
`k_s=1`: вдвічі точніше → 1.176×, вчетверо → 1.694×, ідеально → 14.893×).
Але «точніше `m` на 30 шарах» — це не ТЗ, бо шари коштують по-різному:
мілке замикання і точніше (шар 1 ТОЧНИЙ аналітично), і дешевше (рекурсія `C`
до шару 8 = 8 матмулів проти 31).

🔑 ПИТАННЯ: виправлення ЯКИХ шарів дає виграш? Рекурсія похибки виміряна:
`b_ℓ = Φ(t)(b_{ℓ−1}@W) + φ(t)δs + e_лок`, перенос 93.9%, локальне джерело 2.3%
⟹ зміщення переважно УСПАДКОВАНЕ. Але `b` РОСТЕ з глибиною.

ОЧІКУВАННЯ ДО ПРОГОНУ: важитимуть ГЛИБОКІ шари (там `b` найбільше). Якщо
навпаки — мілкі, то відкривається ДЕШЕВИЙ шлях, бо сабміт уже якорить 1..8.

ЯК. Та сама лінійність, що в `ladder31.py`: `A(k) = (1−k)A_іст + k·A_замик`,
але `k` тепер ВЕКТОР по шарах ⟹ будь-яка підмножина шарів задарма.
`k_s = 1` скрізь (вся `s` замикальна) — щоб бачити ЧИСТИЙ `m`-канал.

ПЕЧАТКИ: «жоден шар» → 1.007×; «усі 30» → 14.893× (з `ladder31.py`);
⛔НЕГ → ≈1.0; 🚨крос-ціль обов'язкова.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
KS = 31
SIG31 = list(range(0, 31))
MU30 = list(range(KS, KS + 30))
IDX = SIG31 + MU30

# (назва, множина шарів 1..30, де `m` ІДЕАЛЬНЕ)
SETS = [('⛔жоден шар (пломба 1.007×)', []),
        ('мілкі 1..4', list(range(1, 5))),
        ('мілкі 1..8  ← сабміт якорить саме тут', list(range(1, 9))),
        ('мілкі 1..16', list(range(1, 17))),
        ('глибокі 15..30', list(range(15, 31))),
        ('глибокі 23..30', list(range(23, 31))),
        ('глибокі 27..30', list(range(27, 31))),
        ('лише 30 (один найглибший)', [30]),
        ('лише 29..30', [29, 30]),
        ('✅усі 1..30 (пломба 14.893×)', list(range(1, 31)))]


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    nv = len(SETS)
    masks = []
    for _nm, ls in SETS:
        w = torch.zeros(KS, 1, dtype=torch.float64, device=dev)
        for i in ls:
            w[i - 1, 0] = 1.0            # 1 ⟹ брати ІДЕАЛЬНЕ `m` на цьому шарі
        masks.append(w)

    base_mse, C = [], []
    G = [[] for _ in range(nv)]
    B = [[] for _ in range(nv)]
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)
            m_A = TA['m'][:KS]

            bm = 0.0
            nreg = 2 * KS
            gg = torch.zeros(nv, nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     collect_sigma=True)
                _, A_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                    lam_s=None, collect_anchors=True)
                _, A_A = run_side1(wc, npair, s_cl, m_A, dev, sd,
                                   lam_s=None, collect_anchors=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                for v in range(nv):
                    A = masks[v] * A_A + (1.0 - masks[v]) * A_cl
                    R = torch.cat([S_cl, A], 0)
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                RI = torch.cat([S_cl, A_A], 0)
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = RI
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    np.savez('/mnt/win_d/llm/White-Box/side66/logs/mdepth.npz',
             base=base, C=np.array(C),
             G=np.stack([torch.stack(g).cpu().numpy() for g in G]),
             B=np.stack([torch.stack(b).cpu().numpy() for b in B]),
             names=np.array([nm for nm, _ in SETS]))
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. `k_s=1` (вся `s` '
          f'замикальна) ⟹ ЧИСТИЙ `m`-канал.\n')
    print(f'{"де `m` ІДЕАЛЬНЕ":>42s} {"шарів":>6s} {"сума":>9s} '
          f'{"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 80)
    for v, (nm, ls) in enumerate(SETS):
        val = loo([g[IDX][:, IDX] for g in G[v]],
                  [b[IDX] for b in B[v]], C, len(IDX))
        s, md, w3 = three_slices(base, val)
        print(f'{nm:>42s} {len(ls):6d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    vN = loo([g[IDX][:, IDX] for g in Gn], [b[IDX] for b in Bn], Cn, len(IDX))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ (зсув жеребу)":>42s} {30:6d} {s:8.3f}× {md:8.3f}× '
          f'{w3:9.3f}×')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
