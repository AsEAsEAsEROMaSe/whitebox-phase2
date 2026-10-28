"""ДРАБИНА: НАСКІЛЬКИ ТОЧНО ТРЕБА ЗНАТИ `m` І `s₃₁` — ТЗ ЗАМІСТЬ СТЕЛІ.

ЗВІДКИ. `last31.py` дав стелю класу **23.8×** (ідеальні `m` 1..30 + `s` 1..31,
крос-ціль). Але це ОРАКУЛ. Правило [[feedback-measure-the-gain-not-the-ceiling]]:
за тиждень сім стель, з них жодна не сказала ЩО РОБИТИ. Драбина каже.

ЩО РАХУЄМО. `m_k = m_іст + k_m·(m_замик − m_іст)`, `s_k` аналогічно з `k_s`.
🔑 ОБИДВА канали ЛІНІЙНІ за переданим значенням:
   `S[i−1] = φ(m̂/ŝ)·(ŝ − s_передане)`,  `A[i−1] = x̄_ℓ − m_передане`,
а перенос до виходу теж лінійний і від `s`,`m` не залежить ⟹
   `S(k) = (1−k)·S_іст + k·S_замик`,  `A(k) = (1−k)·A_іст + k·A_замик`.
⟹ УСЯ сітка `k` будується з ЧОТИРЬОХ матриць, БЕЗ зайвих прогонів.

ОЧІКУВАННЯ ДО ПРОГОНУ: вимога до `m` ЖОРСТКІША за вимогу до `s₃₁` — `m` входить
у `E[relu]` через `Φ(t)`, `s` через `φ(t)`, чутливість менша. Якщо навпаки —
це новина, і хороша: `s` ЄДИНИЙ, кого можна оцінити з ВИБІРКИ.

ПЕЧАТКИ (кути сітки — вже відомі числа з `last31.py`)
  * `k_m=k_s=1` (усе замикання) → мусить дати **1.007×**;
  * `k_m=k_s=0` (усе ідеальне)  → мусить дати **23.775×**;
  * ⛔НЕГ: регресори жеребу `d−1` проти помилки жеребу `d` → ≈1.0;
  * 🚨 КРОС-ЦІЛЬ обов'язкова: з однією істиною стеля приладу 43× (`last31.py` §1).
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
KGRID = [1.0, 0.5, 0.25, 0.1, 0.0]

SIG31 = list(range(0, 31))
MU30 = list(range(KS, KS + 30))
IDX = SIG31 + MU30            # `m` до 30 (31 — тавтологія), `s` до 31


def build(S_A, S_cl, A_A, A_cl, km, ks, ks31=None):
    """rows(k). ks31 — окреме `k` для σ-рядка шару 31 (None ⟹ = ks)."""
    S = (1.0 - ks) * S_A + ks * S_cl
    if ks31 is not None:
        S = S.clone()
        S[30] = (1.0 - ks31) * S_A[30] + ks31 * S_cl[30]
    A = (1.0 - km) * A_A + km * A_cl
    return torch.cat([S, A], 0)


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    # сітка (km, ks, ks31)
    VAR = [(km, ks, None) for km in KGRID for ks in KGRID]
    NG = len(VAR)
    # 🔑 СПЕЦВАРІАНТИ: скільки коштує САМЕ `s₃₁`, коли решта `s` — замикальна
    SPEC = [('🆕лише s₃₁ ідеальне, s 1..30 замикальні, m ідеальне',
             0.0, 1.0, 0.0),
            ('🆕те саме, але m замикальне (як у сабміті)', 1.0, 1.0, 0.0),
            ('⚠️m ідеальне, ВСЯ s замикальна', 0.0, 1.0, 1.0),
            ('⚠️m ідеальне, s 1..30 ідеальні, s₃₁ замикальна', 0.0, 0.0, 1.0)]
    VAR += [(a, b, c) for _, a, b, c in SPEC]
    nv = len(VAR)

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
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)   # 🚨 КРОС-ЦІЛЬ
            s_A, m_A = TA['s'][:KS], TA['m'][:KS]

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
                _, S_A = run_side1(wc, npair, s_A, m_cl, dev, sd,
                                   collect_sigma=True)
                _, A_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                    lam_s=None, collect_anchors=True)
                _, A_A = run_side1(wc, npair, s_cl, m_A, dev, sd,
                                   lam_s=None, collect_anchors=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                for v, (km, ks, ks31) in enumerate(VAR):
                    R = build(S_A, S_cl, A_A, A_cl, km, ks, ks31)
                    gg[v] += R @ R.T
                    bb[v] += R @ e
                RI = build(S_A, S_cl, A_A, A_cl, 0.0, 0.0, None)
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
    np.savez('/mnt/win_d/llm/White-Box/side66/logs/ladder31.npz',
             base=base, C=np.array(C),
             G=np.stack([torch.stack(g).cpu().numpy() for g in G]),
             B=np.stack([torch.stack(b).cpu().numpy() for b in B]),
             var=np.array([[a, b, -1 if c is None else c] for a, b, c in VAR]))

    def val_of(v):
        return loo([g[IDX][:, IDX] for g in G[v]],
                   [b[IDX] for b in B[v]], C, len(IDX))

    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. База = САБМІТ. '
          f'`λ` LOO ПО МЕРЕЖАХ.\n')
    print('ДРАБИНА (сума). Рядок = `k_m` (точність `m` 1..30), '
          'стовпець = `k_s` (точність `s` 1..31). k=0 ідеальне, k=1 замикання.')
    print(f'{"k_m \\ k_s":>10s}' + ''.join(f'{k:>10.2f}' for k in KGRID))
    print('-' * 62)
    for i, km in enumerate(KGRID):
        row = ''
        for j, _ks in enumerate(KGRID):
            s, _md, _w3 = three_slices(base, val_of(i * len(KGRID) + j))
            row += f'{s:9.3f}×'
        print(f'{km:>10.2f}' + row)
    print('\n✅ПЛОМБИ: кут (1,1) мусить дати 1.007×, кут (0,0) — 23.775×\n')
    print('МЕДІАНА для тих самих кутів і спецваріанти:')
    for q, (nm, km, ks, ks31) in enumerate(SPEC):
        s, md, w3 = three_slices(base, val_of(NG + q))
        print(f'{nm:>52s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    vN = loo([g[IDX][:, IDX] for g in Gn], [b[IDX] for b in Bn], Cn, len(IDX))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ (зсув жеребу)":>52s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
