"""ТОЧНІСТЬ ІЗ ТОГО САМОГО ЯДРА: КЗ НА СТАТИСТИКАХ, ЩО ВЖЕ ОБЧИСЛЕНІ.

МЕТА (юзер): витиснути ТОЧНІСТЬ, не бюджет. Тобто ρ² вище при ТІЙ САМІЙ ціні.

🔑 ІДЕЯ. З кожного шару ми беремо лише `m̂` (μ-якір) і `ŝ` (σ-канал). Але в
ТОМУ САМОМУ проході вже лежать:
  * `q̂ = частка активних` (`ind`) — ціль `Φ(t)` із замикання;
  * `â² = E[relu²]` (`a2`)      — ціль `(m²+s²)Φ + m·s·φ` із замикання.
Обидві коштують `O(N·n)` проти `O(N·n²)` за матмул ⟹ **0.8% ціни шару**, а їхні
аналітичні цілі дає ТА САМА рекурсія (`closure_parts(extras=True)`) — БЕЗ
жодного зайвого множення. Тобто канал практично БЕЗКОШТОВНИЙ.

🚨 ВІДМІННІСТЬ ВІД ⛔`perfect2.py` (там вищі моменти дали +2% «у шумі»): там
вони лягали ПОВЕРХ ІДЕАЛЬНИХ `m,s`, де вже нічого не лишалось. Тут — поверх
ЗАМИКАЛЬНИХ, тобто поганих, і можуть нести НЕЗАЛЕЖНУ інформацію.

ОЧІКУВАННЯ ДО ПРОГОНУ: **1.02–1.10×**. ПОРІГ вартості сабміта: **1.05×**.
🚨 <12% ⟹ вирок остаточно виносить ЛИШЕ ГРЕЙДЕР.

ПЕЧАТКИ: ⛔шум тієї самої норми мусить сидіти на базі · ⛔НЕГ (зсув жеребу) ≈1.0
· KSIG=8 як у сабміті ⟹ нуль додаткових флопів замикання.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, KSIG)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import loo, three_slices                             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
K = KSIG                      # 8 — рівно як у сабміті

SIG = list(range(0, K))               # σ-канал
MU = list(range(K, 2 * K))            # μ-якір ('a')
IND = list(range(2 * K, 3 * K))       # частка активних ('ind')
A2 = list(range(3 * K, 4 * K))        # другий момент ('a2')

VAR = [('⛔σ+μ (нинішня КЗ, вільний λ)', SIG + MU),
       ('🎯+ q̂ (частка активних)', SIG + MU + IND),
       ('🎯+ â² (другий момент)', SIG + MU + A2),
       ('🎯+ обидві', SIG + MU + IND + A2),
       ('⛔ШУМ тієї самої норми', None)]


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

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
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, K, dev, extras=True)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)   # КРОС-ЦІЛЬ
            kinds = ['a', 'ind', 'a2']
            refs = [m_cl, f_cl, q_cl]

            bm = 0.0
            nreg = 4 * K
            gg = torch.zeros(nv, nreg, nreg, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, nreg, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            g = torch.Generator(device=dev); g.manual_seed(9090 + n)

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  collect_sigma=True)
                _, A = run_side1(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                 collect_anchors=True,
                                 anchor_multi=(kinds, refs))
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                R = torch.cat([S, A], 0)
                assert R.shape[0] == nreg, R.shape
                # ⛔ШУМ: замінити ДОДАНІ рядки шумом тієї самої норми
                Rn = R.clone()
                nz = torch.randn(2 * K, R.shape[1], generator=g, device=dev,
                                 dtype=torch.float64)
                nz = nz * (R[2 * K:].norm() / nz.norm().clamp_min(1e-30))
                Rn[2 * K:] = nz
                for v, (_nm, idx) in enumerate(VAR):
                    RR = Rn if idx is None else R
                    gg[v] += RR @ RR.T
                    bb[v] += RR @ e
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = R
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    np.savez('/mnt/win_d/llm/White-Box/side66/logs/freecv.npz',
             base=base, C=np.array(C),
             G=np.stack([torch.stack(g).cpu().numpy() for g in G]),
             B=np.stack([torch.stack(b).cpu().numpy() for b in B]))
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ. База = САБМІТ. '
          f'KSIG={K} (нуль додаткових флопів).\n')
    print(f'{"варіант":>34s} {"регр":>5s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 72)
    ref = None
    for v, (nm, idx) in enumerate(VAR):
        ii = (SIG + MU + IND + A2) if idx is None else idx
        val = loo([g[ii][:, ii] for g in G[v]],
                  [b[ii] for b in B[v]], C, len(ii))
        s, md, w3 = three_slices(base, val)
        if v == 0:
            ref = s
        print(f'{nm:>34s} {len(ii):5d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×'
              + ('' if v == 0 else f'   ⟹ {s/ref:.3f}× над КЗ'))
    ii = SIG + MU + IND + A2
    vN = loo([g[ii][:, ii] for g in Gn], [b[ii] for b in Bn], Cn, len(ii))
    s, md, w3 = three_slices(base, vN)
    print(f'\n{"⛔НЕГ (зсув жеребу)":>34s} {len(ii):5d} {s:8.3f}× {md:8.3f}× '
          f'{w3:9.3f}×')
    print('\n🎯 ПОРІГ: >1.05× над нинішньою КЗ ⟹ кандидат на сабміт.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
