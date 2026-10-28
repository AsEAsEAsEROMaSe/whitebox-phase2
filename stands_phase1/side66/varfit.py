"""ГОЛОВНИЙ ВИМІР: ВАРІАЦІЙНЕ ЗАМИКАННЯ + ПЕРЕФІТ УСІХ ЧОТИРЬОХ РОДИН λ.

════════ ЗВІДКИ ════════
`varsub.py` з РІДНИМИ λ дав ~1.01×, і показав чому: λ сабміту підігнані під
СТАРУ похибку замикання. `deepvar.py` (дві родини, LOO-перефіт) дав
**1.099× проти 1.040×** ⟹ чистий внесок варіаційного замикання **1.057×**.
Тут — те саме, але на ВСІХ ЧОТИРЬОХ родинах діючого `sub72`:
    δs (`_LAM_S`) · δx̄ (`_LAM_A`) · δE[x²] (`_LAM_Q`) · δмаски (`_LAM_G`)
Регресори: `collect_sigma` дає σ-родину, `anchor_multi=['a','ind','a2']` —
решту три (це рівно опори `m_cl`, `f_cl`, `q_cl`).

🚨 ЩО САМЕ Є ВІДПОВІДДЮ. Перефіт λ САМ ПО СОБІ дає виграш (у `deepvar` це
1.040×), і його треба відняти. Тому головне число — відношення
    (α* з перефітом) / (α=1 з перефітом),
обидва — та сама процедура LOO на тому самому зрізі.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 Чистий внесок буде **1.03–1.06×** — не більше, ніж на двох родинах, бо
   `varcal.py` показав: варіаційна ширина покращує `m` (1.49×) і трохи `s`
   (1.15×), а опори з НАЙБІЛЬШИМИ λ — `q` (1.000×, тобто НУЛЬ) і `g`
   (1.044×) — майже не покращуються ⟹ дві нові родини нічого не додадуть.
🟢 ЖИВИЙ при чистому >1.05× на сумі І медіані.

🚨 ПЛОМБИ
  ✅ `α=1` з перефітом мусить бути ≥1.0 проти рідних λ (перефіт не шкодить);
  ⛔ НЕГ: регресори з СУСІДНЬОГО жереба ≈1.000× (32 параметри на 24 мережі —
     переганяння тут головний ризик, і саме НЕГ його ловить).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget                               # noqa: E402
from base72 import run_side1 as run72                              # noqa: E402
from varsub import closure_var, S72, A72, Q72, G72                 # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402
from price31 import loo, three_slices                              # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:2'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 48
K = 8
ALPHAS = (1.0, 0.9975, 0.9950)
KINDS = ['a', 'ind', 'a2']          # опори m_cl, f_cl, q_cl


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'📣 ПЕРЕДБАЧЕНО: чистий внесок 1.03–1.06× (нові дві родини нічого '
          f'не додадуть — `q` калібрувалась у 1.000×)')
    print(f'🟢 ЖИВИЙ при чистому >1.05× на сумі І медіані.  '
          f'мереж {nets} × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    nreg = 4 * K
    base_mse = []
    G = {a: [] for a in ALPHAS}
    Bv = {a: [] for a in ALPHAS}
    C = {a: [] for a in ALPHAS}
    Gn, Bn, Cn = [], [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            CL = {a: closure_var(wc, K, dev, a, s_raw=False) for a in ALPHAS}
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            bm = 0.0
            gg = {a: torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
                  for a in ALPHAS}
            bb = {a: torch.zeros(nreg, dtype=torch.float64, device=dev)
                  for a in ALPHAS}
            cc = {a: 0.0 for a in ALPHAS}
            gN = torch.zeros(nreg, nreg, dtype=torch.float64, device=dev)
            bN = torch.zeros(nreg, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None

            for d in range(ndraw):
                sd = 77_000_000 + 733 * n + d
                s0, m0, f0, q0 = CL[1.0]
                # ✅база = ДІЮЧИЙ sub72: чотири родини з РІДНИМИ λ
                pb, _ = run72(wc, npair, s0, m0, dev, sd, lam_s=S72,
                              lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q0, p_ext=f0)
                bm += float(((pb - T1) * (pb - T2)).sum())
                for a in ALPHAS:
                    s_cl, m_cl, f_cl, q_cl = CL[a]
                    refs = [m_cl, f_cl, q_cl]
                    p0, S = run72(wc, npair, s_cl, m_cl, dev, sd,
                                  collect_sigma=True)
                    _, A = run72(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                                 collect_anchors=True,
                                 anchor_multi=(KINDS, refs))
                    e1, e2 = p0 - T1, p0 - T2
                    e = 0.5 * (e1 + e2)
                    cc[a] += float((e1 * e2).sum())
                    R = torch.cat([S, A], 0)
                    gg[a] += R @ R.T
                    bb[a] += R @ e
                    if a == 1.0:
                        if prevR is not None:
                            gN += prevR @ prevR.T
                            bN += prevR @ e
                            cN += float((e1 * e2).sum())
                        prevR = R
            for a in ALPHAS:
                G[a].append(gg[a]); Bv[a].append(bb[a]); C[a].append(cc[a])
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
            base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {ndraw} жеребів. КРОС-ЦІЛЬ. '
          f'✅база = ДІЮЧИЙ sub72 (4 родини, рідні λ).\n')
    print(f'{"α":>8s} {"родин":>6s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s}')
    print('-' * 48)
    res = {}
    for a in ALPHAS:
        for nf in (2, 4):
            idx = list(range(nf * K))
            val = loo([g[idx][:, idx] for g in G[a]],
                      [b[idx] for b in Bv[a]], C[a], len(idx))
            s, md, w3 = three_slices(base, val)
            res[(a, nf)] = (s, md, w3)
            mark = ' ✅перефіт без α' if a == 1.0 else ''
            print(f'{a:8.4f} {nf:6d} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{mark}')
        print('-' * 48)

    idx = list(range(nreg))
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, nreg)
    s, md, w3 = three_slices(base, vN)
    print(f'{"⛔НЕГ":>15s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×  '
          f'(мусить бути ≈1.000×)')

    print(f'\n🔑 ЧИСТИЙ ВНЕСОК ВАРІАЦІЙНОГО ЗАМИКАННЯ '
          f'(до тієї самої процедури перефіту):')
    best = (0.0, None, None)
    for nf in (2, 4):
        r1 = res[(1.0, nf)]
        for a in ALPHAS:
            if a == 1.0:
                continue
            ra = res[(a, nf)]
            cs, cm = ra[0] / r1[0], ra[1] / r1[1]
            if cs > best[0]:
                best = (cs, cm, a, nf)
            print(f'   α={a:.4f}, {nf} родин: сума {cs:.3f}× · '
                  f'медіана {cm:.3f}×')
    cs, cm, a, nf = best
    print(f'\n🔑 ВИРОК: чистий {cs:.3f}× (сума), {cm:.3f}× (медіана) '
          f'при α={a:.4f}, {nf} родин')
    print(f'   {"ЖИВИЙ" if (cs > 1.05 and cm > 1.05) else "МЕРТВИЙ"} '
          f'(🟢гейт >1.05× на обох)')
    tot = res[(a, nf)][0]
    print(f'📊 БАЛ: разом із перефітом {tot:.3f}× ⟹ '
          f'2.05e-07 → {2.05e-07 / max(tot, 1e-9):.3e}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
