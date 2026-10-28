"""ЧИ КОНВЕРТУЄТЬСЯ ВИВЕДЕНЕ `b̂`? — виграш, а не AUC (урок `maskgain.py`).

ЗВІДКИ. `rpredict2.py` вивів ПІДПИСАНЕ зміщення замикання з члена Еджворта
(`b_i = κ₃(h_i)·m_i·φ(t_i)/(6s_i³)`, ✅пломба 1.24e-14) і рекурсію юзера
`b̂_ℓ = Φ(t)·(Wᵀb̂_{ℓ−1}) + джерело`. Підписана кореляція з істинним зміщенням
**+0.246** наскрізь, і — головне — ВОНА НЕ ПАДАЄ З ГЛИБИНОЮ (0.305/0.256/0.259),
на відміну від локальних ознак (R² 0.31→0.08→0.04). ⛔НЕГ −0.004.
🚨 Але `maskgain.py` уже показав, що AUC — НЕ проксі виграшу. Тому тут одразу
виграш, і в ДВОХ різних ролях `b̂`:

  A. **ВІДНІМАННЯ** (пряме, бо `b̂` ПІДПИСАНЕ — цього не вміла жодна маска):
     `якір = x̄_ℓ − (m_замик − γ·b̂)`. Ланцюг переносу ЛІНІЙНИЙ за `m_замик`, тож
     двох прогонів (`m_замик` і `m_замик − b̂`) досить на ВЕСЬ свіп `γ`:
     `A_γ = A₀ + γ·(A₁ − A₀)`.
  B. **УСАДКА**: `r̂ = c·|b̂|/шум`, `β̂ = 1/(1+r̂²)` — доставна заміна оракульній
     `β`, з одним глобальним `c` (параметрів МЕНШЕ, ніж зараз).
  C. те саме на v-відібраній підмножині (`v²σ²β̂`) — доставний аналог 2.00×.

КОНТРОЛІ
  * ✅`γ=0` мусить збігтися з `рівно` ПОБІТОВО (пломба лінійності);
  * ⚠️оракульні рядки (`β` з істинним `r`, маска q=.3) — стеля на ЦИХ САМИХ
    жеребах, щоб конверсія рахувалась чесно;
  * ⛔НЕГ: `b̂`, ПЕРЕМІШАНЕ всередині шару (зсув) — для ОБОХ ролей;
  * 📊 три зрізи; `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ варіанті.
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
from vrank import state_and_v                                     # noqa: E402
from rpredict2 import features2                                   # noqa: E402

L_ = 32
DEV = 'cuda:1'
NDRAW = 96
KS = 30
CGRID = [0.3, 1.0, 3.0]
GGRID = [0.0, 0.3, 1.0, 3.0, -1.0]
BANDS = {'1..8': list(range(0, 8)), '24..30': list(range(23, 30)),
         '1..30': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def main(nets=24, dev=DEV):
    torch.manual_seed(2718)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    wnames = (['рівно']
              + [f'🟡β̂ вивед. c={c}' for c in CGRID]
              + ['🟡маска r̂<1 (вивед.)']
              + [f'🟡β̂·маска v²σ²β̂ q=.3 c={c}' for c in CGRID]
              + ['⚠️β оракул', '⚠️β·маска q=.3 оракул', '⛔НЕГ (b̂ перемішане)'])
    gnames = [f'🟢віднімання γ={g}' for g in GGRID] + ['⛔НЕГ віднімання']
    names = wnames + gnames
    base_mse, G, B, C, seal = [], None, None, [], []

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            vs, nz = state_and_v(wc, dev, npair)
            FF = features2(wc, dev, npair)

            bh = [FF[i][2] for i in range(KS)]              # рекурсивне b̂
            bn = [torch.roll(b, 7) for b in bh]             # ⛔НЕГ у шарі
            m_sub = [m_cl[i] - bh[i] for i in range(KS)]    # для віднімання
            m_sbn = [m_cl[i] - bn[i] for i in range(KS)]
            AW = []
            for i in range(KS):
                noise = nz[i + 1]                # 🚨 `m_cl[i]` — шар `i+1` у `wc`
                ro = ((m_cl[i] - TL['m'][i]).abs() / noise).clamp_min(1e-12)
                bo = 1.0 / (1.0 + ro * ro)
                co = (vs[i + 1] * noise) ** 2 * bo
                rows = [torch.ones_like(noise)]
                for c in CGRID:
                    rows.append(1.0 / (1.0 + (c * bh[i] / noise) ** 2))
                rows.append(((bh[i].abs() / noise) < 1.0).double())
                for c in CGRID:
                    bb_ = 1.0 / (1.0 + (c * bh[i] / noise) ** 2)
                    cr = (vs[i + 1] * noise) ** 2 * bb_
                    rows.append(bb_ * (cr >= torch.quantile(cr, 0.7)).double())
                rows.append(bo)
                rows.append(bo * (co >= torch.quantile(co, 0.7)).double())
                rows.append(1.0 / (1.0 + (bn[i] / noise) ** 2))
                AW.append(torch.stack(rows))
            nw = AW[0].shape[0]
            nv = nw + len(gnames)
            if G is None:
                G = [[] for _ in range(nv)]
                B = [[] for _ in range(nv)]

            gg = torch.zeros(nv, KS, KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, KS, dtype=torch.float64, device=dev)
            cc = 0.0
            for d in range(NDRAW):
                sd = 33_000_000 + 733 * n + d
                pr, A = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM31, collect_anchors=True,
                                  anchor_w=AW)
                e = pr - T
                cc += float((e * e).sum())
                Am = A.view(nw, KS, -1)
                # 🔑 перенос ЛІНІЙНИЙ за `m_замик` ⟹ один додатковий прогін дає
                #    ВЕСЬ свіп γ: `A_γ = A₀ + γ(A₁ − A₀)`
                _, A1 = run_side1(wc, npair, s_cl, m_sub, dev, sd,
                                  lam_s=LAM31, collect_anchors=True)
                _, A1n = run_side1(wc, npair, s_cl, m_sbn, dev, sd,
                                   lam_s=LAM31, collect_anchors=True)
                Tb, Tbn = A1 - Am[0], A1n - Am[0]
                extra = [Am[0] + g * Tb for g in GGRID] + [Am[0] + Tbn]
                if d == 0:
                    seal.append(float((extra[0] - Am[0]).abs().max()))
                Af = torch.cat([Am, torch.stack(extra)], 0)
                gg += Af @ Af.transpose(1, 2)
                bb += Af @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc); base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    base = np.array(base_mse)
    print(f'\n✅ПЛОМБА лінійності (γ=0 проти «рівно»): {max(seal):.2e}')
    print(f'\n{nets} мереж × {NDRAW} жеребів, `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ '
          f'варіанті\n')
    for bnm, idx in BANDS.items():
        print(f'--- шари {bnm} ---')
        print(f'{"варіант":>28s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = '  ← У САБМІТІ' if nm == 'рівно' else ''
            tag = '  ⛔' if nm.startswith('⛔') else tag
            print(f'{nm:>28s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    print('🔑 читати так: 🟡 і 🟢 — ДОСТАВНІ (нуль істини). Порівнювати їх треба '
          'з ⚠️оракулами В ЦІЙ ЖЕ таблиці: конверсія = (доставне−1)/(оракул−1). '
          'Якщо віднімання б\'є усадку — `b̂` цінне ЗНАКОМ, а не лише модулем, '
          'і тоді вся лінія «класифікації» була не тим питанням.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
