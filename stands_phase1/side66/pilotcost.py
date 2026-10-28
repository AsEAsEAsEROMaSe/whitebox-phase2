"""СКІЛЬКИ БЮДЖЕТУ КОШТУЄ ПІЛОТНЕ `r̂`? — економіка знахідки `vsplit.py`.

ЗВІДКИ. `vsplit.py` показав несподіване: `r̂`, оцінене з ОДНОГО НЕЗАЛЕЖНОГО
жеребу (`b̂ = x̄_пілот − m_замик`, жодної істини), дає **1.362× на шарах 1..30**
проти 1.058× рівної ваги при оракульній стелі 1.992× ⟹ **конверсія 32.5%**,
у ШІСТЬ разів більша за 5.6% від класифікатора на 12 ознаках. Тобто вибірка
знає про своє зміщення набагато більше, ніж будь-яка обчислювана ознака.

🚨 АЛЕ ТАМ ПІЛОТ КОШТУВАВ ЦІЛИЙ ДОДАТКОВИЙ ЖЕРЕБ — тобто +100% бюджету, а
бюджет ПО-МЕРЕЖНИЙ і ми стоїмо на підлозі 10%. Питання тому не «чи працює», а
«скільки треба». Тут — драбина розміру пілота: 1/1, 1/2, 1/4, 1/8, 1/16 від
робочого `npair`.

🔑 ДЕБІАС МУСИТЬ ЗНАТИ РОЗМІР ПІЛОТА. `E[b̂²] = b² + σ_пілот²`, тож
    `r̂² = b̂²/σ_прод² − npair/npair_пілот`  (клампінг у нуль),
бо `σ_пілот²/σ_прод² = npair/npair_пілот`. У `vsplit` пілот був повного розміру,
там відніматься мала одиниця — тут це вже не так.

КОНТРОЛІ
  * ⚠️оракул `β·маска q=.3` — стеля на ТИХ САМИХ жеребах;
  * ⛔НЕГ: пілотне `β`, перемішане всередині шару;
  * 📊 три зрізи; `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ варіанті.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, W_)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402
from vrank import state_and_v                                     # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
KS = 30
FRAC = [1, 2, 4, 8, 16]
BANDS = {'1..8': list(range(0, 8)), '24..30': list(range(23, 30)),
         '1..30': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def main(nets=24, dev=DEV):
    torch.manual_seed(1729)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = (['рівно']
             + [f'🟡пілот 1/{k} бюджету, сирий' for k in FRAC]
             + [f'🟡пілот 1/{k}, дебіас' for k in FRAC]
             + [f'🟡пілот 1/{k}, дебіас + v-маска q=.3' for k in FRAC]
             + ['⚠️оракул β', '⚠️оракул β·маска q=.3', '⛔НЕГ (пілот 1/1 roll)'])
    base_mse, G, B, C = [], None, None, []

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

            g = torch.Generator(device=dev).manual_seed(44_000_000 + 733 * n)
            zp = torch.randn(npair, W_, device=dev, generator=g)
            PIL = {}
            for k in FRAC:
                sub = zp[:max(64, npair // k)]
                _, MP = run_side1(wc, int(sub.shape[0]), s_cl, m_cl, dev, 0,
                                  lam_s=LAM31, collect_means=True, z_ext=sub)
                PIL[k] = [MP[i][0] for i in range(KS)]

            AW = []
            for i in range(KS):
                noise = nz[i + 1]           # 🚨 `m_cl[i]` — шар `i+1` у `wc`
                ro = ((m_cl[i] - TL['m'][i]).abs() / noise).clamp_min(1e-12)
                bo = 1.0 / (1.0 + ro * ro)
                co = (vs[i + 1] * noise) ** 2 * bo
                rows = [torch.ones_like(noise)]
                bd = {}
                for k in FRAC:
                    q2 = ((m_cl[i] - PIL[k][i]) / noise) ** 2
                    rows.append(1.0 / (1.0 + q2))
                for k in FRAC:
                    q2 = ((m_cl[i] - PIL[k][i]) / noise) ** 2
                    # 🔑 дебіас знає розмір пілота: віднімаємо npair/npair_пілот
                    rr = (q2 - float(npair) / max(64, npair // k)).clamp_min(0.0)
                    bd[k] = 1.0 / (1.0 + rr)
                    rows.append(bd[k])
                for k in FRAC:
                    cr = (vs[i + 1] * noise) ** 2 * bd[k]
                    rows.append(bd[k] * (cr >= torch.quantile(cr, 0.7)).double())
                rows.append(bo)
                rows.append(bo * (co >= torch.quantile(co, 0.7)).double())
                rows.append(torch.roll(bd[1], 7))
                AW.append(torch.stack(rows))
            nv = AW[0].shape[0]
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
                Am = A.view(nv, KS, -1)
                gg += Am @ Am.transpose(1, 2)
                bb += Am @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc); base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів, `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ '
          f'варіанті\n')
    for bn, idx in BANDS.items():
        print(f'--- шари {bn} ---')
        print(f'{"варіант":>34s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = '  ← У САБМІТІ' if nm == 'рівно' else ''
            tag = '  ⛔' if nm.startswith('⛔') else tag
            print(f'{nm:>34s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    print('🔑 читати так: рядок «1/8» коштує +12.5% бюджету, тобто множник ціни '
          '1.125 при підлозі 10%. Виграш мусить бити цей множник, інакше метод '
          'платить більше, ніж дає. Порівнювати з ⚠️оракулом У ЦІЙ ЖЕ таблиці.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
