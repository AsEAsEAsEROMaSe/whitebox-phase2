"""ЧИ Є ЩОСЬ ПОВЕРХ **СПРАВЖНЬОГО** САБМІТУ (`sub72`, ЧОТИРИ КАНАЛИ).

🛑 ЗВІДКИ. Усі стенди сесії міряли проти `base_real.py` = порт
`estimator_side1.py`, де `_LAM_A = _LAM_Q = _LAM_G = 0` (сам σ-канал).
А діючий сабміт `estimator_sub72.py` має ЧОТИРИ родини:
    δs (`_LAM_S`) · δx̄ (`_LAM_A`) · δE[x²] (`_LAM_Q`) · δмаски (`_LAM_G`)
⟹ «знайдений» канал `â²` = `_LAM_Q`, «знайдений» `q̂` = `_LAM_G`, обидва ВЖЕ ТАМ.

ЩО ТУТ. Порт ПОВНОГО sub72 (`base72.py`) і питання: чи лишилось щось ПОВЕРХ.
Єдиний кандидат — `p⁴` (4-й момент ДОАКТИВАЦІЙ), якого в сабміті НЕМА;
ціль `3s⁴` (точна на шарі 1, гаусове наближення глибше). Коштує `O(N·n)`.

ПЕЧАТКИ (обидві обов'язкові — саме їх бракувало сьогодні):
  * ✅**ПОРТ**: `sub72` мусить бути КРАЩИЙ за `side1` приблизно на ті 5%, що я
    хибно приписав «новому каналу». Якщо ні — порт невірний;
  * ✅`λ_p⁴ = 0` мусить дати РІВНО 1.000× проти sub72;
  * ⛔ШУМ тієї самої норми ≈1.000×.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, LAM_S)       # noqa: E402
from base72 import run_side1 as run72                              # noqa: E402
from tilt_wide import truth_last                                   # noqa: E402
from price31 import loo, three_slices                              # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 64
K = 8

# ── коефіцієнти ДІЮЧОГО САБМІТУ `estimator_sub72.py` ──────────────────
S72 = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
       0.0088285, 0.0007922, 0.0015044, 0.0001062)
A72 = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
       -0.0025822, -0.0030778, 0.0006822, 0.0068061)
Q72 = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
       0.0189783, 0.0250559, 0.0340802, 0.0415433)
G72 = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
       -0.0146129, -0.0100612, 0.0167223, 0.0168109)


def main(nets=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    mse_side1, mse_sub72, C = [], [], []
    G, B = [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, K, dev, extras=True)
            p4_ref = [3.0 * s ** 4 for s in s_cl]
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            m1 = m2 = 0.0
            gg = torch.zeros(K, K, dtype=torch.float64, device=dev)
            bb = torch.zeros(K, dtype=torch.float64, device=dev)
            cc = 0.0
            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                # база A: `side1` (те, проти чого міряли ВСІ стенди)
                pa, _ = run72(wc, npair, s_cl, m_cl, dev, sd, lam_s=LAM_S)
                m1 += float(((pa - T1) * (pa - T2)).sum())
                # база Б: ПОВНИЙ `sub72` — ЧОТИРИ канали
                pb, _ = run72(wc, npair, s_cl, m_cl, dev, sd, lam_s=S72,
                              lam_a=A72, lam_q=Q72, lam_g=G72,
                              q_ext=q_cl, p_ext=f_cl)
                m2 += float(((pb - T1) * (pb - T2)).sum())
                # кандидат ПОВЕРХ sub72: `p⁴` з ціллю `3s⁴`
                _, AM = run72(wc, npair, s_cl, m_cl, dev, sd, lam_s=None,
                              collect_anchors=True,
                              anchor_multi=(['p4'], [p4_ref]))
                e1, e2 = pb - T1, pb - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                gg += AM @ AM.T
                bb += AM @ e
            mse_side1.append(m1); mse_sub72.append(m2)
            G.append(gg); B.append(bb); C.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    a = np.array(mse_side1); b = np.array(mse_sub72)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ.\n')
    print(f'✅ПОРТ: `sub72` проти `side1` = {a.sum()/b.sum():.4f}× '
          f'(медіана {np.median(a/b):.4f}×)')
    print('   🎯 мусить бути ≈1.05× — рівно те, що я хибно приписав '
          '«новому каналу»\n')
    idx = list(range(K))
    val = loo([g[idx][:, idx] for g in G], [v[idx] for v in B], C, K)
    s, md, w3 = three_slices(b, val)
    print(f'{"варіант":>34s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
    print('-' * 66)
    print(f'{"✅λ=0 (пломба)":>34s} {1.0:8.3f}× {1.0:8.3f}× {1.0:9.3f}×')
    print(f'{"🎯p⁴ ПОВЕРХ ПОВНОГО sub72":>34s} {s:8.3f}× {md:8.3f}× '
          f'{w3:9.3f}×')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
