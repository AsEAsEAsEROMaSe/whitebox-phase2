"""ДРАБИНА ПО ТОЧНОСТІ `q`: НАСКІЛЬКИ ТОЧНИМ МУСИТЬ БУТИ ЕТАЛОН ВОРІТ? (С81)

## ЗВІДКИ ПИТАННЯ
`side66/qchan.py` (С80, 24 мережі × 64 жереби, крос-ціль, LOO) дав пару, яка
з того часу висить нерозв'язаною:
    база                       1.0073×
    🟢 ВІЛЬНИЙ (еталон = `Φ(t)` із замикання)   **0.9968×**  ← ШКОДИТЬ
    🎯 ОРАКУЛ (еталон = істинне `q`)            **1.3236×**  ← найбільший
                                                               живий важіль
Тобто канал воріт коштує НУЛЬ флопів (обидві половини вже рахуються), але
працює лише з ТОЧНИМ еталоном. Питання, яке ніхто не міряв: **де межа?**
Скільки точності треба еталону, щоб канал перестав шкодити й почав платити?

## ЩО МІРЯЄМО
Еталон беремо з драбини між істиною й замиканням:
    `q_k = q_іст + k·(Φ_замик − q_іст)`,  k = 0 · 0.125 · 0.25 · 0.5 · 0.75 · 1
k=0 — оракул, k=1 — нинішній вільний. Для кожного щабля друкуємо ТАКОЖ
відносну похибку еталона `‖q_k − q_іст‖ / ‖q_іст‖`, щоб відповідь була у
вигляді «канал платить, коли еталон точний краще ніж X%».

⛔ ПЛОМБИ:
 * k=1 мусить відтворити 0.9968×, k=0 — 1.3236× (той самий стенд, ті самі
   мережі й жереби, що в `qchan.py`);
 * ⛔НЕГ: еталон із ЗСУНУТОГО жеребу мусить сісти в базу.

📣 ПЕРЕДБАЧЕНО: залежність РІЗКО НЕЛІНІЙНА — більша частина виграшу зникне
   вже при k=0.25, тобто канал вимагатиме еталона з похибкою в одиниці
   відсотків. Причина: канал — це контрольна змінна, а КЗ із НЕТОЧНО відомим
   середнім вносить ЗМІЩЕННЯ, яке росте ЛІНІЙНО з похибкою еталона, тоді як
   виграш росте лише як КОРІНЬ із кореляції.
   🔑 Якщо ж крива пласка до k≈0.5 — задача стає реальною: half-way еталон
   можна зробити з пілота, який уже порахований і викидається.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qladder.py [NETS]
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
from price31 import loo, three_slices                             # noqa: E402
from qchan import truth_ind                                       # noqa: E402

L_, W_ = 32, 256
DEV = 'cuda:3'
NDRAW = 64
KS = 31
KGRID = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

    VAR = ['база (σ-канал + μ-якір)']
    VAR += [f'k={k:<5g} ({"ОРАКУЛ" if k == 0 else ("ВІЛЬНИЙ" if k == 1 else "суміш")})'
            for k in KGRID]
    VAR += ['⛔НЕГ (еталон зі зсунутого жеребу)']
    NV = len(VAR)
    G = [[] for _ in VAR]
    B = [[] for _ in VAR]
    C, base_mse, RELERR = [], [], [[] for _ in KGRID]

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, f_cl, _ = closure_parts(wc, KS, dev, extras=True)
            q_tru = truth_ind(wc, dev, 830_000 + 7919 * n)
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

            # 🪜 драбина еталонів + їхня власна відносна похибка
            QREF = []
            for ki, k in enumerate(KGRID):
                qk = [q_tru[i] + k * (f_cl[i] - q_tru[i]) for i in range(KS)]
                num = sum(float(((qk[i] - q_tru[i]) ** 2).sum())
                          for i in range(KS))
                den = sum(float((q_tru[i] ** 2).sum()) for i in range(KS))
                RELERR[ki].append(np.sqrt(num / max(den, 1e-300)))
                QREF.append(qk)

            nreg = [2 * KS] + [3 * KS] * (NV - 1)
            gg = [torch.zeros(k, k, dtype=torch.float64, device=dev)
                  for k in nreg]
            bb = [torch.zeros(k, dtype=torch.float64, device=dev) for k in nreg]
            cc, bm, prevQ = 0.0, 0.0, None

            for d in range(NDRAW):
                sd = 77_000_000 + 733 * n + d
                prb, _ = run_side1(wc, npair, s_cl[:8], m_cl[:8], dev, sd,
                                   lam_s=LAM_S)
                bm += float(((prb - T1) * (prb - T2)).sum())
                p0, S_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     collect_sigma=True)
                _, A_cl = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                    lam_s=None, collect_anchors=True)
                e1, e2 = p0 - T1, p0 - T2
                e = 0.5 * (e1 + e2)
                cc += float((e1 * e2).sum())
                RB = torch.cat([S_cl, A_cl], 0)
                gg[0] += RB @ RB.T
                bb[0] += RB @ e
                lastQ = None
                for ki, qk in enumerate(QREF):
                    _, Q = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     lam_s=None, collect_anchors=True,
                                     anchor_multi=(['ind'], [qk]))
                    R = torch.cat([RB, Q], 0)
                    gg[1 + ki] += R @ R.T
                    bb[1 + ki] += R @ e
                    if KGRID[ki] == 1.0:
                        lastQ = Q
                if prevQ is not None:                   # ⛔НЕГ: зсунутий жереб
                    RN = torch.cat([RB, prevQ], 0)
                    gg[NV - 1] += RN @ RN.T
                    bb[NV - 1] += RN @ e
                prevQ = lastQ
            for v in range(NV):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc); base_mse.append(bm)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів. КРОС-ЦІЛЬ, LOO ПО МЕРЕЖАХ.\n')
    print(f'{"варіант":34s} {"похибка q":>10s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s} {"понад базу":>11s}')
    print('-' * 90)
    out = []
    for v, nm in enumerate(VAR):
        val = loo(G[v], B[v], C, G[v][0].shape[0])
        s, md, w3 = three_slices(base, val)
        out.append(s)
        re = (f'{100 * float(np.mean(RELERR[v - 1])):9.3f}%'
              if 1 <= v <= len(KGRID) else ' ' * 10)
        gain = f'×{s / out[0]:.4f}' if v else '     —'
        print(f'{nm:34s} {re} {s:8.4f}× {md:8.4f}× {w3:9.4f}× {gain:>11s}')
    print('\n🔑 ЧИТАННЯ: шукаємо найбільше `k` (найгрубіший еталон), при якому '
          'колонка «понад базу» ще >1.00')
    best = max(range(len(KGRID)), key=lambda i: out[1 + i])
    print(f'📐 найкращий щабель: k={KGRID[best]:g}, похибка еталона '
          f'{100 * float(np.mean(RELERR[best])):.3f}%, приріст '
          f'×{out[1 + best] / out[0]:.4f}')
    print(f'📊 БАЛ 1.954e-07 ⟹ при цьому прирості '
          f'{1.954e-07 / (out[1 + best] / out[0]):.4e}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
