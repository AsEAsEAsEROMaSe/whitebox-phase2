"""РОДИНА `G` (МАСКА) НА ГЛИБИНІ 9..22 — ЧИ КУПУЄ ВОНА ЩОСЬ?

════════ ЗВІДКИ (прайсинг `snr.py`, С75) ════════
Відношення `r = ‖зміщення замикання‖/‖шум вибірки‖` по родинах і шарах:
    перетин `r=1`:  `S`→шар 3 · `A`→шар 3 · `Q`→шар 5 · **`G`→шар 22**
🔑 `G` (`gate − Φ_замик`) — ЄДИНА родина, чия опора лишається чистою на
глибині, і вона чиста ВТРИЧІ-ВП'ЯТЕРО за решту на КОЖНОМУ шарі.
🎓 Механізм: `Φ(t)` залежить від `t=m/s`, а похибки `m` і `C` у замиканні
дивляться в ПРОТИЛЕЖНІ боки ([[finding-closure-is-a-balance-point...]]) ⟹
саме у цьому відношенні вони й гасяться. `G` сидить у точці балансу.

════════ ЧОМУ ЦЕ НЕ СУПЕРЕЧИТЬ ⛔`KSIG` 8→24 ════════
Той провал (1.032→0.973) тягнув ГЛИБШЕ ВСІ ЧОТИРИ родини, а `S`/`A`/`Q`
за `snr.py` брудні вже з шарів 3–5 ⟹ глибокі шари вливали чисте зміщення.
Тут глибше йде ЛИШЕ `G`; родини 1..8 лишаються рівно як у сабміті.

════════ ЦІНА ПОРАХОВАНА ДО ПРОГОНУ ════════
Замикання = 2 матмули 256³ = 6.71e7 флоп/шар = **0.0247% бюджету**.
`n_pairs` рахується лише під основний цикл ⟹ ці флопи йдуть ПОВЗ бюджет:
    KSIG 8→12 гейт **>1.007×** · 8→16 **>1.017×** · 8→18 **>1.022×**
    8→20 **>1.027×** · 8→22 **>1.032×**
Приз із карти глибини: K=8 1.72× · K=12 2.21× · K=16 2.89× · K=20 4.09×.

════════ 📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ ════════
📣 `G` на 1..8 у `biassplit` дала лише **1.006×** попри чисту опору ⟹ низьке
   `r` НЕОБХІДНЕ, але НЕ достатнє. Приз росте 1.72→2.89× від шару 8 до 16.
📣 очікую **1.01–1.06×** — тобто гейт 1.022× це підкидання монети, і саме
   тому вимір вартий прогону.
🟢 ЖИВЕ, якщо найкращий зріз > свого гейта на СУМІ І МЕДІАНІ.

🚨 ПЛОМБИ
  ✅ β=0 = РІВНО 1.000× · ✅база мусить відтворити MSE 2.300e-06 із `biassplit`
  ⛔ НЕГ: рядки з СУСІДНЬОГО жереба ⟹ ≈1.000×
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import cut_and_budget, closure_parts, W_                  # noqa: E402
from varsub import S72, A72, Q72, G72                                 # noqa: E402
from price31 import loo, three_slices                                 # noqa: E402
from biassplit import truth_full, run_collect                         # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:3'
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 32
KSUB = 8            # родини сабміту лишаються рівно на 1..8
KCL = 22            # глибше опора `G` вже брудна (r>1)
LAM = (S72, A72, Q72, G72)
BANDS = ((9, 12), (13, 16), (17, 19), (20, 22))
GATES = {12: 1.007, 16: 1.017, 19: 1.0245, 22: 1.032}


def main(nets=24, dev=DEV, ndraw=NDRAW):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NB = len(BANDS)
    print(f'📣 ПЕРЕДБАЧЕНО 1.01–1.06×; гейти за ціною: '
          + ' · '.join(f'до {k}: >{v:.3f}×' for k, v in GATES.items()))
    print(f'🟢 ЖИВЕ, якщо зріз б\'є СВІЙ гейт на сумі І медіані.  '
          f'{nets} мереж × {ndraw} жеребів, КРОС-ЦІЛЬ\n')

    base, G, B, C = [], [], [], []
    Gn, Bn, Cn = [], [], []
    mse_chk = 0.0
    nout = 0
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            d = len(wc)
            kcl = min(KCL, d - 1)
            cl = closure_parts(wc, kcl, dev, extras=True)
            T1 = truth_full(W, dev, 810_000 + 7919 * n)
            T2 = truth_full(W, dev, 910_000 + 7919 * n)
            nout += T1.shape[0]
            # λ сабміту: тільки шари 1..8 кожної родини, решта НУЛЬ
            lam = torch.zeros(4 * kcl, dtype=torch.float64, device=dev)
            for q in range(4):
                for i in range(min(KSUB, kcl)):
                    lam[q * kcl + i] = LAM[q][i]
            # рядки каналу: родина G (індекс 3), шари смуги, СУМОЮ
            rows = []
            for lo, hi in BANDS:
                m = torch.zeros(4 * kcl, dtype=torch.float64, device=dev)
                for i in range(lo, min(hi, kcl) + 1):
                    m[3 * kcl + i - 1] = 1.0
                rows.append(m)
            M = torch.stack(rows, 0)                    # (NB, 4*kcl)

            bm = 0.0
            gg = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bb = torch.zeros(NB, dtype=torch.float64, device=dev)
            cc = 0.0
            gN = torch.zeros(NB, NB, dtype=torch.float64, device=dev)
            bN = torch.zeros(NB, dtype=torch.float64, device=dev)
            cN, prevR = 0.0, None
            for dr in range(ndraw):
                sd = 88_000_000 + 733 * n + dr
                p0, F = run_collect(wc, npair, cl, dev, sd)
                pb = p0 - lam @ F                       # = ПОВНИЙ сабміт
                R = M @ F                               # (NB, n_L)
                e1, e2 = pb - T1, pb - T2
                e = 0.5 * (e1 + e2)
                bm += float((e1 * e2).sum())
                cc += float((e1 * e2).sum())
                gg += R @ R.T
                bb += R @ e
                if prevR is not None:
                    gN += prevR @ prevR.T
                    bN += prevR @ e
                    cN += float((e1 * e2).sum())
                prevR = R
            mse_chk += bm / ndraw
            base.append(bm)
            G.append(gg); B.append(bb); C.append(cc)
            Gn.append(gN); Bn.append(bN); Cn.append(cN)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair}, kcl={kcl})', flush=True)

    bs = np.array(base)
    idx = list(range(NB))
    mse = mse_chk / nout
    print(f'\n✅ПЛОМБА БАЗИ: MSE {mse:.3e} (biassplit дав 2.300e-06, '
          f'розбіжність {abs(mse-2.300e-6)/2.300e-6*100:.1f}%)')
    print(f'\n{nets} мереж × {ndraw} жеребів. База = ПОВНИЙ сабміт. '
          f'LOO ПО МЕРЕЖАХ.\n')
    print(f'{"зріз (родина G)":>28s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s} {"гейт":>8s} {"вирок":>7s}')
    print('-' * 78)
    print(f'{"✅β=0 (пломба)":>28s} {1.0:8.3f}× {1.0:8.3f}× {1.0:9.3f}× '
          f'{"—":>8s} {"—":>7s}')
    SUBS = [('9–12', [0], 12), ('9–16', [0, 1], 16),
            ('9–19', [0, 1, 2], 19), ('9–22 (усі 4 смуги)', idx, 22)]
    best = (0.0, None)
    for nm, ii, K in SUBS:
        val = loo([g[ii][:, ii] for g in G], [b[ii] for b in B], C, len(ii))
        sv, mdv, w3v = three_slices(bs, val)
        gt = GATES[K]
        ok = min(sv, mdv) > gt
        if min(sv, mdv) / gt > best[0]:
            best = (min(sv, mdv) / gt, nm)
        print(f'{"🎯 " + nm:>28s} {sv:8.3f}× {mdv:8.3f}× {w3v:9.3f}× '
              f'{gt:7.3f}× {"ЖИВЕ" if ok else "мертво":>7s}')
    vN = loo([g[idx][:, idx] for g in Gn], [b[idx] for b in Bn], Cn, NB)
    sN, mN, wN = three_slices(bs, vN)
    print(f'{"⛔НЕГ (сусідній жереб)":>28s} {sN:8.3f}× {mN:8.3f}× {wN:9.3f}×')

    print(f'\n🔑 НАЙКРАЩЕ ВІДНОШЕННЯ ДО СВОГО ГЕЙТА: {best[1]} ⟹ '
          f'{best[0]:.3f}× (>1 = чистий виграш)')
    print(f'🔑 ВИРОК: {"ЖИВЕ" if best[0] > 1.0 else "МЕРТВО"}')
    print(f'   📣передбачено 1.01–1.06× при гейтах 1.007–1.032×')
    if best[0] > 1.0:
        print(f'📊 БАЛ: 1.954e-07 → {1.954e-07 / best[0]:.3e} '
              f'(ЧИСТИЙ, ціну вже враховано в гейті)')
    else:
        print(f'📊 БАЛ: без змін 1.954e-07')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
