"""РЕАЛІСТИЧНИЙ ЕДЖВОРТ: ЧИ ПЛАТИТЬ КАНАЛ БЕЗ ЖОДНОГО ОРАКУЛА? (С81)

## ЛАНЦЮГ, ЯКИЙ ПРИВІВ СЮДИ
`qladder.py` → ТЗ: `q` треба точний до 0.222% (⛔рівень контролю ×1.0165).
`qsplit.py`  → 87% похибки — ФОРМА... але то було з ІСТИННИМИ `m,s`.
`qedge.py`   → Еджворт з істинними `m,s,γ₁`: 0.385% ⟹ **×1.1296**.
`qgam.py`    → 🚨 з аргументом `t` ІЗ ЗАМИКАННЯ навіть ІСТИННИЙ `γ₁` дає лише
               **0.7319%**, а `γ₁` з 4096 семплів — 0.7536% (пілота замало:
               61% похибки `γ₁` робить `q` ГІРШИМ).
🎓 Дві похибки — форми й аргумента — ЧАСТКОВО ГАСИЛИ ОДНА ОДНУ. Виправив
   форму — оголив аргумент.

## ЩО МІРЯЄМО ТУТ (остаточно, без інференції)
Канал воріт із РЕАЛІСТИЧНИМ еталоном, тобто таким, який можна побудувати
в сабміті: `t` із замикання + `γ₁` з НЕЗАЛЕЖНИХ 4096 семплів.
  база · 🎯ОРАКУЛ `q_іст` · 🟢РЕАЛІСТИЧНИЙ · 🔵напів-оракул (істинний `γ₁`)
  · 🟢вільний `Φ` · ⛔НЕГ

⛔ ПЛОМБИ: оракул 1.3236×, вільний 0.9968×, НЕГ 1.0240× (усі з `qladder.py`).

📣 ПЕРЕДБАЧЕНО: реалістичний дасть ×1.00…1.04, тобто НЕ ВИЩЕ за контроль
   ×1.0165 ⟹ ВИРОК «гілка закрита». Причина: 0.75% лежить майже там само,
   де нинішні 0.887%. Але `qedge.py` уже раз показав, що ФОРМА залишку важить
   більше за розмір, тому інференцію не приймаю — міряю.
   🚨 І навіть виграш не закриє питання ЦІНИ: незалежні 4096 семплів — це
   ~половина головної вибірки (≈4200 пар), тобто прямий податок на дисперсію.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qedge2.py [NETS]
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
NPAIR, CH, I2PI = 524_288, 32768, 0.3989422804014327


def _phi(x):                       # 🔁 інлайн із `qsplit.py` (щоб не тягти
    return torch.exp(-0.5 * x * x) * I2PI          # його верхній рівень)


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def rel(pred, tru):
    num = sum(float(((pred[i] - tru[i]) ** 2).sum()) for i in range(31))
    den = sum(float((tru[i] ** 2).sum()) for i in range(31))
    return np.sqrt(num / max(den, 1e-300))


def moments(wc, dev, seed, npair=NPAIR):
    """Істинні `q`, `m`, `s`, `γ₁` перед-активацій шарів 1..31."""
    ns = [w.shape[1] for w in wc]
    acc = [[torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
           for _ in range(4)]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.tensor(rng.standard_normal((k, 256)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = (x @ w).double()
            acc[0][i] += (p > 0).double().sum(0)
            p2 = p * p
            acc[1][i] += p.sum(0)
            acc[2][i] += p2.sum(0)
            acc[3][i] += (p2 * p).sum(0)
            x = torch.relu(p).float()
        done += k
    n = 2.0 * done
    out = []
    for i in range(1, 32):
        m1, m2, m3 = acc[1][i] / n, acc[2][i] / n, acc[3][i] / n
        s = (m2 - m1 * m1).clamp_min(1e-300).sqrt()
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        out.append((acc[0][i] / n, m1, s, c3 / s ** 3))
    return out

L_, W_, KS = 32, 256, 31
DEV = 'cuda:3'
NDRAW = 64


def main(nets=24, dev=DEV):
    torch.manual_seed(108)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

    VAR = ['база (σ-канал + μ-якір)', '🎯 ОРАКУЛ `q_іст`',
           '🟢 РЕАЛІСТИЧНИЙ (t замик. + γ₁ з 4096)',
           '🔵 напів-оракул (t замик. + γ₁ іст.)',
           '🟢 ВІЛЬНИЙ `Φ` із замикання',
           '⛔НЕГ (еталон зі зсунутого жеребу)']
    NV = len(VAR)
    G = [[] for _ in VAR]
    B = [[] for _ in VAR]
    C, base_mse, RE = [], [], [[], [], [], []]

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, f_cl, _ = closure_parts(wc, KS, dev, extras=True)
            M = moments(wc, dev, 830_000 + 7919 * n)
            q_tru = [M[i][0] for i in range(KS)]
            g_t = [M[i][3] for i in range(KS)]
            M4 = moments(wc, dev, 55_000_000 + 7919 * n, 2048)   # 4096 семплів
            g_4 = [M4[i][3] for i in range(KS)]
            # 🔑 `t` із ЗАМИКАННЯ, дістане точним оберненням `f_cl = Φ(t)`
            t = [1.4142135623730951 * torch.erfinv(
                2.0 * f_cl[i].clamp(1e-12, 1.0 - 1e-12) - 1.0)
                for i in range(KS)]
            cor = [_phi(t[i]) * (t[i] ** 2 - 1.0) / 6.0 for i in range(KS)]
            e_r = [_Phi(t[i]) + cor[i] * g_4[i] for i in range(KS)]
            e_o = [_Phi(t[i]) + cor[i] * g_t[i] for i in range(KS)]
            for v, P in enumerate((q_tru, e_r, e_o, f_cl)):
                RE[v].append(rel(P, q_tru))
            T1 = truth_last(wc, dev, 510_000 + 7919 * n)
            T2 = truth_last(wc, dev, 610_000 + 7919 * n)

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
                for v, qk in enumerate((q_tru, e_r, e_o, f_cl)):
                    _, Q = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                     lam_s=None, collect_anchors=True,
                                     anchor_multi=(['ind'], [qk]))
                    R = torch.cat([RB, Q], 0)
                    gg[1 + v] += R @ R.T
                    bb[1 + v] += R @ e
                    if v == 3:
                        lastQ = Q
                if prevQ is not None:
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
    print(f'{"варіант":36s} {"похибка q":>10s} {"сума":>9s} {"МЕДІАНА":>9s} '
          f'{"без топ-3":>10s} {"понад базу":>11s}')
    print('-' * 92)
    out = []
    for v, nm in enumerate(VAR):
        val = loo(G[v], B[v], C, G[v][0].shape[0])
        s, md, w3 = three_slices(base, val)
        out.append(s)
        re = (f'{100 * float(np.mean(RE[v - 1])):9.3f}%'
              if 1 <= v <= 4 else ' ' * 10)
        gain = f'×{s / out[0]:.4f}' if v else '     —'
        print(f'{nm:36s} {re} {s:8.4f}× {md:8.4f}× {w3:9.4f}× {gain:>11s}')
    ed = out[2] / out[0]
    print(f'\n🎯 РЕАЛІСТИЧНИЙ проти рівня ⛔контролю ×{out[5]/out[0]:.4f}: '
          f'{"🟢 ВИЩЕ ⟹ гілка жива" if ed > out[5]/out[0] + 0.01 else "⛔ НЕ ВИЩЕ ⟹ гілка мертва"}')
    print(f'📊 БАЛ 1.954e-07 ⟹ з реалістичним Еджвортом {1.954e-07/ed:.4e}')
    print(f'🚨 ЦІНА НЕ ВРАХОВАНА: 4096 незалежних семплів ≈ половина '
          f'головної вибірки')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
