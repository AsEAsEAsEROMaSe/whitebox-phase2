"""ЧИ НЕ ОТРУЄНА СТЕЛЯ 2.00×? — РОЗЩЕПЛЕННЯ ЖЕРЕБІВ + ПОЗ.КОНТРОЛЬ ОТРУЄННЯ.

ЗВІДКИ. Юзер: стеля виросла втретє (1.79→1.87→2.00), і це рівно той підпис, який
дає оракул, що ділить жереб із ціллю. Питання поставлене однозначно: звідки
береться `m_іст` у `vrank.py`. Відповідь із коду: `truth_layers31(810_000+7919n)`,
2^19 антитетичних пар = 2^20 семплів, а жереби оцінювача — `33_000_000+733n+d`,
а ціль — `truth_last(510_000+7919n)`. ТРИ РОЗДІЛЬНІ СІМ'Ї СІДІВ. Але прочитаний
сід — не вимір, тому тут це перетворюється на числа.

🚨 І ЮЗЕР МАЄ РАЦІЮ ЩОДО НЕГ: перемішування критерію руйнує і сигнал, і отруєння
одночасно ⟹ чистий НЕГ не є доказом чистоти джерела. Потрібен ПОЗ.КОНТРОЛЬ, який
отруєння ВІДТВОРЮЄ.

ЩО МІРЯЄМО (усе одним проходом — перенос якоря лінійний)
  A. РЕПЛІКА НА РОЗДІЛЬНІЙ ІСТИНІ: `r` з потоку A (сід 810k) проти `r` з потоку
     B (сід 910k), обидва по 2^18 пар, жодного спільного семплу. Якщо приз
     тримається на обох — він у СТРУКТУРІ МЕРЕЖІ, а не в жеребі.
  B. ДРАБИНА ТОЧНОСТІ ІСТИНИ: `r` з 2^18 / 2^16 / 2^13 / 2^10 пар одного потоку.
     Якщо приз падає з погіршенням істини — `r` несе реальний сигнал; якщо
     тримається — його несе шум істини (тобто нічого).
  C. 🚨ПОЗ.КОНТРОЛЬ ОТРУЄННЯ: `r̂` з ТОГО САМОГО жеребу, на якому міряється
     виграш (`x̄_ℓ` цього жеребу проти замикання). Це рівно та схема, якої
     боїться юзер. Мусить дати ЗАВИЩЕНЕ число — інакше стенд сліпий.
  D. 🟡ДОСТАВНА ВЕРСІЯ (бонус, вона ж ідея юзера про емпіричний Байєс):
     `r̂` з ПІЛОТНОГО жеребу (інший сід, не в оцінюванні), у двох формах —
     сирій і з дебіасом `r̂² = d²/σ² − 1` (бо `E[d²] = b² + σ²`).

КОНТРОЛІ
  * ⛔НЕГ: критерій A, перемішаний у шарі (`roll`);
  * 📊 три зрізи: сума · МЕДІАНА · без топ-3; `λ_ℓ` — LOO ПО МЕРЕЖАХ у КОЖНОМУ
    варіанті;
  * 🔍 кореляція `log r_A` з `log r_B` — скільки в `r` взагалі сигналу.
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
from price31 import loo, three_slices                             # noqa: E402
from vrank import state_and_v                                     # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
KS = 30
QM = 0.3                              # квантиль маски, на якій стоїть 2.00×
MARKS = [1024, 8192, 65536, 262144]
CHUNK = 65536
BANDS = {'1..8': list(range(0, 8)), '24..30': list(range(23, 30)),
         '1..30': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def truth_ladder(wc, dev, seed, marks=MARKS):
    """`m` активацій шарів 1..KS на кількох рівнях точності з ОДНОГО потоку.

    Той самий обхід, що в `truth_layers31`, лише зі знімками на позначках.
    """
    ns = [w.shape[1] for w in wc]
    a1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    rng = np.random.default_rng(int(seed))
    out, done = {}, 0
    for mk in sorted(marks):
        while done < mk:
            k = min(CHUNK, mk - done)
            z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                             device=dev)
            x = torch.cat([z, -z], 0)
            for i, w in enumerate(wc):
                x = torch.relu(x @ w)
                a1[i] += x.double().sum(0)
            done += k
        n = 2.0 * done
        out[mk] = [a1[i] / n for i in range(1, KS + 1)]
    return out


def crit_rows(m_cl, mtr, nz, vs, i, debias=False):
    """(β, маска r<1, β·маска(crit,q), crit) для одного шару."""
    noise = nz[i + 1]                       # 🚨 `m_cl[i]` — шар `i+1` у `wc`
    d = (m_cl[i] - mtr[i]).abs()
    if debias:
        r2 = ((d / noise) ** 2 - 1.0).clamp_min(0.0)
    else:
        r2 = (d / noise) ** 2
    beta = 1.0 / (1.0 + r2)
    crit = (vs[i + 1] * noise) ** 2 * beta
    msk = (crit >= torch.quantile(crit, 1.0 - QM)).double()
    return beta, (r2 < 1.0).double(), beta * msk, crit


def main(nets=24, dev=DEV):
    torch.manual_seed(271828)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = ['рівно',
             '⚠️β  (істина A)', '⚠️β  (істина B)',
             '⚠️маска r<1 (A)', '⚠️маска r<1 (B)',
             '🟢β·маска q=.3 (A)', '🟢β·маска q=.3 (B)',
             'β·маска A@2^16', 'β·маска A@2^13', 'β·маска A@2^10',
             '🟡пілот, сирий r̂', '🟡пілот, дебіас r̂',
             '⛔НЕГ (crit A roll)']
    base_mse, G, B, C, corr = [], None, None, [], []
    Gp, Bp, Cp = [], [], []                       # 🚨поз.контроль отруєння

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            LA = truth_ladder(wc, dev, 810_000 + 7919 * n)
            LB = truth_ladder(wc, dev, 910_000 + 7919 * n, marks=[262144])
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            vs, nz = state_and_v(wc, dev, npair)

            # 🟡 пілотний жереб: сід поза оцінюваним діапазоном
            _, MP = run_side1(wc, npair, s_cl, m_cl, dev,
                              44_000_000 + 733 * n, lam_s=LAM31,
                              collect_means=True)
            mpil = [MP[i][0] for i in range(KS)]

            AW = []
            for i in range(KS):
                bA, kA, mA, cA = crit_rows(m_cl, LA[262144], nz, vs, i)
                bB, kB, mB, _ = crit_rows(m_cl, LB[262144], nz, vs, i)
                rows = [torch.ones_like(bA), bA, bB, kA, kB, mA, mB]
                for mk in (65536, 8192, 1024):
                    rows.append(crit_rows(m_cl, LA[mk], nz, vs, i)[2])
                rows.append(crit_rows(m_cl, mpil, nz, vs, i)[2])
                rows.append(crit_rows(m_cl, mpil, nz, vs, i, debias=True)[2])
                rows.append(torch.roll(mA, 7))
                AW.append(torch.stack(rows))
                la, lb = bA.clamp_min(1e-300).log(), bB.clamp_min(1e-300).log()
                corr.append(float(torch.corrcoef(torch.stack([la, lb]))[0, 1]))
            nv = AW[0].shape[0]
            if G is None:
                G = [[] for _ in range(nv)]
                B = [[] for _ in range(nv)]

            gg = torch.zeros(nv, KS, KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, KS, dtype=torch.float64, device=dev)
            gp = torch.zeros(2, KS, KS, dtype=torch.float64, device=dev)
            bp = torch.zeros(2, KS, dtype=torch.float64, device=dev)
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
                # 🚨 ОТРУЄНИЙ варіант: `r̂` з ЦЬОГО САМОГО жеребу
                _, MD = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM31, collect_means=True)
                msd = [MD[i][0] for i in range(KS)]
                AP = [torch.stack([torch.ones_like(nz[i + 1]),
                                   crit_rows(m_cl, msd, nz, vs, i)[2]])
                      for i in range(KS)]
                _, Ap = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM31, collect_anchors=True,
                                  anchor_w=AP)
                Apm = Ap.view(2, KS, -1)
                gp += Apm @ Apm.transpose(1, 2)
                bp += Apm @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            Gp.append(gp); Bp.append(bp); Cp.append(cc)
            C.append(cc); base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів, `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ '
          f'варіанті')
    print(f'🔍 кореляція log β(A) з log β(B) — дві РОЗДІЛЬНІ істини: '
          f'{np.nanmean(corr):+.3f}   (1 ⟹ `r` це структура; 0 ⟹ шум істини)\n')
    for bn, idx in BANDS.items():
        print(f'--- шари {bn} ---')
        print(f'{"варіант":>22s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = '  ← У САБМІТІ' if nm == 'рівно' else ''
            tag = '  ⛔' if nm.startswith('⛔') else tag
            print(f'{nm:>22s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        for v, nm in enumerate(['рівно (звірка)', '🚨ОТРУЄНИЙ: r̂ з ТОГО САМОГО жеребу']):
            val = loo([g[v][idx][:, idx] for g in Gp],
                      [b[v][idx] for b in Bp], Cp, len(idx))
            s, md, w3 = three_slices(base, val)
            print(f'{nm:>22s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
        print()
    print('🔑 читати так: якщо A і B дають те саме, а драбина точності ПАДАЄ — '
          'приз реальний. Якщо 🚨ОТРУЄНИЙ помітно вищий за 🟢 — стенд ловить '
          'отруєння, і його відсутність у 🟢 щось означає. Якщо 🚨 не вищий — '
          'стенд сліпий і вирок давати не можна.')


if __name__ == '__main__':
    if len(sys.argv) > 2:
        NDRAW = int(sys.argv[2])
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
