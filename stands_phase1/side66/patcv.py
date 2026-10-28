"""КОНТРОЛЬНІ ЗМІННІ НА ПАТЕРНІ, А НЕ НА АКТИВАЦІЯХ — ідея юзера, прямий наслідок
нашого ж виміру «дисперсія живе в тому, В ЯКУ комірку ти потрапив».

ЗВІДКИ. Якщо патерн і є відповідь, то КЗ має бути функцією ПАТЕРНУ. Найдешевша
така функція — **частка активних** `n_ℓ = (1/N)Σ1[h_i>0]`; вона на прямому
проході, коштує нуль, а її замикальне середнє = `(1/N)ΣΦ(m_i/s_i)`. Ключова
відмінність від `E[a_ℓ]`: це СКАЛЯР від відношення `m/s`, і `Φ` НАСИЧУЄТЬСЯ, тож
похибка замикання на ньому — зовсім інша величина. Друга сім'я — `E[a_ℓ²]`, чиє
замикальне середнє теж у замкнутій формі: `(m²+s²)Φ(t)+m·s·φ(t)`.
🔑 Це прямо б'є в мій вирок про ACV («усі сурогати впираються в те саме невідоме
середнє»): для `E[a_ℓ]` це правда, для часток і норм — треба МІРЯТИ.

ГОЛОВНЕ ЧИСЛО — не виграш, а **`r = |замикання − істина| / std(статистики)`**:
відношення зміщення до шуму, з якого виводиться усадка `1/(1+r²)`. Якщо для
часток `r` помітно менший, ніж для `E[a_ℓ]`, гілка жива, і тоді вже цікавий
виграш. Якщо той самий — ACV нема з чого будувати.

КОНТРОЛІ
  * ⚠️ORACLE-стеля: `β` фітується ПО МЕРЕЖІ (фіт/оцінка по половинах жеребів) —
    це ВЕРХНЯ МЕЖА, не метод: у сабміті померéжевого фіту нема;
  * ⛔НЕГ: статистика зі ЗСУНУТОГО жеребу (`roll`) — мусить дати ≈1.0;
  * ✅для порівняння в тій самій таблиці — `E[a_ℓ]`, тобто вже наявний μ-якір.
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

L_ = 32
DEV = 'cuda:3'
NDRAW = 128
K31 = 31
PROBE = [1, 4, 8, 16, 24, 30]
STATS = ('E[a]', 'частка', 'E[a²]')
RIDGE = 1e-2
LAM31 = tuple(LAM_S) + (0.0,) * (K31 - len(LAM_S))


def truth_stats(wc, dev, seed, npair=262_144, chunk=32768):
    """Істинні `E[a_ℓ]`, частка активних, `E[a_ℓ²]` для шарів 1..31."""
    ns = [w.shape[1] for w in wc]
    a1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    fr = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    q1 = [torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = x @ w
            fr[i] += (p > 0).double().sum(0)
            x = torch.relu(p)
            xd = x.double()
            a1[i] += xd.sum(0)
            q1[i] += (xd * xd).sum(0)
        done += k
    n = 2.0 * done
    return ([a1[i] / n for i in range(1, K31 + 1)],
            [fr[i] / n for i in range(1, K31 + 1)],
            [q1[i] / n for i in range(1, K31 + 1)])


def fit_eval(R, E, ridge=RIDGE):
    """`β` (вектор на кожен скаляр) із 1-ї половини жеребів, оцінка на 2-й.

    🚨 Стовпці СТАНДАРТИЗУЮТЬСЯ і ридж помітний: у першій версії ⛔НЕГ давав
    0.250× замість ≈1.0 — стовпці (середні по шарах) майже колінеарні, і при
    ридж=1e−6 розв'язок вибухав. Вада стенда, спіймана власним контролем.
    """
    h = R.shape[0] // 2
    sd = R[:h].std(0, keepdim=True).clamp_min(1e-30)
    R = R / sd
    Rf, Ef, Rv, Ev = R[:h], E[:h], R[h:], E[h:]
    G = Rf.T @ Rf
    rg = ridge * float(torch.diagonal(G).mean().clamp_min(1e-300))
    B = torch.linalg.solve(G + rg * torch.eye(G.shape[0], dtype=R.dtype,
                                              device=R.device), Rf.T @ Ef)
    res = Ev - Rv @ B
    return float((Ev * Ev).sum()), float((res * res).sum())


def main(nets=24, dev=DEV):
    torch.manual_seed(2718)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    rr = {s: {lay: [] for lay in PROBE} for s in STATS}     # r = зміщ./шум
    gain = {(s, src): [0.0, 0.0] for s in STATS
            for src in ('замик.', '⚠️оракул')}
    gain[('усі три', 'замик.')] = [0.0, 0.0]
    gain[('усі три', '⚠️оракул')] = [0.0, 0.0]
    gain[('⛔НЕГ', 'замик.')] = [0.0, 0.0]

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl, f_cl, q_cl = closure_parts(wc, K31, dev, extras=True)
            TA, TF, TQ = truth_stats(wc, dev, 820_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            Es, S = [], {s: [] for s in STATS}
            for d in range(NDRAW):
                sd = 88_000_000 + 733 * n + d
                pr, Ms = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                   lam_s=LAM31, collect_means=True)
                Es.append(pr - T)
                for lay in PROBE:
                    a, fa, q = Ms[lay - 1]
                    S['E[a]'].append(a.mean())
                    S['частка'].append(fa.mean())
                    S['E[a²]'].append(q.mean())
            E = torch.stack(Es)
            cl = {'E[a]': [f_cl and m_cl[l - 1].mean() for l in PROBE],
                  'частка': [f_cl[l - 1].mean() for l in PROBE],
                  'E[a²]': [q_cl[l - 1].mean() for l in PROBE]}
            tr = {'E[a]': [TA[l - 1].mean() for l in PROBE],
                  'частка': [TF[l - 1].mean() for l in PROBE],
                  'E[a²]': [TQ[l - 1].mean() for l in PROBE]}
            cols = {}
            for s in STATS:
                M = torch.stack(S[s]).view(NDRAW, len(PROBE))
                cols[s] = M
                for j, lay in enumerate(PROBE):
                    sd_ = float(M[:, j].std())
                    bias = float(cl[s][j] - tr[s][j])
                    rr[s][lay].append(abs(bias) / max(sd_, 1e-300))
            for s in STATS:
                for src, ref in (('замик.', cl), ('⚠️оракул', tr)):
                    Rm = cols[s] - torch.tensor([float(v) for v in ref[s]],
                                                dtype=torch.float64,
                                                device=dev).unsqueeze(0)
                    a, b = fit_eval(Rm, E)
                    gain[(s, src)][0] += a
                    gain[(s, src)][1] += b
            for src, ref in (('замик.', cl), ('⚠️оракул', tr)):
                Rm = torch.cat([cols[s] - torch.tensor(
                    [float(v) for v in ref[s]], dtype=torch.float64,
                    device=dev).unsqueeze(0) for s in STATS], 1)
                a, b = fit_eval(Rm, E)
                gain[('усі три', src)][0] += a
                gain[('усі три', src)][1] += b
            Rm = torch.cat([cols[s] - torch.tensor(
                [float(v) for v in cl[s]], dtype=torch.float64,
                device=dev).unsqueeze(0) for s in STATS], 1)
            a, b = fit_eval(torch.roll(Rm, 7, 0), E)
            gain[('⛔НЕГ', 'замик.')][0] += a
            gain[('⛔НЕГ', 'замик.')][1] += b
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів\n')
    print('🔑 ГОЛОВНЕ: `r = |замикання − істина| / std(статистики)` — '
          'відношення зміщення до шуму (усадка = 1/(1+r²))')
    print(f'{"статистика":>10s} ' + ' '.join(f'{"шар " + str(l):>9s}'
                                             for l in PROBE))
    for s in STATS:
        print(f'{s:>10s} ' + ' '.join(f'{np.median(rr[s][l]):9.2f}'
                                      for l in PROBE))
    print('\n⚠️ВИГРАШ з померéжевим `β` (ВЕРХНЯ МЕЖА, не метод; фіт/оцінка по '
          'половинах жеребів):')
    print(f'{"статистика":>10s} {"замикання":>11s} {"⚠️оракул":>11s}')
    for s in list(STATS) + ['усі три']:
        a, b = gain[(s, 'замик.')]
        c, e = gain[(s, '⚠️оракул')]
        print(f'{s:>10s} {a / max(b, 1e-300):10.3f}× {c / max(e, 1e-300):10.3f}×')
    a, b = gain[('⛔НЕГ', 'замик.')]
    print(f'{"⛔НЕГ":>10s} {a / max(b, 1e-300):10.3f}×   ⛔ мусить бути ≈1.0')
    print('\n🔑 читати так: якщо `r` для «частка» помітно МЕНШИЙ, ніж для `E[a]`, '
          'то в сурогатів РІЗНА похибка замикання — і ACV має з чого будувати. '
          'Якщо однаковий — вирок «усі впираються в те саме» лишається.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
