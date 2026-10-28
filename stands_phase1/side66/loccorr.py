"""ПРЯМИЙ A/B: чи дає безкоштовна локальна поправка виграш У РЕАЛЬНОМУ САБМІТІ?

ЗВІДКИ. `k3rec.py`: поправка до `m` з ОЗНАК, які вже є в рекурсії
(`s·t·φ`, `s·(t²−1)·φ`, `s·φ`, `s·Φ`, `s·t²·φ`, `s`), дає R²=0.544 на локальній
похибці кроку і **1.077× за драбиною**. ⛔випадкові ознаки — 1.002× (пломба).

🚨 АЛЕ ЦЕ ПРАЙС-ЛИСТ, А ВОНИ ЗАВИЩУЮТЬ (`tfit`: 1.70× обіцяно → 1.056× дано).
Дві конкретні причини сумніватись:
  1. вимір ішов на рекурсії **30 шарів**, а в сабміті `KSIG=8` ⟹ замикання
     робить 9 кроків, і поправка застосовна ЛИШЕ там;
  2. драбина `price_closure` міряна на `base_real` — порті УРІЗАНОГО `side1`,
     а не повного sub72 з чотирма родинами каналів.

ЩО ТУТ. Прямий A/B на `base72.run_side1` (ПОВНИЙ sub72): рахуємо `s_cl, m_cl,
q_cl, P_cl` двічі — як є і з поправкою — і міряємо MSE проти незалежної істини.
🔑 `λ` НЕ чіпаємо: питання саме в тому, чи краще замикання допомагає тим самим
каналам. (Якщо допоможе — перефіт `λ` дасть ЩЕ, але це окремий крок.)

ПЛОМБИ
  ✅ β=0 мусить дати РІВНО базу;
  ⛔ ВИПАДКОВІ β тієї ж норми ⟹ ≈1.00×;
  🔑 β фітяться LOO ПО МЕРЕЖАХ; істина — НЕЗАЛЕЖНИЙ жереб від робочого.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (cut_and_budget, run_side1, W_, _phi, _Phi,    # noqa: E402
                    LAM_S, KSIG)

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = 24
NP_, CH = 65_536, 8192

S72 = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
       0.0088285, 0.0007922, 0.0015044, 0.0001062)
A72 = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
       -0.0025822, -0.0030778, 0.0006822, 0.0068061)
Q72 = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
       0.0189783, 0.0250559, 0.0340802, 0.0415433)
G72 = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
       -0.0146129, -0.0100612, 0.0167223, 0.0168109)


def mfeats(t, ph, Ph, s):
    """Шість ознак, УСІ вже є в кроці замикання."""
    return torch.stack([s * t * ph, s * (t * t - 1.0) * ph, s * ph,
                        s * Ph, s * t * t * ph, s], 0)


def closure(wc, ksig, dev, beta=None):
    """Замикання sub72 + (опційно) локальна поправка до `m` на кожному кроці."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    s_o, m_o, f_o, q_o = [], [], [], []
    for i in range(ksig + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        if i >= 1:
            s_o.append(s)
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if beta is not None:
            m = m + mfeats(t, ph, Ph, s).T @ beta[i]
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if i >= 1:
            m_o.append(m)
            f_o.append(Ph)
            q_o.append(d + m * m)
    return s_o, m_o, f_o, q_o


def truth_layers(wc, dev, seed, upto):
    """Істинні `m` активацій шарів 1..upto (для цілі фіту `δ`) і `C`."""
    n = [wc[i].shape[1] for i in range(upto + 1)]
    S = [torch.zeros(q, dtype=torch.float64, device=dev) for q in n]
    Q = [torch.zeros(q, q, dtype=torch.float64, device=dev) for q in n]
    rng = np.random.default_rng(seed)
    done = 0
    while done < NP_:
        c = min(CH, NP_ - done)
        z = torch.tensor(rng.standard_normal((c, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(upto + 1):
            a = torch.relu(a @ wc[i])
            ad = a.double()
            S[i] += ad.sum(0)
            Q[i] += ad.T @ ad
        done += c
    N = float(2 * NP_)
    M = [S[i] / N for i in range(upto + 1)]
    C = [Q[i] / N - torch.outer(M[i], M[i]) for i in range(upto + 1)]
    return M, C


def truth_last(wc, dev, seed, npair=131_072, chunk=16384):
    acc = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(L_):
            a = torch.relu(a @ wc[i])
        acc += a.double().sum(0)
        done += k
    return acc / float(2 * npair)


def run_collect(wc, npair, s_cl, m_cl, f_cl, q_cl, dev, seed):
    """Прогін sub72 БЕЗ каналів + МАТРИЦЯ внесків усіх 4 родин, перенесених
    до виходу. `pred(λ) = pred₀ − λᵀA` ⟹ λ фітуються з одного прогону.

    🔑 перенос ЛІНІЙНИЙ за `v`, тож усі `4·KSIG` ознак несуться одним матмулом
    поруч із `v` — рівно як `collect_anchors` у `base72`, але для 4 родин.
    """
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (x.T @ x) / denom
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    xa = torch.relu(x @ (wh @ wc[0]))
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = xa.double().mean(0) - sg0 * 0.3989422804014327
    ks = len(s_cl)
    A = torch.zeros(4 * ks, wc[0].shape[1], dtype=torch.float64, device=dev)
    for i in range(1, depth):
        p = xa @ wc[i]
        gate = (p[:1024] > 0).to(p.dtype).sum(0).double() / 1024.0
        wd = wc[i].double()
        v = (v @ wd) * gate
        A = (A @ wd) * gate.unsqueeze(0)
        if i <= ks:
            pd = p.double()
            mh = pd.mean(0)
            sh = (pd.pow(2).mean(0) - mh * mh).clamp_min(1e-30).sqrt()
            xa = torch.relu(p)
            xd = xa.double()
            A[0 * ks + i - 1] += _phi(mh / sh) * (sh - s_cl[i - 1])
            A[1 * ks + i - 1] += xd.mean(0) - m_cl[i - 1]
            A[2 * ks + i - 1] += xd.pow(2).mean(0) - q_cl[i - 1]
            A[3 * ks + i - 1] += gate - f_cl[i - 1]
        else:
            xa = torch.relu(p)
    return xa.double().mean(0) - v, A


def main(nets=8, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    NET = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            M, Ct = truth_layers(wc, dev, 700_000 + 7919 * n, KSIG)
            T = truth_last(wc, dev, 260_000 + 7919 * n)
            # нормальні рівняння для `β`: ціль — ЛОКАЛЬНА похибка кроку
            n0 = wc[0].shape[0]
            MA, MB = [], []
            for i in range(KSIG + 1):
                w = wc[i].double()
                Cin = (torch.eye(n0, dtype=torch.float64, device=dev)
                       if i == 0 else Ct[i - 1])
                min0 = (torch.zeros(n0, dtype=torch.float64, device=dev)
                        if i == 0 else M[i - 1])
                cp = w.T @ (Cin @ w)
                s = cp.diagonal().clamp_min(1e-20).sqrt()
                mp = min0 @ w
                t = mp / s
                ph, Ph = _phi(t), _Phi(t)
                mg = mp * Ph + s * ph
                X = mfeats(t, ph, Ph, s).T
                dl = M[i] - mg
                MA.append(X.T @ X)
                MB.append(X.T @ dl)
            NET.append(dict(wc=wc, npair=npair, T=T, MA=MA, MB=MB))
        del W
        torch.cuda.empty_cache()
        print(f'  … збір {n+1}/{nets}', flush=True)

    res = {k: [] for k in ('A база sub72', '✅B локальна поправка',
                           '⛔C випадкові β')}
    NRM = {k: [] for k in res}
    STOR = []
    for j, net in enumerate(NET):
        wc, npair, T = net['wc'], net['npair'], net['T']
        oth = [q for q in range(len(NET)) if q != j]
        bl = []
        for i in range(KSIG + 1):
            A = sum(NET[q]['MA'][i] for q in oth)
            B = sum(NET[q]['MB'][i] for q in oth)
            rg = 1e-8 * float(A.diagonal().mean())
            bl.append(torch.linalg.solve(
                A + rg * torch.eye(A.shape[0], dtype=A.dtype, device=A.device),
                B))
        g = torch.Generator(device=dev).manual_seed(11 + j)
        rb = [torch.randn(b.shape, dtype=b.dtype, device=dev, generator=g)
              * b.norm() / (b.numel() ** 0.5) for b in bl]
        VAR = (('A база sub72', None), ('✅B локальна поправка', bl),
               ('⛔C випадкові β', rb))
        with torch.no_grad():
            for nm, bb in VAR:
                s_cl, m_cl, f_cl, q_cl = closure(wc, KSIG, dev, beta=bb)
                acc = 0.0
                GG = torch.zeros(4 * KSIG, 4 * KSIG, dtype=torch.float64,
                                 device=dev)
                BB = torch.zeros(4 * KSIG, dtype=torch.float64, device=dev)
                C0 = 0.0
                for d in range(NDRAW):
                    pr, _ = run_side1(wc, npair, s_cl, m_cl, dev,
                                      41_000_000 + 977 * j + d,
                                      lam_s=S72, lam_a=A72, lam_q=Q72,
                                      lam_g=G72, q_ext=q_cl, p_ext=f_cl)
                    acc += float(((pr - T) ** 2).mean())
                    p0, Am = run_collect(wc, npair, s_cl, m_cl, f_cl, q_cl,
                                         dev, 41_000_000 + 977 * j + d)
                    GG += Am @ Am.T
                    BB += Am @ (p0 - T)
                    C0 += float(((p0 - T) ** 2).sum())
                res[nm].append(acc / NDRAW)
                NRM[nm].append((GG, BB, C0, float(T.numel())))
        STOR.append((wc, npair, T, bl, rb))
        print(f'  … прогін {j+1}/{len(NET)}', flush=True)

    # ── ПЕРЕФІТ λ (LOO ПО МЕРЕЖАХ), аналітично з нормальних рівнянь ──
    # 🔑 MSE(λ) = (C0 − 2λᵀB + λᵀGλ) / (NDRAW·n) — повторні прогони не потрібні
    ref = {}
    for nm in NRM:
        tot = 0.0
        for j in range(len(NET)):
            oth = [q for q in range(len(NET)) if q != j]
            G = sum(NRM[nm][q][0] for q in oth)
            B = sum(NRM[nm][q][1] for q in oth)
            rg = 1e-10 * float(G.diagonal().mean())
            lam = torch.linalg.solve(
                G + rg * torch.eye(G.shape[0], dtype=G.dtype, device=G.device), B)
            Gj, Bj, C0j, nj = NRM[nm][j]
            v = (C0j - 2.0 * float(lam @ Bj) + float(lam @ (Gj @ lam)))
            tot += v / (NDRAW * nj)
        ref[nm] = tot / len(NET)
    a_ref = ref['A база sub72']
    print(f'\n📐 З ПЕРЕФІЧЕНИМИ `λ` (LOO по мережах, усі 4 родини):')
    print(f'{"варіант":>26s} {"MSE":>12s} {"ВИГРАШ проти A":>16s}')
    print('-' * 58)
    for nm in ('A база sub72', '⛔C випадкові β', '✅B локальна поправка'):
        print(f'{nm:>26s} {ref[nm]:12.4e} {a_ref/ref[nm]:15.4f}×')
    print('-' * 58)
    print(f'✅ПЛОМБА: «A + перефічені λ» = {a_ref:.4e} проти «A + λ сабміту» '
          f'{np.mean(res["A база sub72"]):.4e}\n   ⟹ якщо близько, `λ` сабміту '
          f'СПРАВДІ в оптимумі.\n')

    print(f'\n{len(NET)} мереж × {NDRAW} жеребів, ПОВНИЙ sub72 (4 родини), '
          f'KSIG={KSIG}\n')
    print(f'{"варіант":>26s} {"MSE":>12s} {"ВИГРАШ":>9s} {"кращих":>8s}')
    print('-' * 60)
    a0 = res['A база sub72']
    for nm in ('A база sub72', '⛔C випадкові β', '✅B локальна поправка'):
        v = res[nm]
        # 🔑 відношення СУМ (не середнє відношень)
        gain = sum(a0) / max(sum(v), 1e-300)
        better = sum(1 for i in range(len(v)) if v[i] < a0[i])
        print(f'{nm:>26s} {np.median(v):12.4e} {gain:8.4f}× '
              f'{better:4d}/{len(v)}')
    print('-' * 60)
    print('🎯 ГЕЙТ: ✅B помітно >1.00× і ⛔C ≈1.00× ⟹ поправка ЖИВА В САБМІТІ.\n'
          '🚨 прайс-лист обіцяв 1.077× на 30 шарах; тут KSIG=8 ⟹ очікую МЕНШЕ.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
