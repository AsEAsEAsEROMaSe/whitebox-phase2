"""ПІЛОТ І ПІДЛОГА: свіп `PILOT_PAIRS` × `TARGET_UTIL`. (С74)

ЗВІДКИ. [[finding-we-are-3.5-percent-from-the-cost-floor]]: ми на множнику
0.1034 при підлозі 0.1 ⟹ 3.5% флопів до зони, де ціна не рахується.
Пілот коштує 3.9% флопів — але керує ще й АГРЕСИВНІСТЮ ОБРІЗАННЯ.

🔑 ДВА ЕФЕКТИ, І ДРУГИЙ БІЛЬШИЙ:
  1. менший пілот ⟹ менше флопів на сам пілот (`used` ↓);
  2. менший пілот ⟹ БІЛЬШЕ нейронів виглядають мертвими ⟹ менші матриці ⟹
     `per_cut` ↓ ⟹ **більше `n_pairs` на ті самі флопи** (85% бюджету — цикл).
`n_pairs = (util·B − used) / (2·per_cut)` ⟹ обидва ефекти ЗБІЛЬШУЮТЬ вибірку
при НЕЗМІННОМУ `F/B`. Виграш іде прямо в MSE.
🚨 ПЛАТА: обрізаний ЖИВИЙ нейрон — це ЗМІЩЕННЯ, яке не падає з `n`.

Друга вісь — `TARGET_UTIL`: щоб сісти РІВНО на підлогу (`F/B ≤ 0.0941`), її
треба опустити з 0.0942 приблизно до 0.0908.

ЩО РАХУЄМО для кожної пари (пілот, util):
  `n_pairs` · `F/B` (аналітично, по доданках) · MSE · БАЛ = MSE·max(0.1, F/B+λR/B)
  — і те саме з ПЕРЕФІЧЕНИМИ `λ` (бо `λ` є функцією `n`).

ПЛОМБИ
  ✅ (128, 0.0942) мусить відтворити sub86: `F/B≈0.0975`, бал ≈1.97e-07;
  ✅ `λ` перефічені на (128, 0.0942) мусять дати ≈ те саме, що `λ` сабміту
     (у `loccorr.py` різниця була 0.3%);
  📊 8 мереж × 24 жереби; істина — НЕЗАЛЕЖНИЙ жереб.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base72 import (run_side1, closure_parts, W_, _phi,           # noqa: E402
                    KSIG, B_BUDGET, MIN_PAIRS, SEED)

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
NDRAW = 24
LAM_R = 0.0059            # λR/B діючого сабміту (грейдерний)

PILOTS = (16, 32, 64, 128, 256)
UTILS = (0.0908, 0.0942)

S72 = (0.0017899, 0.0047168, 0.0071225, 0.0040152,
       0.0088285, 0.0007922, 0.0015044, 0.0001062)
A72 = (-0.0067395, -0.0091340, -0.0092425, -0.0049866,
       -0.0025822, -0.0030778, 0.0006822, 0.0068061)
Q72 = (0.0010829, 0.0000248, 0.0095619, 0.0034287,
       0.0189783, 0.0250559, 0.0340802, 0.0415433)
G72 = (-0.0047241, 0.0103696, 0.0100503, 0.0273080,
       -0.0146129, -0.0100612, 0.0167223, 0.0168109)


def cut_and_budget(W, dev, pilot, util):
    """Порт `cut_and_budget` із параметрами пілота і цільового використання."""
    depth = len(W)
    rng = np.random.default_rng(SEED)
    zp = torch.tensor(rng.standard_normal((pilot, W_)).astype(np.float32),
                      device=dev)
    a = torch.cat([zp, -zp], 0)
    alive = []
    for i in range(depth):
        h = a @ W[i]
        if i < depth - 1:
            alive.append(h.max(0).values >= 0.0)
        a = torch.relu(h)
    wc = []
    for i in range(depth):
        wi = W[i]
        if i > 0:
            wi = wi[alive[i - 1], :]
        if i < depth - 1:
            wi = wi[:, alive[i]]
        wc.append(wi.contiguous())
    per_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in wc)
    used = 2.0 * pilot * (2.0 * depth * W_ * W_)
    left = util * float(B_BUDGET) - used
    npair = max(MIN_PAIRS, int(left / (2.0 * max(per_cut, 1.0))))
    return wc, npair, per_cut, used


def flops_total(wc, npair, per_cut, used, ksig):
    """F по доданках: пілот + цикл + замикання + ZCA/шар0."""
    f = used + 2.0 * npair * per_cut
    for i in range(ksig + 1):                    # замикання: C@w, wᵀ@(·)
        a, b = wc[i].shape
        f += 2.0 * a * a * b + 2.0 * a * b * b + 6.0 * b * b
    n0 = wc[0].shape[0]
    f += 2.0 * n0 * n0 * n0                      # ZCA: eigh+матмули ~n³
    f += 2.0 * npair * n0 * n0                   # cov на `n` рядках (sub86)
    return f


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
    """Прогін БЕЗ каналів + матриця внесків 4 родин ⟹ λ фітуються лінійно."""
    depth = len(wc)
    g = torch.Generator(device=dev).manual_seed(int(seed))
    z = torch.randn(npair, W_, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    denom = float(2 * npair)
    cov = (z.T @ z) / float(npair)
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

    KEYS = [(p, u) for u in UTILS for p in PILOTS]
    acc = {k: {'mse': [], 'fb': [], 'np': [], 'nrm': []} for k in KEYS}
    NORM = {k: [] for k in KEYS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            Tfull = None
            for (pl, ut) in KEYS:
                wc, npair, per_cut, used = cut_and_budget(W, dev, pl, ut)
                if Tfull is None:
                    Tfull = truth_last(W, dev, 260_000 + 7919 * n)
                # 🚨 істина рахується на ПОВНІЙ мережі; обрізання — частина
                #    методу, тож його похибка ВХОДИТЬ у MSE (як і має бути)
                T = Tfull
                ks = min(KSIG, len(wc) - 1)
                s_cl, m_cl, f_cl, q_cl = closure_parts(wc, ks, dev, extras=True)
                F = flops_total(wc, npair, per_cut, used, ks)
                a2 = 0.0
                GG = torch.zeros(4 * ks, 4 * ks, dtype=torch.float64, device=dev)
                BB = torch.zeros(4 * ks, dtype=torch.float64, device=dev)
                C0 = 0.0
                for d in range(NDRAW):
                    sd = 52_000_000 + 977 * n + d
                    pr, _ = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                      lam_s=S72[:ks], lam_a=A72[:ks],
                                      lam_q=Q72[:ks], lam_g=G72[:ks],
                                      q_ext=q_cl, p_ext=f_cl)
                    a2 += float(((pr - T) ** 2).mean())
                    p0, Am = run_collect(wc, npair, s_cl, m_cl, f_cl, q_cl,
                                         dev, sd)
                    GG += Am @ Am.T
                    BB += Am @ (p0 - T)
                    C0 += float(((p0 - T) ** 2).sum())
                acc[(pl, ut)]['mse'].append(a2 / NDRAW)
                acc[(pl, ut)]['fb'].append(F / B_BUDGET)
                acc[(pl, ut)]['np'].append(npair)
                NORM[(pl, ut)].append((GG, BB, C0, float(T.numel())))
                del wc
                torch.cuda.empty_cache()
        del W
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    md = np.median
    print(f'\n{nets} мереж × {NDRAW} жеребів, λR/B={LAM_R} (грейдерний)\n')
    print(f'{"пілот":>6s} {"util":>7s} {"n_pairs":>8s} {"F/B":>7s} '
          f'{"множник":>8s} {"MSE(λ72)":>11s} {"БАЛ":>11s} {"виграш":>8s} │ '
          f'{"MSE(λ*)":>11s} {"БАЛ*":>11s} {"виграш*":>8s}')
    print('-' * 118)
    base = None
    for (pl, ut) in KEYS:
        a = acc[(pl, ut)]
        fb = md(a['fb'])
        mult = max(0.1, fb + LAM_R)
        m1 = md(a['mse'])
        s1 = m1 * mult
        # ── перефіт λ, LOO по мережах, аналітично ──
        tot = 0.0
        for j in range(nets):
            oth = [q for q in range(nets) if q != j]
            G = sum(NORM[(pl, ut)][q][0] for q in oth)
            Bv = sum(NORM[(pl, ut)][q][1] for q in oth)
            rg = 1e-10 * float(G.diagonal().mean())
            lam = torch.linalg.solve(
                G + rg * torch.eye(G.shape[0], dtype=G.dtype, device=G.device),
                Bv)
            Gj, Bj, C0j, nj = NORM[(pl, ut)][j]
            tot += (C0j - 2.0 * float(lam @ Bj)
                    + float(lam @ (Gj @ lam))) / (NDRAW * nj)
        m2 = tot / nets
        s2 = m2 * mult
        if base is None:
            base = (s1, s2)
        tag = ' ✅ПЛОМБА sub86' if (pl == 128 and ut == 0.0942) else ''
        print(f'{pl:6d} {ut:7.4f} {int(md(a["np"])):8d} {fb:7.4f} {mult:8.4f} '
              f'{m1:11.4e} {s1:11.4e} {base[0]/s1:7.3f}× │ '
              f'{m2:11.4e} {s2:11.4e} {base[1]/s2:7.3f}×{tag}')
    print('-' * 118)
    print('🎯 ЧИТАТИ: колонка «виграш*» (з перефіченими `λ`) — це те, що можна '
          'узяти.\n'
          '✅ПЛОМБА: рядок (128, 0.0942) мусить дати F/B≈0.0975 і бал ≈1.97e-07.\n'
          '🚨 менший пілот ⟹ агресивніше обрізання: якщо MSE РОСТЕ — ріжемо '
          'ЖИВИХ.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
