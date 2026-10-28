"""ЧИ ЖИВЕ ПОХИБКА ЗАМИКАННЯ І ШУМ ВИБІРКИ В РІЗНИХ НАПРЯМКАХ. (мета 1e-8)

ЗВІДКИ (ідея юзера про каустики, переведена у вимірне).
`last_real.py` виміряв `r_m = 4.85` і звідси `β_m = 0.041` ⟹ виграш лише 1.04×.
АЛЕ це число ІЗОТРОПНЕ: `r_m = mean_j|c_j| / mean_j|шум_j|` — усереднення по
всіх 256 нейронах ОКРЕМО. Оптимальна усадка так не працює: вона поканальна,
    β_u = Var(шум_u) / (Var(шум_u) + c_u²)
і якщо шум СКОНЦЕНТРОВАНИЙ у кількох напрямках (С65: похибка живе в ~6 із 256),
а похибка замикання РОЗМАЗАНА по всіх 256, то в тих кількох напрямках
    r_u² ≈ (‖c‖²/256) / (Σvar/6) = r² · 6/256 = 4.85²·0.023 = 0.55
тобто `r_u ≈ 0.74`, `β_u ≈ 0.65` і виграш ~2.8× замість 1.04×. Ізотропне
число тоді СИСТЕМАТИЧНО ховає важіль. Цього не міряли жодного разу.

ЩО САМЕ ПЕРЕВІРЯЄМО (три різні твердження, не одне)
  1. чи залежить `r` від напрямку взагалі (рядки «по нейронах», «власні»);
  2. чи вгадуються ці напрямки З САМИХ ВАГ — власні вектори `C_p` із замикання
     (це і є «структурна стійкість»: напрямок задано геометрією, не жеребом);
  3. чи ПЕРЕНОСИТЬСЯ профіль `β(ранг)` між мережами (LOO по мережах) — без
     цього все залишається оракулом.

ЯК. Ціль — СПРАВЖНЯ похибка фіналу, без лінеаризації:
    y = mean[relu(p₃₁)] − істина                (256)
    x = Φ(m̂/ŝ) ⊙ (m̂ − m_замикання)             (256, БЕЗ істини)
β підганяється на ПОЛОВИНІ жеребів, міряється на ДРУГІЙ половині.

КОНТРОЛІ
  * ⛔ ВИПАДКОВИЙ базис — має дати ≈ скаляр; якщо він дає стільки ж, скільки
    «власні», то виграш = переобучення на 256 параметрах, а не геометрія;
  * ✅ ОРАКУЛ `x = Φ(m̂ − m_істина)`, β=1 — має відтворити ~19.9× (last_ceiling);
  * ✅ слід Σvar однаковий у всіх базисах (друкуємо) — арифметична пломба;
  * β на held-out жеребах ⟹ оракул не ділить жереб із ціллю;
  * істина власна, 4М свіжих семплів.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import (GATE_N, INV_SQRT_2PI, MIN_PAIRS,     # noqa: E402
                         PILOT_PAIRS, TARGET_UTIL, B_BUDGET)
from transport_r2 import cut_net                              # noqa: E402
from born_source import truth_moments, _Phi                   # noqa: E402
from closure_precision import NQ, relu_cov0, relu_cov_gauss   # noqa: E402


def closure_last(wc, nodes, wts, upto):
    """(m, s, C) доактиваційних `p` шару `upto` із самих ваг — ПОВНА коваріація."""
    w0 = wc[0].double()
    m_est = (w0.pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    C_est = relu_cov0(wc[0])
    for i in range(1, upto + 1):
        Wd = wc[i].double()
        C_p = Wd.T @ (C_est @ Wd)
        m_p = m_est @ Wd
        if i == upto:
            return m_p, C_p.diagonal().clamp_min(1e-30).sqrt(), C_p
        C_est, m_est = relu_cov_gauss(m_p, C_p, nodes, wts)

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_N = 4_194_304
UPTO = 31
NSEED = 32
SEED0 = 400_000


def run_one(wc, seed, dev):
    """Один жереб оцінювача sub66: (mean relu(p₃₁), m̂₃₁, ŝ₃₁)."""
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    per_cut = sum(2.0 * w.shape[0] * w.shape[1] for w in wc)
    used = 2.0 * PILOT_PAIRS * (2.0 * L_ * W_ * W_)
    npair = max(MIN_PAIRS, int((TARGET_UTIL * B_BUDGET - used) / (2.0 * per_cut)))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    x = torch.cat([z, -z], 0)
    n2 = float(x.shape[0])
    cov = (x.T @ x) / n2
    ev, evec = torch.linalg.eigh(cov.double())
    wh = (evec * (1.0 / ev.clamp_min(1e-8).sqrt()) @ evec.T).float()
    x = torch.relu(x @ (wh @ wc[0]))
    m = s = None
    for i in range(1, L_):
        p = x @ wc[i]
        if i == L_ - 1:
            pd = p.double()
            m = pd.sum(0) / n2
            s = ((pd * pd).sum(0) / n2 - m * m).clamp_min(1e-30).sqrt()
        x = torch.relu(p)
    return x.double().sum(0) / n2, m, s


def held_out(Y, X, U, nfit):
    """β по кожній координаті базису U на перших nfit жеребах, MSE на решті."""
    Yb, Xb = Y @ U, X @ U
    a = (Yb[:nfit] * Xb[:nfit]).sum(0)
    b = (Xb[:nfit] * Xb[:nfit]).sum(0)
    beta = a / b.clamp_min(1e-300)
    return float(((Yb[nfit:] - beta * Xb[nfit:]) ** 2).sum()), beta


def scalar_ho(Y, X, nfit):
    a = float((Y[:nfit] * X[:nfit]).sum())
    b = float((X[:nfit] * X[:nfit]).sum())
    beta = a / max(b, 1e-300)
    return float(((Y[nfit:] - beta * X[nfit:]) ** 2).sum()), beta


def main(nets=6, dev=DEV, nseed=NSEED):
    torch.manual_seed(1234)
    nd, wt = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nd, dtype=torch.float64, device=dev)
    wts = torch.tensor(wt / math.sqrt(2 * math.pi), dtype=torch.float64, device=dev)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]
    nfit = nseed // 2

    Q, _ = torch.linalg.qr(torch.randn(W_, W_, dtype=torch.float64, device=dev))
    store = []           # (Y, X, U_clo, U_emp) на мережу
    diag, rprof = [], []
    for k, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            tr, mu_t, sg_t = truth_moments(wc, dev, 90001 + k, TRUTH_N)
            mc, sc, C_clo = closure_last(wc, nodes, wts, UPTO)
            mt = mu_t[L_ - 2].double()
            t_last = tr[-1].double()
            Y, X, XO = [], [], []
            for j in range(nseed):
                raw, m, s = run_one(wc, SEED0 + 1009 * j, dev)
                Ph = _Phi(m / s)
                Y.append(raw - t_last)
                X.append(Ph * (m - mc))
                XO.append(Ph * (m - mt))
            Y = torch.stack(Y); X = torch.stack(X); XO = torch.stack(XO)

            # базис 1: власні вектори передбаченої коваріації шуму, ЛИШЕ з ваг
            ph = _Phi(mc / sc)
            Cn = (ph.unsqueeze(1) * C_clo) * ph.unsqueeze(0)
            _, U_clo = torch.linalg.eigh(Cn)
            U_clo = U_clo.flip(1)                       # спадання власних чисел
            # базис 2: емпіричні власні вектори шуму з ПІДГІННОЇ половини
            Yc = Y[:nfit] - Y[:nfit].mean(0, keepdim=True)
            _, U_emp = torch.linalg.eigh(Yc.T @ Yc)
            U_emp = U_emp.flip(1)

            store.append((Y, X, XO, U_clo, U_emp))
            ev_pred = (U_clo * (Cn @ U_clo)).sum(0)
            pr = float(ev_pred.sum() ** 2 / (ev_pred * ev_pred).sum())
            evY = ((Y - Y.mean(0, keepdim=True)) @ U_clo).pow(2).mean(0)
            prY = float(evY.sum() ** 2 / (evY * evY).sum())
            cU = ((mc - mt) * ph) @ U_clo
            top = int(max(1, round(prY)))
            r_top = float((cU[:top].pow(2).sum() / evY[:top].sum()).sqrt())
            r_iso = float((cU.pow(2).sum() / evY.sum()).sqrt())
            diag.append((pr, prY, r_iso, r_top))
            rprof.append((cU.abs() / evY.clamp_min(1e-300).sqrt()).cpu().numpy())
        print(f'  мережа {k+1}/{len(fs)}  PR(перед)={pr:.1f} PR(факт)={prY:.1f} '
              f'r_ізо={r_iso:.2f} r_топ={r_top:.2f}', flush=True)

    print(f'\n{len(fs)} мереж × {nseed} жеребів ({nfit} підгін / {nseed-nfit} вимір), '
          f'істина {TRUTH_N}')
    print(f'середні: PR передбачений {np.mean([d[0] for d in diag]):.1f}, '
          f'фактичний {np.mean([d[1] for d in diag]):.1f}; '
          f'r ізотропний {np.mean([d[2] for d in diag]):.2f}, '
          f'r у топ-напрямках {np.mean([d[3] for d in diag]):.2f}')

    rows = {k: 0.0 for k in (
        'база (без поправки)', 'скаляр β (нинішнє)', 'по нейронах β',
        'власні напрямки замикання ★', 'власні напрямки, β з ІНШИХ мереж',
        'емпіричні власні напрямки', '⛔ випадковий базис',
        '✅ ОРАКУЛ Φ(m̂−істина), β=1')}
    I = torch.eye(W_, dtype=torch.float64, device=dev)
    betas = []
    for Y, X, XO, U_clo, U_emp in store:
        _, b = held_out(Y, X, U_clo, nfit)
        betas.append(b)
    for i, (Y, X, XO, U_clo, U_emp) in enumerate(store):
        rows['база (без поправки)'] += float((Y[nfit:] ** 2).sum())
        rows['скаляр β (нинішнє)'] += scalar_ho(Y, X, nfit)[0]
        rows['по нейронах β'] += held_out(Y, X, I, nfit)[0]
        rows['власні напрямки замикання ★'] += held_out(Y, X, U_clo, nfit)[0]
        rows['емпіричні власні напрямки'] += held_out(Y, X, U_emp, nfit)[0]
        rows['⛔ випадковий базис'] += held_out(Y, X, Q, nfit)[0]
        bl = torch.stack([b for j, b in enumerate(betas) if j != i]).mean(0)
        Yb, Xb = Y[nfit:] @ U_clo, X[nfit:] @ U_clo
        rows['власні напрямки, β з ІНШИХ мереж'] += float(((Yb - bl * Xb) ** 2).sum())
        rows['✅ ОРАКУЛ Φ(m̂−істина), β=1'] += float(((Y[nfit:] - XO[nfit:]) ** 2).sum())

    base = rows['база (без поправки)']
    print(f'\n{"варіант":34s} {"MSE":>11s} {"×":>8s}')
    print('-' * 56)
    for k in ('база (без поправки)', 'скаляр β (нинішнє)', 'по нейронах β',
              'власні напрямки замикання ★', 'власні напрямки, β з ІНШИХ мереж',
              'емпіричні власні напрямки', '⛔ випадковий базис',
              '✅ ОРАКУЛ Φ(m̂−істина), β=1'):
        v = rows[k] / (len(store) * (NSEED - nfit) * W_)
        print(f'{k:34s} {v:11.4e} {base/rows[k]:8.3f}')

    R = np.log(np.array(rprof) + 1e-12)
    cc = [float(np.corrcoef(R[i], R[j])[0, 1])
          for i in range(len(R)) for j in range(i + 1, len(R))]
    print(f'\nстійкість профілю r(ранг) між мережами: ρ = {np.mean(cc):+.3f} '
          f'(розкид {np.std(cc):.3f}, {len(cc)} пар) — >0.5 означало б, що '
          f'напрямки похибки задані геометрією, а не жеребом')
    print('  r(ранг) по сегментах, середнє по мережах: ' + '  '.join(
        f'{a}..{b-1}: {np.mean(np.array(rprof)[:, a:b]):.2f}'
        for a, b in ((0, 4), (4, 16), (16, 64), (64, 256))))

    B = torch.stack(betas).cpu().numpy()
    print('\nпрофіль β по рангу власного напрямку (середнє ± розкид по мережах):')
    for a, b in ((0, 4), (4, 16), (16, 64), (64, 256)):
        seg = B[:, a:b]
        print(f'  ранги {a:3d}..{b-1:3d}: β = {seg.mean():+.3f} ± {seg.mean(1).std():.3f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
