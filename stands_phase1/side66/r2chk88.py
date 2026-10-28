"""ПЕРЕВІРКА ЧУЖОГО ЧИСЛА: чи справді `R²=0.666` від `(t_j,t_k,ρ)`? (С88)

`pairscan87` знімав це на **6 мережах × 4 шари** — це МАЛИЙ ПРОГІН за нашим
власним правилом (вирок ≥24 мережі). Тут: ≥24 мережі × УСІ 31 шар.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) стеля 14³ впаде нижче 0.666, бо 4 «пробні» шари (7,15,23,30) — не
     репрезентативні, а квантильні краї на 6 мережах підігнані. Шанс 55%.
 (б) стеля лишиться ≥0.60 (число підтверджено). Шанс 40%.
 (в) стеля виявиться СИЛЬНО нижчою (<0.35) ⟹ гілка мертва до всякого
     scheduled sampling. Шанс 20%.
 (г) R² сильно залежить від шару (глибокі шари гірші). Шанс 50%.

⛔ПЛОМБИ: P1 тавтологія=1.0 · P2 випадкова ознака≈0 · P3 перемішані пари≈0 ·
P4 LOO по мережах · P5 симетризація j↔k.
"""
import glob, math, sys, os
import numpy as np, torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NPAIR = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
assert DEV not in ('cuda:0', 'cuda:4')
CACHE = '/tmp/claude-1000/-mnt-win-d-llm-White-Box/5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/mc88cache'
L_, W_ = 32, 256
I2 = 1.0 / math.sqrt(2.0 * math.pi)
phi = lambda x: torch.exp(-0.5 * x * x) * I2
Phi = lambda x: 0.5 * torch.erfc(-x * 0.7071067811865476)
print(__doc__.split('📣')[1], flush=True)

files = sorted(glob.glob(f'{CACHE}/n*.npz'))[:NETS]
print(f'мереж у кеші: {len(files)} · {NPAIR} пар/шар/мережа\n', flush=True)
ys, tjs, tks, rs, gs, ls = [], [], [], [], [], []
for n, f in enumerate(files):
    d = np.load(f)
    mhA = torch.tensor(d['mh'], dtype=torch.float64, device=DEV)
    ChA = torch.tensor(d['Ch'], dtype=torch.float64, device=DEV)
    CaA = torch.tensor(d['Ca'], dtype=torch.float64, device=DEV)
    rg = torch.Generator(device=DEV); rg.manual_seed(500 + n)
    for q in range(L_ - 1):
        mh, Ch, Ca = mhA[q], ChA[q], CaA[q]
        s = torch.sqrt(torch.clamp(torch.diagonal(Ch), min=1e-30))
        t = mh / s; P, fq = Phi(t), phi(t)
        m_a = mh * P + s * fq
        Ccl = Ch * torch.outer(P, P) + (0.5 * (Ch * Ch)) * torch.outer(fq / s, fq / s)
        Y = (Ca - Ccl) / torch.outer(s, s)
        rho = Ch / torch.outer(s, s)
        iu = torch.randint(0, W_, (NPAIR,), device=DEV, generator=rg)
        ju = torch.randint(0, W_, (NPAIR,), device=DEV, generator=rg)
        m_ = iu != ju; iu, ju = iu[m_], ju[m_]
        ys.append(Y[iu, ju].cpu().numpy())
        tjs.append(t[iu].cpu().numpy()); tks.append(t[ju].cpu().numpy())
        rs.append(rho[iu, ju].cpu().numpy())
        gs.append(np.full(len(iu), n)); ls.append(np.full(len(iu), q))
    print(f'  net{n}', flush=True)
y = np.concatenate(ys); tj = np.concatenate(tjs); tk = np.concatenate(tks)
rho = np.concatenate(rs); grp = np.concatenate(gs); lay = np.concatenate(ls)
print(f'\n{"="*70}\nточок: {len(y)} · RMS цілі: {y.std():.4e} · середнє: {y.mean():+.4e}')

X = np.column_stack([rho, np.abs(rho), rho**2, rho**3, tj + tk, np.abs(tj - tk),
                     tj * tk, Phi(torch.tensor(tj)).numpy() * Phi(torch.tensor(tk)).numpy(),
                     phi(torch.tensor(tj)).numpy() * phi(torch.tensor(tk)).numpy(),
                     rho * (tj * tk), rho * np.abs(tj - tk), rho * rho * (tj * tk)])
FEAT = ('rho', '|rho|', 'rho^2', 'rho^3', 't_j+t_k', '|t_j-t_k|', 't_j*t_k',
        'Phi_j*Phi_k', 'phi_j*phi_k', 'rho*tt', 'rho*|dt|', 'rho^2*tt')


def r2lin(Xs, yy, gg):
    """LOO по мережах через «сума мінус група» — без 40 копій великої матриці."""
    Xs = np.column_stack([Xs, np.ones(len(Xs))])
    ug = np.unique(gg)
    XtX = {gi: Xs[gg == gi].T @ Xs[gg == gi] for gi in ug}
    Xty = {gi: Xs[gg == gi].T @ yy[gg == gi] for gi in ug}
    TA = sum(XtX.values()); TB = sum(Xty.values())
    out = np.zeros(len(yy))
    for gi in ug:
        te = gg == gi
        A = TA - XtX[gi] + 1e-8 * np.eye(Xs.shape[1])
        out[te] = Xs[te] @ np.linalg.solve(A, TB - Xty[gi])
    return 1.0 - ((yy - out)**2).sum() / ((yy - yy.mean())**2).sum()


print('\nОДИНИЧНІ КОРЕЛЯЦІЇ (24+ мереж, УСІ шари)')
for a, nm, c in sorted([(abs(np.corrcoef(X[:, k], y)[0, 1]), nm,
                         np.corrcoef(X[:, k], y)[0, 1]) for k, nm in enumerate(FEAT)],
                       reverse=True):
    print(f'  {nm:14s} {c:+.4f}')
print(f'\nЛІНІЙНИЙ R² (LOO по мережах)')
print(f'  усі 12 парних ознак           {r2lin(X, y, grp):.4f}')
print(f'  лише rho-родина (4)           {r2lin(X[:, :4], y, grp):.4f}')
print(f'  лише t-родина (5..9)          {r2lin(X[:, 4:9], y, grp):.4f}')
rng = np.random.default_rng(7)
print(f'  ⛔P2 підлога (випадкова)       {r2lin(rng.standard_normal((len(y), 3)), y, grp):+.5f}')
print(f'  ⛔P3 перемішані пари           {r2lin(X[rng.permutation(len(y))], y, grp):+.5f}')
print(f'  ⛔P1 тавтологія                {r2lin(y[:, None], y, grp):.4f}')


def binr2(nb, extra=None, nex=1, tag=''):
    ed_t = np.quantile(np.concatenate([tj, tk]), np.linspace(0, 1, nb + 1))
    ed_r = np.quantile(rho, np.linspace(0, 1, nb + 1))
    a_ = np.clip(np.searchsorted(ed_t, tj) - 1, 0, nb - 1)
    b_ = np.clip(np.searchsorted(ed_t, tk) - 1, 0, nb - 1)
    c_ = np.clip(np.searchsorted(ed_r, rho) - 1, 0, nb - 1)
    lo, hi = np.minimum(a_, b_), np.maximum(a_, b_)
    idx = (lo * nb + hi) * nb + c_
    nbin = nb ** 3
    if extra is not None:
        idx = idx + nbin * extra; nbin = nbin * nex
    ug = np.unique(grp)
    SG = {gi: np.bincount(idx[grp == gi], weights=y[grp == gi], minlength=nbin) for gi in ug}
    CG = {gi: np.bincount(idx[grp == gi], minlength=nbin) for gi in ug}
    TS = sum(SG.values()); TC = sum(CG.values()); ymu = y.mean()
    out = np.zeros(len(y))
    for gi in ug:
        te = grp == gi
        sm = TS - SG[gi]; cn = TC - CG[gi]
        mu = np.where(cn > 5, sm / np.maximum(cn, 1), ymu)
        out[te] = mu[idx[te]]
    return 1.0 - ((y - out)**2).sum() / ((y - y.mean())**2).sum(), int((TC > 0).sum())


print(f'\n{"="*70}\n🔑 НЕПАРАМЕТРИЧНА СТЕЛЯ ВІД (t_j,t_k,ρ) — квантильні краї, LOO')
print(f'{"сітка":>10s} {"комірок":>9s} {"R²":>10s}')
for nb in (6, 10, 14, 20):
    r, nc = binr2(nb)
    print(f'{nb:>7d}^3 {nc:9d} {r:10.4f}')
print(f'\n🔑 ТЕ САМЕ + ІНДЕКС ШАРУ (F залежить від глибини)')
for nlb in (4, 31):
    lb = np.minimum((lay * nlb) // 31, nlb - 1) if nlb < 31 else lay
    for nb in (10, 14):
        r, nc = binr2(nb, extra=lb, nex=nlb)
        print(f'  шар-бакетів {nlb:2d} · сітка {nb}^3 · комірок {nc:6d} → R² {r:.4f}')

print(f'\n🔑 R² ПО ШАРАХ (сітка 14³, навчена НА ВСІХ шарах, LOO по мережах)')
ed_t = np.quantile(np.concatenate([tj, tk]), np.linspace(0, 1, 15))
ed_r = np.quantile(rho, np.linspace(0, 1, 15))
a_ = np.clip(np.searchsorted(ed_t, tj) - 1, 0, 13)
b_ = np.clip(np.searchsorted(ed_t, tk) - 1, 0, 13)
c_ = np.clip(np.searchsorted(ed_r, rho) - 1, 0, 13)
idx = (np.minimum(a_, b_) * 14 + np.maximum(a_, b_)) * 14 + c_
out = np.zeros(len(y))
_SG = {gi: np.bincount(idx[grp == gi], weights=y[grp == gi], minlength=14**3) for gi in np.unique(grp)}
_CG = {gi: np.bincount(idx[grp == gi], minlength=14**3) for gi in np.unique(grp)}
_TS = sum(_SG.values()); _TC = sum(_CG.values())
for gi in np.unique(grp):
    te = grp == gi
    sm = _TS - _SG[gi]; cn = _TC - _CG[gi]
    mu = np.where(cn > 5, sm / np.maximum(cn, 1), y.mean())
    out[te] = mu[idx[te]]
for q in range(0, 31, 3):
    m_ = lay == q
    rr = 1.0 - ((y[m_] - out[m_])**2).sum() / ((y[m_] - y[m_].mean())**2).sum()
    print(f'  шар {q:2d}: RMS цілі {y[m_].std():.3e} · R² {rr:+.4f}')
print('\n📌 якщо стеля ≈0.666 — число ЮЗЕРА підтверджено · якщо <0.35 — гілка мертва')
