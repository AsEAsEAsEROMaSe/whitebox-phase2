"""16.10.2026, ідея юзера «струмені з сопла: сили між вихорами рахуються рекурсією» = вихрові методи + дерево Barnes-Hut / FMM,
тобто ієрархічний низький ранг (HODLR: позадіагональні блоки кожного рівня бінарного дерева стиснуті до рангу r(δ)).
Глобальний низький ранг уже мертвий (D21 ранг 512 = MSE x1.26 гірше). Питання: чи стискаються БЛОКИ після впорядкування нейронів.
Матриці: C_off ланцюга (wblogs/live, дамп 0) і істинний зріз κ3 D21[i,j] = κ(p_i,p_i,p_j) з банку (шум 1e8 семплів ~3%).
Впорядкування: випадкове (контроль), за t, за першим сингулярним вектором. Метрика: пам'ять HODLR / n² при відносній похибці блоку δ.
Передбачення ДО прогону: геометрії нема (похибка обмінна по нейронах, W випадкова) ⟹ t і спектральне ≈ випадкове, HODLR/n² ≥ 0.7
при δ=0.01 для C_off; шанс ≤ 0.3: 5%."""
import numpy as np, torch
dev = 'cuda'; i = 0; n = 1024; LEAF = 64
kp = np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz' % i)

def rank_at(S, d):
    e = (S * S).flip(0).cumsum(0).flip(0)          # хвостова енергія від індексу k
    tot = e[0]
    ok = (e <= (d * d) * tot).nonzero()
    return int(ok[0]) if len(ok) else len(S)

def hodlr_cost(X, d):
    """пам'ять HODLR (слабка допустимість): на кожному рівні обидва позадіагональні блоки r(m+m'), листки щільні."""
    def rec(a, b):
        m = b - a
        if m <= LEAF: return m * m
        h = a + m // 2; c = 0
        for (r0, r1, c0, c1) in ((a, h, h, b), (h, b, a, h)):
            B = X[r0:r1, c0:c1]
            if float(B.norm()) == 0: continue
            S = torch.linalg.svdvals(B)
            c += rank_at(S, d) * ((r1 - r0) + (c1 - c0))
        return c + rec(a, h) + rec(h, b)
    return rec(0, n) / (n * n)

def orders(X, t):
    g = torch.Generator().manual_seed(7)
    U, S, Vh = torch.linalg.svd(X)
    return {'випадк': torch.randperm(n, generator=g).to(dev), 'за t': torch.argsort(t), 'спектр': torch.argsort(U[:, 0])}

for l in (4, 8, 12, 15):
    lv = np.load('/home/trim/wblogs/live/d0_l%02d.npz' % l)
    C = torch.tensor(lv['C'], dtype=torch.float64, device=dev); mu = torch.tensor(lv['mu'], dtype=torch.float64, device=dev)
    Coff = C - torch.diag(torch.diagonal(C)); t = mu / torch.diagonal(C).sqrt()
    a = torch.tensor(kp['pre_mean'][l], dtype=torch.float64, device=dev); E2 = torch.tensor(kp['pre_m2'][l], dtype=torch.float64, device=dev)
    M11 = torch.tensor(kp['pre_M11'][l], dtype=torch.float64, device=dev); M21 = torch.tensor(kp['pre_M21'][l], dtype=torch.float64, device=dev)
    D21 = M21 - a[None, :] * E2[:, None] - 2 * a[:, None] * M11 + 2 * (a * a)[:, None] * a[None, :]
    D21 = D21 - torch.diag(torch.diagonal(D21))
    for name, X, ds in (('C_off', Coff, (0.1, 0.03, 0.01)), ('D21', D21, (0.3, 0.1, 0.05))):
        Sg = torch.linalg.svdvals(X)
        glob = ' '.join('δ%.2f r%d (%.2f)' % (d, rank_at(Sg, d), 2 * rank_at(Sg, d) / n) for d in ds)
        print('l=%2d %-5s глобально: %s' % (l, name, glob), flush=True)
        for on, p in orders(X, t).items():
            Xp = X[p][:, p]
            print('      %-7s HODLR/n²: %s' % (on, ' '.join('δ%.2f %.3f' % (d, hodlr_cost(Xp, d)) for d in ds)), flush=True)
