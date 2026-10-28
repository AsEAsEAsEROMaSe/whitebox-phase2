"""НУЛЬ ДЛЯ ANOVA СКЛЕЙКИ ПОЛОВИН (аудит С48 пойнта finding-half-splice-anova).

ДВІ ПРЕТЕНЗІЇ ДО splice.py
1) БАГ: двоблокова ANOVA рахується НЕЗМІЩЕНО (модель випадкових ефектів, віднімає vAB),
   а чотириблокова — СИРИМИ середніми по ~100 семплах, без віднімання внутрішньогрупової
   дисперсії ⟹ головні ефекти завищені приблизно на 4*vtot/100 = 0.04.
   Висновок «взаємодія РОСТЕ з дрібнішим дробленням» порівнює виправлене з невиправленим.
2) НУЛЬ НЕ ПОРАХОВАНО: «взаємодія переважає (70%)» — а чиста ВИПАДКОВА КВАДРАТИКА вже дає
   50%, чиста квартика 87.5%. Питання не «чи переважає», а «чи більше, ніж зобов'язана».

ЩО РОБИМО: обидва оцінювачі незміщені, і прогін на СИНТЕТИЧНИХ функціях відомого порядку
(нуль ВИМІРЯНИЙ, не паперовий):
    f2 = z^T A z            чистий степінь 2  -> очікуємо головні 0.500 / 0.250
    f4 = sum_k c_k He4(w_k.z) чистий степінь 4 -> очікуємо головні 0.125 / 0.0156
Якщо синтетика лягає на передбачення — формула працює, і виміри мережі можна читати як
розклад по порядках. Якщо ні — формула неповна, і тоді ANOVA несе власну інформацію.
"""
import torch, time, numpy as np
LOG = '/mnt/win_d/llm/White-Box/splice_null.log'; T0 = time.time()
open(LOG, 'w').close()
def log(m):
    s = '[{:6.1f}s] {}'.format(time.time() - T0, m); print(s, flush=True)
    open(LOG, 'a').write(s + '\n')

dev = 'cuda:1'; D = 256; L = 32
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float32), device=dev) for i in range(L)]

def mk_even(W):
    def f(x):
        a = x
        for w in W: a = torch.relu(a @ w)
        b = -x
        for w in W: b = torch.relu(b @ w)
        return (a + b) * 0.5
    return f

# ---------- синтетичні функції відомого степеня (векторні, 64 виходи) ----------
NOUT = 64
def mk_quad(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    A = torch.randn(NOUT, D, D, device=dev, generator=g)
    A = (A + A.transpose(1, 2)) * 0.5
    def f(x): return torch.einsum('bi,oij,bj->bo', x, A, x)
    return f
def mk_quart(seed, NK=192):
    g = torch.Generator(device=dev).manual_seed(seed)
    Wp = torch.randn(D, NK, device=dev, generator=g); Wp /= Wp.norm(dim=0, keepdim=True)
    C = torch.randn(NK, NOUT, device=dev, generator=g)
    def f(x):
        t = x @ Wp                                   # одинична дисперсія по кожному стовпцю
        return (t ** 4 - 6.0 * t ** 2 + 3.0) @ C     # He4 -> чистий степінь 4
    return f

# ---------- 2 блоки: повна сітка M×K, незміщена модель випадкових ефектів ----------
def anova2(f, seed, M=96, K=96):
    g = torch.Generator(device=dev).manual_seed(seed)
    HA = torch.randn(M, D // 2, device=dev, generator=g)
    HB = torch.randn(K, D // 2, device=dev, generator=g)
    Y = None
    for i in range(M):
        x = torch.cat([HA[i].unsqueeze(0).expand(K, -1), HB], 1)
        y = f(x).double()
        if Y is None: Y = torch.empty(M, K, y.shape[1], dtype=torch.float64, device=dev)
        Y[i] = y
    gm = Y.mean((0, 1)); ri = Y.mean(1); cj = Y.mean(0)
    SSA = float((K * ((ri - gm) ** 2)).sum()); SSB = float((M * ((cj - gm) ** 2)).sum())
    R = Y - ri.unsqueeze(1) - cj.unsqueeze(0) + gm
    vAB = float((R * R).sum()) / ((M - 1) * (K - 1))
    vA = (SSA / (M - 1) - vAB) / K; vB = (SSB / (K - 1) - vAB) / M
    tot = vA + vB + vAB
    return (vA + vB) / tot

# ---------- 4 блоки: незміщений одноfactorний random-effects на кожен блок ----------
def anova4(f, seed, NB=4, m=40, NS=20000, CH=4000):
    g = torch.Generator(device=dev).manual_seed(seed)
    SZ = D // NB
    H = [torch.randn(m, SZ, device=dev, generator=g) for _ in range(NB)]
    idx = torch.randint(0, m, (NS, NB), device=dev, generator=g)
    ys = []
    for s in range(0, NS, CH):
        x = torch.cat([H[b][idx[s:s + CH, b]] for b in range(NB)], 1)
        ys.append(f(x).double())
    y = torch.cat(ys, 0)
    vtot = float(y.var(0, unbiased=True).sum())
    ybar = y.mean(0)
    main = 0.0
    for b in range(NB):
        acc = torch.zeros(m, y.shape[1], dtype=torch.float64, device=dev)
        cn = torch.zeros(m, dtype=torch.float64, device=dev)
        acc.index_add_(0, idx[:, b], y); cn.index_add_(0, idx[:, b], torch.ones(NS, dtype=torch.float64, device=dev))
        mb = acc / cn.unsqueeze(1)
        SSb = float((cn.unsqueeze(1) * (mb - ybar) ** 2).sum())          # між групами
        SSw = float(((y - mb[idx[:, b]]) ** 2).sum())                    # усередині груп
        MSb = SSb / (m - 1); MSw = SSw / (NS - m)
        nbar = NS / m
        main += max(0.0, (MSb - MSw) / nbar)                             # незміщена сигма^2_b
    return main / vtot

# ================================================================= прогін
log('=' * 86)
log('{:34s} {:>14s} {:>14s}'.format('функція', 'головні 2 бл.', 'головні 4 бл.'))
log('-' * 86)
CASES = [('НУЛЬ: чистий степінь 2 (z^T A z)', mk_quad(11), 'очік 0.500 / 0.250'),
         ('НУЛЬ: чистий степінь 4 (He4)',     mk_quart(13), 'очік 0.125 / 0.016')]
for name, f, exp in CASES:
    a2 = anova2(f, 770001); a4 = anova4(f, 880001)
    log('{:34s} {:14.4f} {:14.4f}   {}'.format(name, a2, a4, exp))
log('-' * 86)
for net in (0, 1):
    f = mk_even(load(net))
    a2 = anova2(f, 770000 + net); a4 = anova4(f, 880000 + net)
    log('{:34s} {:14.4f} {:14.4f}'.format('МЕРЕЖА %d (парна частина)' % net, a2, a4))
log('-' * 86)
log('splice.py звітував: головні 2 бл. 0.302/0.315 ; 4 бл. 0.227/0.239 (4 бл. БЕЗ корекції)')
log('done.')
