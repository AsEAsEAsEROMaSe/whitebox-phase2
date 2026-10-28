"""E-A++ (18.10.2026, сесія a2): спільний фіт таблиць усіх шарів НА ПОХИБКУ ВИХОДУ.
Нині (fit87c) кожен шар фітиться окремо на свій «свіжий крок» s_l, тобто мінімізує Σ_l ‖s_l − X_l β_l‖². Але бал бачить
лише вихід: e = y_15 − Σ_l (X_l β_l)·J_l, де J_l = перенос шару l до виходу лінійним відгуком ((v @ W_m)·Φ_m, m = l+1..15).
Перенос підсилює одні напрямки і гасить інші ⟹ пошаровий фіт витрачає ступені свободи на напрямки, що не доходять.
Спільна задача лінійна по всіх β: T = [X_0 J_0 | … | X_15 J_15] (1024 × 16·87 на мережу), мінімізуємо
‖y_15 − T β‖² + λ‖β − β_0‖², β_0 = пошарове рішення (пріор: якщо спільний фіт не допомагає, лишаємось на β_0).
Крос: ті самі 5 блоків по мережах, що в fit87c; β_0 теж з тренувальних блоків; λ = κ·tr(G)/p.
🔒 Пломба: κ → ∞ дає рівно пошарове рішення ⟹ вихід ×1.0847 (48/50) як у fit87c.
ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (18.10, a2): найкраще κ дає вихід ×1.10–×1.15 проти ×1.085 пошарового (шанс ≥×1.10: ~40%);
без пріору (β_0 = 0) перефіт: гірше за пріор на малих κ.
RAM ~0.2 ГБ (ознаки на льоту, ваги mmap по шару з /home/trim/wbcache). Запуск: /home/trim/fs123/bin/python3 -I fitj.py"""
import glob, json, math, os, sys, time
import numpy as np

NNET = 50
CDIR = '/home/trim/wbcache'
DDIR = '/home/trim/wblogs/corrdump'
OUT = os.environ.get('FIT_OUT', '/home/trim/wb2lean/v39fija/fitj_tables.json')
_erf = np.vectorize(math.erf)
if os.path.exists(CDIR + '/truth.npz'):
    truth = dict(np.load(CDIR + '/truth.npz'))
else:   # легке читання лише колонки істини (без ваг)
    import pyarrow.parquet as pq
    P_ = ('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/'
          'aa99830fdc09fad15407b10e8e3459d3e18bba0a/data')
    truth = {}
    for f in sorted(glob.glob(P_ + '/mini-*.parquet')):
        t = pq.read_table(f, columns=['mlp_name', 'all_layer_means'])
        for nm, alm in zip(t.column('mlp_name').to_pylist(), t.column('all_layer_means').to_pylist()):
            truth[nm] = np.array(alm, dtype=np.float64)
allnames = list(truth)
files = sorted(glob.glob(DDIR + '/net_*.npz'), key=lambda f: int(f.split('_')[-2]))[-NNET:]
dump, names = {}, []
for f in files:
    d = dict(np.load(f))
    p0 = d['pk1'][0].astype(np.float64)
    errs = sorted((np.sqrt(((p0 - truth[nm][0]) ** 2).mean()), nm) for nm in allnames)
    assert errs[0][0] < 1e-3 and errs[1][0] > 20 * errs[0][0]
    dump[errs[0][1]] = d; names.append(errs[0][1])
assert names == json.load(open('/home/trim/wb2lean/v39fija/fit87c_tables.json'))['step_F87']['nets']
names = names[:int(os.environ.get('FIT_NUSE', len(names)))]
L, NF, N = 16, 5, 1024
fold = {nm: i % NF for i, nm in enumerate(names)}
class _FakeW:   # тест логіки без кешу ваг
    def __init__(self, nm):
        self.seed = sum(map(ord, nm))
    def __getitem__(self, l):
        return np.random.default_rng(self.seed * 100 + l).normal(0, math.sqrt(2 / 1024), (1024, 1024)).astype(np.float32)


loadW = (lambda nm: _FakeW(nm)) if os.environ.get('FIT_FAKE_W') else (lambda nm: np.load('%s/W/%s.npy' % (CDIR, nm), mmap_mode='r'))


def herm(a, k):
    h0, h1 = np.ones_like(a), a
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, a * h1 - j * h0
    return h1


def feats87(d, l):
    g = lambda k: d[k][l].astype(np.float64)
    mu, var = g('mu'), g('var')
    s = np.sqrt(var); a = mu / s
    phi = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + _erf(a / math.sqrt(2)))
    one = np.ones_like(a)
    B = {'1': one, 'mu': mu, 'var': var, 'sigma': s, 'alpha': a, '|alpha|': np.abs(a), 'phi': phi, 'Phi': Phi,
         'pred': g('pk1'), 'D3': g('d3'), 'D21n': g('d21n'), 'K4': one * float(d['k4s'][l]), 'sigma*phi': s * phi,
         'pk2': g('pk2'), 'pk3': g('pk3'), 'pk4': g('pk4'), 'g4': g('g4')}
    iv = 1.0 / var
    M = {'s3': B['D3'] * iv / s, 's4': B['g4'] * iv * iv, 'dn': B['D21n'] * iv, 'v': var,
         'ds': g('d21s') * iv, 'cr': g('crs') * iv, 'c2': g('cr2') * iv * iv, 'k4': B['K4']}
    M['s33'] = M['s3'] ** 2
    cols = [B[x] for x in ['1', 'mu', 'var', 'sigma', 'alpha', '|alpha|', 'phi', 'Phi', 'pred', 'D3', 'D21n', 'K4',
                           'sigma*phi', 'pk2', 'pk3', 'pk4', 'g4']]
    for k in range(7):
        h = s * phi * herm(a, k)
        cols.append(h)
        cols += [h * M[m] for m in ['s3', 's4', 's33', 'dn', 'v', 'ds', 'cr', 'c2', 'k4']]
    return np.stack(cols, 1), Phi


P = 87
LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
t0 = time.time()
PHI, Y, STEP = {}, {}, {}
MU, SC = np.zeros((L, P)), np.ones((L, P))
for l in range(L):                                # стандартизація по шару (ознаки рахуються на льоту, RAM мала)
    Xl = np.concatenate([feats87(dump[nm], l)[0] for nm in names])
    MU[l], SC[l] = Xl.mean(0), Xl.std(0)
    del Xl
SC[SC < 1e-12] = 1.0; MU[:, 0] = 0.0; SC[:, 0] = 1.0
for nm in names:
    d = dump[nm]
    PHI[nm] = np.stack([feats87(d, l)[1] for l in range(L)])
    Y[nm] = np.stack([truth[nm][l] - d['pk1'][l].astype(np.float64) for l in range(L)])
zf = lambda nm, l: (feats87(dump[nm], l)[0] - MU[l]) / SC[l]


def transported(nm, need_step=False):
    """T (N, L·P): ознаки всіх шарів, перенесені до виходу; за потреби свіжий крок"""
    W = loadW(nm)
    st = Y[nm].copy()
    V = np.zeros((L * P, N))
    V[0:P] = zf(nm, 0).T
    for m in range(1, L):
        Wm = np.asarray(W[m], dtype=np.float64)
        if need_step:
            st[m] = Y[nm][m] - (Y[nm][m - 1] @ Wm) * PHI[nm][m]
        V[:m * P] = (V[:m * P] @ Wm) * PHI[nm][m][None]
        V[m * P:(m + 1) * P] = zf(nm, m).T
    return V.T, st

# прохід 1: свіжий крок і перенесені ознаки T (усі шари однією стопкою), нормальні рівняння по блоках
G = np.zeros((NF, L * P, L * P)); Bv = np.zeros((NF, L * P)); YY = np.zeros(NF)
for nm in names:
    T, STEP[nm] = transported(nm, need_step=True)
    b = fold[nm]
    G[b] += T.T @ T; Bv[b] += T.T @ Y[nm][L - 1]; YY[b] += Y[nm][L - 1] @ Y[nm][L - 1]
print('мереж %d, прохід 1: %.0f с' % (len(names), time.time() - t0), flush=True)


def fit_layer(Zs, Ys):
    nb = 5; idx = np.arange(len(Zs)) % nb
    Gl = [(z.T @ z, z.T @ y) for z, y in zip(Zs, Ys)]
    best = None
    for lam in LAMS:
        sse = 0.0
        for bb in range(nb):
            A = np.sum([Gl[i][0] for i in range(len(Zs)) if idx[i] != bb], axis=0)
            yv = np.sum([Gl[i][1] for i in range(len(Zs)) if idx[i] != bb], axis=0)
            R = lam * np.eye(P); R[0, 0] = 0
            w = np.linalg.solve(A + R, yv)
            sse += sum(((Ys[i] - Zs[i] @ w) ** 2).sum() for i in range(len(Zs)) if idx[i] == bb)
        if best is None or sse < best[0]:
            best = (sse, lam)
    A = np.sum([g[0] for g in Gl], axis=0); yv = np.sum([g[1] for g in Gl], axis=0)
    R = best[1] * np.eye(P); R[0, 0] = 0
    return np.linalg.solve(A + R, yv)


KAPS = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 10.0]
res = {}
t0 = time.time()
for b in range(NF):
    trn = [nm for nm in names if fold[nm] != b]
    tst = [nm for nm in names if fold[nm] == b]
    beta0 = np.concatenate([fit_layer([zf(nm, l) for nm in trn], [STEP[nm][l] for nm in trn]) for l in range(L)])
    Gt = G.sum(0) - G[b]; bt = Bv.sum(0) - Bv[b]
    sc = np.trace(Gt) / (L * P)
    sols = {('perlayer', 0): beta0}
    for kap in KAPS:
        A = Gt + kap * sc * np.eye(L * P)
        sols[('prior', kap)] = np.linalg.solve(A, bt + kap * sc * beta0)
        sols[('zero', kap)] = np.linalg.solve(A, bt)
    for nm in tst:                                # прохід 2 для блоку: перенесені ознаки відкладених мереж
        T = transported(nm)[0]
        for key, beta in sols.items():
            e = Y[nm][L - 1] - T @ beta
            res.setdefault(key, []).append(((e ** 2).mean(), (Y[nm][L - 1] ** 2).mean(), nm))
print('крос: %.0f с' % (time.time() - t0))
print('\nемуляція онлайн (лінійний відгук), вихід на відкладених блоках:')
base = {r[2]: r[0] for r in res[('perlayer', 0)]}
for key in sorted(res, key=lambda k: (k[0], k[1])):
    v = res[key]
    m1 = np.mean([r[0] for r in v]); m0 = np.mean([r[1] for r in v])
    better = sum(r[0] < r[1] for r in v); bpl = sum(r[0] < base[r[2]] for r in v)
    print('%-8s κ=%-7g вихід ×%.4f (кращих за базу %2d/%d, за пошаровий %2d/%d)' % (key[0], key[1], m0 / m1, better, len(v), bpl, len(v)))

# ---- підсумкові таблиці на всіх мережах: найкращий κ з кросу (пріор), формат fit87c_tables.json ----
F13 = ['1', 'mu', 'var', 'sigma', 'alpha', '|alpha|', 'phi', 'Phi', 'pred', 'D3', 'D21n', 'K4', 'sigma*phi']
MODS = ['s3', 's4', 's33', 'dn', 'v', 'ds', 'cr', 'c2', 'k4']
F87 = F13 + ['pk2', 'pk3', 'pk4', 'g4'] + [x for k in range(7) for x in ['h%d' % k] + ['h%d*%s' % (k, m) for m in MODS]]
gain = {k: np.mean([r[1] for r in v]) / np.mean([r[0] for r in v]) for k, v in res.items()}
best = max([k for k in gain if k[0] == 'prior'], key=lambda k: gain[k])
print('\nнайкращий: %s κ=%g (крос ×%.4f; пошаровий ×%.4f)' % (best[0], best[1], gain[best], gain[('perlayer', 0)]))
beta0 = np.concatenate([fit_layer([zf(nm, l) for nm in names], [STEP[nm][l] for nm in names]) for l in range(L)])
Gt, bt = G.sum(0), Bv.sum(0)
sc = np.trace(Gt) / (L * P)
out = {}
for key, kap in (('joint_F87', best[1]), ('perlayer_F87', None)):
    beta = beta0 if kap is None else np.linalg.solve(Gt + kap * sc * np.eye(L * P), bt + kap * sc * beta0)
    out[key] = {'feats': F87, 'beta': [beta[l * P:(l + 1) * P].tolist() for l in range(L)], 'mean': MU.tolist(),
                'scale': SC.tolist(), 'lam': [kap or 0.0] * L, 'nets': names, 'cv_gain': gain[best if kap else ('perlayer', 0)]}
json.dump(out, open(OUT, 'w'))
print('таблиці: %s (joint_F87, perlayer_F87)' % OUT)
