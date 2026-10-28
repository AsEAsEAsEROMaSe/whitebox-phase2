"""E-A+ (18.10.2026, сесія a2): чи додають ознаки з ВАГ до 87 ознак e30_a у корекції середнього (ціль «свіжий крок»).
Ідея: похибка гаусового замикання нейрона j залежить від того, скільки сильних входів його живлять (ЦГТ), а 87 ознак
бачать лише моменти ланцюга. Ознаки з вхідного стовпця w = W[l][:, j] (p = h @ W[l]) і гаусових моментів входів
h_i = ReLU(p_i шару l−1) (q = var, κ3, κ4 випрямленого гауса з mu/var дампу; шар 0: q=1, κ=0):
  i3 = Σ w³κ3 / var^1.5 · i4 = Σ w⁴κ4 / var² · ip = Σ w⁴q² / var² (обернена участь) · iv = Σ w²q / var
набір F87W = F87 + h_k·{i3,i4,ip,iv}, k=0..6 (+28). Ваги з кешу /home/trim/wbcache (wextract.py), читаються mmap
по шару ⟹ RAM < 0.5 ГБ, можна поруч із whest.
Оцінка як у fit87c: 5-кратний крос по мережах (ті самі блоки), (1) виграш шару в один крок, (3) емуляція онлайн.
🔒 Пломба: F87, ціль step, c=1.0, усі шари = fit87c (вихід ×1.0847, 48/50).
ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (18.10, a2): F87W проти F87 в емуляції ×1.00–×1.03 (шанс ≥×1.02: ~25%); на шарі 1–2 приріст
більший (там i3/i4 точніші за ланцюгові), на глибині ~0 (кореляції входів домінують, ознаки незалежних входів слабкі).
Запуск: /home/trim/fs123/bin/python3 -I fitx.py (env FIT_NUSE=k: лише перші k мереж, крива навчання); таблиці → /home/trim/wb2lean/v39fija/fitx_tables.json"""
import glob, json, math, os, sys, time
import numpy as np

NNET = 50
CDIR = '/home/trim/wbcache'
OUT = os.environ.get('FIT_OUT', '/home/trim/wb2lean/v39fija/fitx_tables.json')
DDIR = os.environ.get('FIT_DUMP_DIR', '/home/trim/wblogs/corrdump')
_erf = np.vectorize(math.erf)
if os.path.exists(CDIR + '/truth.npz'):
    truth = dict(np.load(CDIR + '/truth.npz'))
else:   # легке читання лише колонки істини (без ваг)
    import pyarrow.parquet as pq
    P = ('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/'
         'aa99830fdc09fad15407b10e8e3459d3e18bba0a/data')
    truth = {}
    for f in sorted(glob.glob(P + '/mini-*.parquet')):
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
    (e1, nm1), (e2, nm2) = errs[0], errs[1]
    assert e1 < 1e-3 and e2 > 20 * e1, (f, e1, nm1, e2, nm2)
    assert nm1 not in dump
    dump[nm1] = d; names.append(nm1)
assert names == json.load(open('/home/trim/wb2lean/v39fija/fit87c_tables.json'))['step_F87']['nets'], 'порядок мереж ≠ fit87c'
names = names[:int(os.environ.get('FIT_NUSE', len(names)))]   # крива навчання: перші k мереж
L = dump[names[0]]['pk1'].shape[0]
NF = 5
fold = {nm: i % NF for i, nm in enumerate(names)}


class _FakeW:   # тест логіки без кешу ваг: шар за шаром, 4 МБ
    def __init__(self, nm):
        self.seed = sum(map(ord, nm))
    def __getitem__(self, l):
        return np.random.default_rng(self.seed * 100 + l).normal(0, math.sqrt(2 / 1024), (1024, 1024)).astype(np.float32)


def loadW(nm):
    if os.environ.get('FIT_FAKE_W'):
        return _FakeW(nm)
    return np.load('%s/W/%s.npy' % (CDIR, nm), mmap_mode='r')


def herm(a, k):
    h0, h1 = np.ones_like(a), a
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, a * h1 - j * h0
    return h1


def gstats(mu, var):
    """α, φ, Φ і q, κ3, κ4 випрямленого гауса ReLU(N(mu, var))"""
    s = np.sqrt(var); a = mu / s
    phi = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + _erf(a / math.sqrt(2)))
    M1 = a * Phi + phi
    M2 = (a * a + 1) * Phi + a * phi
    M3 = (a ** 3 + 3 * a) * Phi + (a * a + 2) * phi
    M4 = (a ** 4 + 6 * a * a + 3) * Phi + (a ** 3 + 5 * a) * phi
    c2 = M2 - M1 ** 2
    c3 = M3 - 3 * M1 * M2 + 2 * M1 ** 3
    c4 = M4 - 4 * M1 * M3 + 6 * M1 ** 2 * M2 - 3 * M1 ** 4 - 3 * c2 ** 2
    return a, phi, Phi, c2 * var, c3 * var ** 1.5, c4 * var * var


def base(d, l, wf):
    g = lambda k: d[k][l].astype(np.float64)
    mu, var = g('mu'), g('var')
    s = np.sqrt(var); a, phi, Phi = gstats(mu, var)[:3]
    one = np.ones_like(a)
    B = {'1': one, 'mu': mu, 'var': var, 'sigma': s, 'alpha': a, '|alpha|': np.abs(a), 'phi': phi, 'Phi': Phi,
         'pred': g('pk1'), 'D3': g('d3'), 'D21n': g('d21n'), 'K4': one * float(d['k4s'][l]), 'sigma*phi': s * phi,
         'pk2': g('pk2'), 'pk3': g('pk3'), 'pk4': g('pk4'), 'g4': g('g4'), 'd21s': g('d21s')}
    iv = 1.0 / var
    M = {'s3': B['D3'] * iv / s, 's4': B['g4'] * iv * iv, 'dn': B['D21n'] * iv, 'v': var,
         'ds': B['d21s'] * iv, 'cr': g('crs') * iv, 'c2': g('cr2') * iv * iv, 'k4': B['K4'],
         'i3': wf[:, 0] * iv / s, 'i4': wf[:, 1] * iv * iv, 'ip': wf[:, 2] * iv * iv, 'iv': wf[:, 3] * iv}
    M['s33'] = M['s3'] ** 2
    for k in range(7):
        B['h%d' % k] = s * phi * herm(a, k)
    return B, M, Phi


def feats(nm, l, names_):
    B, M, _ = base(dump[nm], l, WF[nm][l])
    cols = []
    for x in names_:
        if x in B:
            cols.append(B[x])
        else:
            hb, _, md = x.partition('*')
            cols.append(B[hb] * M[md])
    return np.stack(cols, 1)


F13 = ['1', 'mu', 'var', 'sigma', 'alpha', '|alpha|', 'phi', 'Phi', 'pred', 'D3', 'D21n', 'K4', 'sigma*phi']
MODS = ['s3', 's4', 's33', 'dn', 'v', 'ds', 'cr', 'c2', 'k4']
F87 = F13 + ['pk2', 'pk3', 'pk4', 'g4'] + [x for k in range(7) for x in ['h%d' % k] + ['h%d*%s' % (k, m) for m in MODS]]
WMODS = ['i3', 'i4', 'ip', 'iv']
F87W = F87 + ['h%d*%s' % (k, m) for k in range(7) for m in WMODS]
assert len(F87) == 87 and len(F87W) == 115
SETS = [('F87', F87), ('F87W', F87W)]
LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]

# ---- прохід 1: свіжий крок і ознаки з ваг (mmap по шару) ----
t0 = time.time()
Y, PHI, STEP, WF = {}, {}, {}, {}
for nm in names:
    d = dump[nm]; W = loadW(nm)
    Y[nm] = np.stack([truth[nm][l] - d['pk1'][l].astype(np.float64) for l in range(L)])
    PHI[nm] = np.stack([gstats(d['mu'][l].astype(np.float64), d['var'][l].astype(np.float64))[2] for l in range(L)])
    st = Y[nm].copy(); wf = np.zeros((L, 1024, 4))
    for l in range(L):
        Wl = np.asarray(W[l], dtype=np.float64)
        if l > 0:
            st[l] = Y[nm][l] - (Y[nm][l - 1] @ Wl) * PHI[nm][l]
            _, _, _, q, k3, k4 = gstats(d['mu'][l - 1].astype(np.float64), d['var'][l - 1].astype(np.float64))
        else:
            q, k3, k4 = np.ones(1024), np.zeros(1024), np.zeros(1024)
        W2 = Wl * Wl
        wf[l, :, 0] = k3 @ (W2 * Wl); wf[l, :, 1] = k4 @ (W2 * W2); wf[l, :, 2] = (q * q) @ (W2 * W2); wf[l, :, 3] = q @ W2
    STEP[nm], WF[nm] = st, wf
print('мереж %d, прохід 1: %.0f с' % (len(names), time.time() - t0), flush=True)
for l in (0, 1, 4, 8, 12, 15):
    r = np.concatenate([WF[nm][l, :, 3] / dump[nm]['var'][l] for nm in names])
    p = np.concatenate([WF[nm][l, :, 2] / WF[nm][l, :, 3] ** 2 for nm in names])
    print('шар %2d: Σw²q/var сер %.3f (sd %.3f) · 1/участь сер %.4f (=1/%.0f)' % (l, r.mean(), r.std(), p.mean(), 1 / p.mean()))


def fit_layer(Xs, Ys):
    Xa = np.concatenate(Xs)
    mu_, sc_ = Xa.mean(0), Xa.std(0); sc_[sc_ < 1e-12] = 1.0; mu_[0] = 0.0; sc_[0] = 1.0
    Z = [(x - mu_) / sc_ for x in Xs]
    nb = 5; idx = np.arange(len(Xs)) % nb
    G = [(z.T @ z, z.T @ y) for z, y in zip(Z, Ys)]
    best = None
    for lam in LAMS:
        sse = 0.0
        for b in range(nb):
            A = np.sum([G[i][0] for i in range(len(Xs)) if idx[i] != b], axis=0)
            yv = np.sum([G[i][1] for i in range(len(Xs)) if idx[i] != b], axis=0)
            R = lam * np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A + R, yv)
            for i in range(len(Xs)):
                if idx[i] == b:
                    sse += ((Ys[i] - Z[i] @ w) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, lam)
    lam = best[1]
    A = np.sum([g[0] for g in G], axis=0); yv = np.sum([g[1] for g in G], axis=0)
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    return np.linalg.solve(A + R, yv), mu_, sc_, lam


t0 = time.time()
TABF, DV = {}, {}
print('\n(1) виграш MSE свіжого кроку шару на відкладених блоках (SST/SSE):\nшар    F87    F87W', flush=True)
for l in range(L):
    row = []
    for s, FN in SETS:
        X = {nm: feats(nm, l, FN) for nm in names}
        e0 = e1 = 0.0
        for b in range(NF):
            trn = [nm for nm in names if fold[nm] != b]
            tab = fit_layer([X[nm] for nm in trn], [STEP[nm][l] for nm in trn])
            TABF[(s, b, l)] = tab
            for nm in names:
                if fold[nm] == b:
                    dv = ((X[nm] - tab[1]) / tab[2]) @ tab[0]
                    DV[(s, nm, l)] = dv
                    e0 += (STEP[nm][l] ** 2).sum(); e1 += ((STEP[nm][l] - dv) ** 2).sum()
        row.append(e0 / e1)
    print('%3d  %.4f  %.4f' % (l, row[0], row[1]), flush=True)
print('фіт кросу: %.0f с' % (time.time() - t0))

print('\n(3) емуляція онлайн-корекції (усі шари, лінійний відгук), відкладені блоки:')
CS = (0.75, 1.0, 1.25)
VAR = [(s, c) for s, _ in SETS for c in CS]
res = {k: [] for k in VAR}
t0 = time.time()
for nm in names:
    W = loadW(nm)
    DD = np.stack([np.stack([c * DV[(s, nm, l)] for l in range(L)]) for (s, c) in VAR])
    acc = DD[:, 0].copy()
    for l in range(1, L):
        acc = (acc @ np.asarray(W[l], dtype=np.float64)) * PHI[nm][l][None] + DD[:, l]
    fin = ((Y[nm][L - 1][None] - acc) ** 2).mean(1)
    for i, key in enumerate(VAR):
        res[key].append((fin[i], (Y[nm][L - 1] ** 2).mean()))
print('прохід 2: %.0f с' % (time.time() - t0))
for key, v in res.items():
    v = np.array(v)
    print('%-5s c=%.2f: вихід ×%.4f (кращих %2d/%d)' % (key[0], key[1], v[:, 1].mean() / v[:, 0].mean(), int((v[:, 0] < v[:, 1]).sum()), len(v)))
a = np.array(res[('F87W', 1.0)]); b = np.array(res[('F87', 1.0)])
print('F87W проти F87 (c=1) помережно: кращих %d/%d, геом. ×%.4f' % (int((a[:, 0] < b[:, 0]).sum()), len(a), math.exp(np.mean(np.log(b[:, 0] / a[:, 0])))))

out = {}
for s, FN in SETS:
    tab = [fit_layer([feats(nm, l, FN) for nm in names], [STEP[nm][l] for nm in names]) for l in range(L)]
    out['step_' + s] = {'feats': FN, 'beta': [t[0].tolist() for t in tab], 'mean': [t[1].tolist() for t in tab],
                        'scale': [t[2].tolist() for t in tab], 'lam': [t[3] for t in tab], 'nets': names}
json.dump(out, open(OUT, 'w'))
print('\nтаблиці (фіт на всіх %d): %s' % (len(names), OUT))
