"""E-A дофіт (18.10.2026, white-box-21): таблиця step_F87 на 90 мережах (mini 10..59 + нові 60..99) проти 50.
ПАРНИЙ протокол: 5 блоків по всіх 90 мережах; для блоку b дві таблиці: «old» (навчена лише на старих 50 поза b)
і «all» (усі 90 поза b); обидві оцінюються на ТИХ САМИХ відкладених мережах блоку b (емуляція онлайн, лінійний відгук).
Окремо на нових 40 (жодна «old»-таблиця їх не бачила). Підсумкові таблиці на всіх 90 → fit90_tables.json (step_F87, step_F87W).
ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ: крива навчання fitx (25/35/50) покаже нахил; очікую «all» проти «old» ×1.003–1.015 на виході,
шанс ≥ ×1.01 ~35%. Якщо ≥ ×1.005 і ≥ 60% мереж краще ⟹ v44 = v43 з таблицею fit90, перевірка ланцюгом на s0..s9.
Дані: дампи /home/trim/wblogs/corrdump (a5, cd_m10..59) і corrdump2 (white-box-21, cd_m60..99); ваги /home/trim/wbcache/W (wextract WX_ALL=1).
КРИВА НАВЧАННЯ (fitx FIT_NUSE, F87 c=1, крос усередині k): фіт на 20 мережах ×1.0729 · 28 ×1.0784 · 40 ×1.0847 ⟹ g∞−a/n:
  g∞≈0.0965, a≈0.47 ⟹ таблиця на 90 проти 50 ≈ ×1.004 (уточнене передбачення, записане до прогону 18.10 21:15).
Запуск: /home/trim/fs123/bin/python3 -I fit90.py (env FIT_NOLD=k лише k старих, FIT_FAKE_W=1 тест без кешу ваг, FIT_OUT)"""
import glob, json, math, os, sys, time
import numpy as np

CDIR = '/home/trim/wbcache'
OUT = os.environ.get('FIT_OUT', '/home/trim/wb2lean/v39fija/fit90_tables.json')
DDIRS = ['/home/trim/wblogs/corrdump', '/home/trim/wblogs/corrdump2']
_erf = np.vectorize(math.erf)
truth = dict(np.load(CDIR + '/truth.npz'))
allnames = list(truth)


def bind(files):
    out = []
    for f in files:
        d = dict(np.load(f))
        p0 = d['pk1'][0].astype(np.float64)
        errs = sorted((np.sqrt(((p0 - truth[nm][0]) ** 2).mean()), nm) for nm in allnames)
        (e1, nm1), (e2, nm2) = errs[0], errs[1]
        assert e1 < 1e-3 and e2 > 20 * e1, (f, e1, nm1, e2, nm2)
        out.append((nm1, d))
    return out


key = lambda f: int(f.split('_')[-2])
old = bind(sorted(glob.glob(DDIRS[0] + '/net_*.npz'), key=key)[-50:])
assert [nm for nm, _ in old] == json.load(open('/home/trim/wb2lean/v39fija/fit87c_tables.json'))['step_F87']['nets'], 'порядок ≠ fit87c'
old = old[:int(os.environ.get('FIT_NOLD', 50))]
new = bind(sorted(glob.glob(DDIRS[1] + '/net_*.npz'), key=key))
dump = dict(old + new)
assert len(dump) == len(old) + len(new), 'мережа двічі'
names = [nm for nm, _ in old] + [nm for nm, _ in new]
ISNEW = {nm: i >= len(old) for i, nm in enumerate(names)}
print('мереж: старих %d, нових %d' % (len(old), len(new)), flush=True)
L = dump[names[0]]['pk1'].shape[0]
NF = 5
fold = {nm: i % NF for i, nm in enumerate(names)}   # старі мають ті самі блоки, що у fitx


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
DV = {}
TV = ('old', 'all')
print('\n(1) виграш MSE свіжого кроку шару на відкладених (SST/SSE), F87: old · all', flush=True)
for l in range(L):
    row = {}
    for s, FN in SETS:
        X = {nm: feats(nm, l, FN) for nm in names}
        e = {v: [0.0, 0.0] for v in TV}
        for b in range(NF):
            for v in TV:
                trn = [nm for nm in names if fold[nm] != b and (v == 'all' or not ISNEW[nm])]
                tab = fit_layer([X[nm] for nm in trn], [STEP[nm][l] for nm in trn])
                for nm in names:
                    if fold[nm] == b:
                        dv = ((X[nm] - tab[1]) / tab[2]) @ tab[0]
                        DV[(s, v, nm, l)] = dv
                        e[v][0] += (STEP[nm][l] ** 2).sum(); e[v][1] += ((STEP[nm][l] - dv) ** 2).sum()
        row[s] = [e[v][0] / e[v][1] for v in TV]
    print('%3d  F87 %.4f %.4f · F87W %.4f %.4f' % (l, *row['F87'], *row['F87W']), flush=True)
print('фіт кросу: %.0f с' % (time.time() - t0))

print('\n(3) емуляція онлайн-корекції (усі шари, лінійний відгук), відкладені блоки:')
VAR = [(s, v, c) for s, _ in SETS for v in TV for c in (1.0, 1.25)]
res = {k_: [] for k_ in VAR}
t0 = time.time()
for nm in names:
    W = loadW(nm)
    DD = np.stack([np.stack([c * DV[(s, v, nm, l)] for l in range(L)]) for (s, v, c) in VAR])
    acc = DD[:, 0].copy()
    for l in range(1, L):
        acc = (acc @ np.asarray(W[l], dtype=np.float64)) * PHI[nm][l][None] + DD[:, l]
    fin = ((Y[nm][L - 1][None] - acc) ** 2).mean(1)
    for i, k_ in enumerate(VAR):
        res[k_].append((fin[i], (Y[nm][L - 1] ** 2).mean()))
print('прохід 2: %.0f с' % (time.time() - t0))
sel = {'усі': [True] * len(names), 'старі': [not ISNEW[nm] for nm in names], 'нові': [ISNEW[nm] for nm in names]}
for k_, v in res.items():
    v = np.array(v)
    print('%-5s %-3s c=%.2f: ' % k_ + ' · '.join('%s ×%.4f (%d/%d)' % (g, v[m, 1].mean() / v[m, 0].mean(), int((v[m, 0] < v[m, 1]).sum()), int(np.sum(m)))
                                               for g, m in ((g, np.array(m_)) for g, m_ in sel.items()) if np.sum(m) > 0))
for s, _ in SETS:
    for c in (1.0, 1.25):
        a = np.array(res[(s, 'all', c)]); b = np.array(res[(s, 'old', 1.0)])
        for g, m_ in sel.items():
            m = np.array(m_)
            if np.sum(m) == 0:
                continue
            print('%-5s all c=%.2f проти old c=1 [%s]: агрегат ×%.4f · кращих %d/%d · геом. ×%.4f' % (
                s, c, g, b[m, 0].mean() / a[m, 0].mean(), int((a[m, 0] < b[m, 0]).sum()), int(np.sum(m)),
                math.exp(np.mean(np.log(b[m, 0] / a[m, 0])))))

out = {}
for s, FN in SETS:
    tab = [fit_layer([feats(nm, l, FN) for nm in names], [STEP[nm][l] for nm in names]) for l in range(L)]
    out['step_' + s] = {'feats': FN, 'beta': [t[0].tolist() for t in tab], 'mean': [t[1].tolist() for t in tab],
                        'scale': [t[2].tolist() for t in tab], 'lam': [t[3] for t in tab], 'nets': names}
json.dump(out, open(OUT, 'w'))
print('\nтаблиці (фіт на всіх %d): %s' % (len(names), OUT))
