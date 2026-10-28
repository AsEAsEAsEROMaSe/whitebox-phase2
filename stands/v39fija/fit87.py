"""E-A (18.10.2026, сесія a2): ridge-корекція середнього, набори ознак 2/13/20/59/87 (87 = як у e30_a).
Дані: дамп v39 (вільний прогін, корекція вимкнена) на мережах mini 0..59; 0..9 = s0..s9 відкладені, фіт на 10..59.
Ціль δ_l = truth_l − pk1_l понейронно (truth = all_layer_means з parquet).
Ridge по шару: стандартизовані колонки, вільний перетин, λ крос-валідацією 5 блоків по мережах (як e30_a).
Друкує: крос по шарах (SST/SSE = виграш MSE шару в один крок), відкладені s0..s9: лише вихідний шар
(вихідний коректор, без перенесення, переноситься в бал прямо), скан множника c, контроль «чужа мережа».
Пишемо таблиці у fit87_tables.json.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (18.10, a2): крос R² на шарі 15: 13 ознак 0.02–0.05, 87 ознак 0.03–0.10;
відкладені, лише вихід: ×1.02–×1.08 (87); контроль «чужа» < ×1.0 окрім внеску перетину. Шанс, що 87 дасть
≥ ×1.05 на виході: ~40%.
Запуск: /home/trim/fs123/bin/python3 fit87.py <dump_dir> <run_json>
"""
import glob, json, math, sys
import numpy as np
import pyarrow.parquet as pq

DUMP, RUNJ = sys.argv[1], sys.argv[2]
P = ('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/'
     'aa99830fdc09fad15407b10e8e3459d3e18bba0a/data')
files = sorted(glob.glob(DUMP + '/net_*.npz'), key=lambda f: int(f.split('_')[-2]))
names = [m['mlp_name'] for m in json.load(open(RUNJ))['results']['per_mlp']]
assert len(files) >= len(names), (len(files), len(names))
files = files[-len(names):]   # старі файли обірваних прогонів відкидаються (беремо найновіші за time_ns)
truth, finals = {}, {}
for f in sorted(glob.glob(P + '/mini-*.parquet')):
    t = pq.read_table(f, columns=['mlp_name', 'all_layer_means', 'final_means'])
    for nm, alm, fm in zip(t.column('mlp_name').to_pylist(), t.column('all_layer_means').to_pylist(),
                           t.column('final_means').to_pylist()):
        truth[nm] = np.array(alm, dtype=np.float64)
        finals[nm] = np.array(fm, dtype=np.float64)
hold = set(names[:10])
_erf = np.vectorize(math.erf)


def herm(a, k):
    h0, h1 = np.ones_like(a), a
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, a * h1 - j * h0
    return h1


def base(d, l):
    g = lambda k: d[k][l].astype(np.float64)
    mu, var = g('mu'), g('var')
    s = np.sqrt(var); a = mu / s
    phi = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + _erf(a / math.sqrt(2)))
    one = np.ones_like(a)
    B = {'1': one, 'mu': mu, 'var': var, 'sigma': s, 'alpha': a, '|alpha|': np.abs(a), 'phi': phi, 'Phi': Phi,
         'pred': g('pk1'), 'D3': g('d3'), 'D21n': g('d21n'), 'K4': one * float(d['k4s'][l]), 'sigma*phi': s * phi,
         'pk2': g('pk2'), 'pk3': g('pk3'), 'pk4': g('pk4'), 'g4': g('g4'), 'd21s': g('d21s')}
    iv = 1.0 / var
    M = {'s3': B['D3'] * iv / s, 's4': B['g4'] * iv * iv, 'dn': B['D21n'] * iv, 'v': var,
         'ds': B['d21s'] * iv, 'cr': g('crs') * iv, 'c2': g('cr2') * iv * iv, 'k4': B['K4']}
    M['s33'] = M['s3'] ** 2
    for k in range(7):
        B['h%d' % k] = s * phi * herm(a, k)
    return B, M


def feats(d, l, names_):
    B, M = base(d, l)
    cols = []
    for nm in names_:
        if nm in B:
            cols.append(B[nm])
        elif nm.startswith('sphiHe'):
            cols.append(B['h%d' % int(nm[6:])])
        else:
            hb, _, md = nm.partition('*')
            cols.append(B[hb] * M[md])
    return np.stack(cols, 1)


F13 = ['1', 'mu', 'var', 'sigma', 'alpha', '|alpha|', 'phi', 'Phi', 'pred', 'D3', 'D21n', 'K4', 'sigma*phi']
F20 = F13 + ['sphiHe%d' % k for k in range(1, 7)] + ['d21s']
MODS = ['s3', 's4', 's33', 'dn', 'v', 'ds', 'cr', 'c2', 'k4']
F87 = F13 + ['pk2', 'pk3', 'pk4', 'g4'] + [x for k in range(7) for x in ['h%d' % k] + ['h%d*%s' % (k, m) for m in MODS]]
F59 = [x for x in F87 if not any(x.endswith('*' + m) for m in ('ds', 'cr', 'c2', 'k4'))]
F2 = ['1', 'pred']
assert len(F87) == 87 and len(F59) == 59, (len(F87), len(F59))
SETS = [('F2', F2), ('F13', F13), ('F20', F20), ('F59', F59), ('F87', F87)]

data = [(nm, np.load(f)) for f, nm in zip(files, names)]
fit = [(nm, d) for nm, d in data if nm not in hold]
hol = [(nm, d) for nm, d in data if nm in hold]
L = fit[0][1]['pk1'].shape[0]
print('мереж: усього %d, фіт %d, відкладено %d, шарів %d' % (len(data), len(fit), len(hol), L))
# пломба: final_means == all_layer_means[15]
nm0 = names[0]
print('пломба final_means vs all_layer_means[-1]: max|Δ| %.3e' % np.abs(finals[nm0] - truth[nm0][-1]).max())
g4v = np.std(fit[0][1]['g4'][L - 1]); print('g4 розкид по нейронах (шар %d): %.3e' % (L - 1, g4v))

LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]


def fit_layer(Xs, Ys):
    """ridge на списку мереж; повертає (w, mu, sc, lam, sse_cv, sst)"""
    Xa = np.concatenate(Xs)
    mu_, sc_ = Xa.mean(0), Xa.std(0); sc_[sc_ < 1e-12] = 1.0; mu_[0] = 0.0; sc_[0] = 1.0
    Z = [(x - mu_) / sc_ for x in Xs]
    nb = 5; idx = np.arange(len(Xs)) % nb
    G = [(z.T @ z, z.T @ y) for z, y in zip(Z, Ys)]
    best = None
    for lam in LAMS:
        sse = sst = 0.0
        for b in range(nb):
            A = sum(G[i][0] for i in range(len(Xs)) if idx[i] != b)
            y = sum(G[i][1] for i in range(len(Xs)) if idx[i] != b)
            R = lam * np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A + R, y)
            for i in range(len(Xs)):
                if idx[i] == b:
                    sse += ((Ys[i] - Z[i] @ w) ** 2).sum(); sst += (Ys[i] ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, sst, lam)
    sse, sst, lam = best
    A = sum(g[0] for g in G); y = sum(g[1] for g in G)
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A + R, y)
    return w, mu_, sc_, lam, sse, sst


out = {}
Yfit = [[truth[nm][l] - d['pk1'][l].astype(np.float64) for nm, d in fit] for l in range(L)]
Yhol = [[truth[nm][l] - d['pk1'][l].astype(np.float64) for nm, d in hol] for l in range(L)]
print('\nкрос (фіт 10..59), SST/SSE по шарах = виграш MSE шару в один крок:')
print('шар  ' + '  '.join('%8s' % s for s, _ in SETS))
tabs = {s: [] for s, _ in SETS}
cvx = {s: [] for s, _ in SETS}
for l in range(L):
    row = []
    for s, FN in SETS:
        Xs = [feats(d, l, FN) for _, d in fit]
        w, mu_, sc_, lam, sse, sst = fit_layer(Xs, Yfit[l])
        tabs[s].append((w, mu_, sc_, lam))
        cvx[s].append(sst / sse)
        row.append(sst / sse)
    print('%3d  ' % l + '  '.join('%8.4f' % v for v in row))

print('\nвідкладені s0..s9: виграш MSE шару в один крок (таблиця свого шару, без перенесення):')
print('шар  ' + '  '.join('%8s' % s for s, _ in SETS))
for l in range(L):
    row = []
    for s, FN in SETS:
        w, mu_, sc_, lam = tabs[s][l]
        e0 = e1 = 0.0
        for (nm, d), y in zip(hol, Yhol[l]):
            x = (feats(d, l, FN) - mu_) / sc_
            e0 += (y ** 2).sum(); e1 += ((y - x @ w) ** 2).sum()
        row.append(e0 / e1)
    print('%3d  ' % l + '  '.join('%8.4f' % v for v in row))

l = L - 1
print('\nвихідний коректор на відкладених s0..s9 (MSE вихідного шару, середнє по мережах):')
for s, FN in SETS:
    w, mu_, sc_, lam = tabs[s][l]
    per0, per1 = [], []
    res = {}
    for c in (0.25, 0.5, 0.75, 1.0, 1.25):
        m1 = []
        for (nm, d), y in zip(hol, Yhol[l]):
            x = (feats(d, l, FN) - mu_) / sc_
            m1.append(((y - c * (x @ w)) ** 2).mean())
        res[c] = np.mean(m1)
    m0 = np.mean([(y ** 2).mean() for y in Yhol[l]])
    # кращих мереж при c=1
    better = 0
    for (nm, d), y in zip(hol, Yhol[l]):
        x = (feats(d, l, FN) - mu_) / sc_
        better += ((y - x @ w) ** 2).mean() < (y ** 2).mean()
    # контроль «чужа»: ознаки мережі i+1 до цілі мережі i
    mf = []
    for i, ((nm, d), y) in enumerate(zip(hol, Yhol[l])):
        dj = hol[(i + 1) % len(hol)][1]
        x = (feats(dj, l, FN) - mu_) / sc_
        mf.append(((y - x @ w) ** 2).mean())
    print('%-4s λ=%-7g MSE до %.4e · ' % (s, lam, m0) + ' '.join('c=%.2f ×%.4f' % (c, m0 / v) for c, v in res.items())
          + ' · кращих %d/10 · чужа ×%.4f' % (better, m0 / np.mean(mf)))

out = {s: {'feats': FN, 'beta': [t[0].tolist() for t in tabs[s]], 'mean': [t[1].tolist() for t in tabs[s]],
           'scale': [t[2].tolist() for t in tabs[s]], 'lam': [t[3] for t in tabs[s]], 'cv_gain': cvx[s]}
       for s, FN in SETS}
json.dump(out, open('/home/trim/wb2lean/v39fija/fit87_tables.json', 'w'))
print('\nтаблиці: /home/trim/wb2lean/v39fija/fit87_tables.json')
