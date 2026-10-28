"""E-A (18.10.2026): ridge-корекція середнього по шарах на дампах v39 (вільний прогін).
Фіт на мережах mini 10..59, s0..s9 (= mini 0..9) не чіпати. Ціль δ_l = truth_l − pk1_l понейронно.
Друкує по шару: R² на кросі (блоки по мережах), відносне зменшення суми квадратів похибки, λ.
Ознаки: 13 старих (1, mu, var, sigma, alpha, |alpha|, phi, Phi, pred, D3, D21n, K4, sigma*phi)
+ sigma*phi*He_k(alpha), k=1..6, + d21s.  Вихід: corr_table.json {feats: [...], beta: 16 x F, mean, scale}.
Запуск: /home/trim/fs123/bin/python3 fit_corr.py <dump_dir> <run_json>"""
import glob, json, math, sys
import numpy as np
import pyarrow.parquet as pq

DUMP, RUNJ = sys.argv[1], sys.argv[2]
P = '/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/aa99830fdc09fad15407b10e8e3459d3e18bba0a/data'
files = sorted(glob.glob(DUMP + '/net_*.npz'), key=lambda f: int(f.split('_')[-2]))
names = [m['mlp_name'] for m in json.load(open(RUNJ))['results']['per_mlp']]
assert len(files) == len(names), (len(files), len(names))
truth = {}
for f in sorted(glob.glob(P + '/mini-*.parquet')):
    t = pq.read_table(f, columns=['mlp_name', 'all_layer_means'])
    for nm, alm in zip(t.column('mlp_name').to_pylist(), t.column('all_layer_means').to_pylist()):
        truth[nm] = np.array(alm, dtype=np.float64)
hold = set(names[:10])


def herm(a, k):
    h0, h1 = np.ones_like(a), a
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, a * h1 - j * h0
    return h1


def feats(d, l):
    mu, var, pred = d['mu'][l].astype(np.float64), d['var'][l].astype(np.float64), d['pk1'][l].astype(np.float64)
    s = np.sqrt(var); a = mu / s
    phi = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2)))
    one = np.ones_like(a)
    cols = [one, mu, var, s, a, np.abs(a), phi, Phi, pred, d['d3'][l], d['d21n'][l], one * d['k4s'][l], s * phi]
    cols += [s * phi * herm(a, k) for k in range(1, 7)] + [d['d21s'][l]]
    return np.stack(cols, 1)


FN = ['1', 'mu', 'var', 'sigma', 'alpha', '|alpha|', 'phi', 'Phi', 'pred', 'D3', 'D21n', 'K4', 'sigma*phi'] + \
     ['sphiHe%d' % k for k in range(1, 7)] + ['d21s']
data = []
for f, nm in zip(files, names):
    d = np.load(f)
    data.append((nm, d))
fit = [(nm, d) for nm, d in data if nm not in hold]
print('мереж: усього %d, фіт %d, відкладено %d' % (len(data), len(fit), len(data) - len(fit)))
L = fit[0][1]['pk1'].shape[0]
out = {'feats': FN, 'beta': [], 'mean': [], 'scale': [], 'lam': []}
for l in range(L):
    X = [feats(d, l) for _, d in fit]
    Y = [truth[nm][l] - d['pk1'][l] for nm, d in fit]
    Xa, Ya = np.concatenate(X), np.concatenate(Y)
    mu_, sc_ = Xa.mean(0), Xa.std(0); sc_[sc_ < 1e-12] = 1.0; mu_[0] = 0.0; sc_[0] = 1.0
    Z = [(x - mu_) / sc_ for x in X]
    nb = 5; idx = np.arange(len(fit)) % nb
    best = None
    for lam in [1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]:
        sse = sst = 0.0
        for b in range(nb):
            tr = [i for i in range(len(fit)) if idx[i] != b]; te = [i for i in range(len(fit)) if idx[i] == b]
            A = np.concatenate([Z[i] for i in tr]); y = np.concatenate([Y[i] for i in tr])
            R = lam * np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A.T @ A + R, A.T @ y)
            for i in te:
                sse += ((Y[i] - Z[i] @ w) ** 2).sum(); sst += (Y[i] ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, sst, lam)
    sse, sst, lam = best
    A = np.concatenate(Z); R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(A.T @ A + R, A.T @ Ya)
    print('шар %2d  крос: SSE/SST %.4f (R2 %.4f)  λ=%g' % (l, sse / sst, 1 - sse / sst, lam))
    out['beta'].append(w.tolist()); out['mean'].append(mu_.tolist()); out['scale'].append(sc_.tolist()); out['lam'].append(lam)
# відкладені s0..s9: лише останній шар (вихід), без перенесення
r = []
for nm, d in data:
    if nm in hold:
        l = L - 1; x = (feats(d, l) - np.array(out['mean'][l])) / np.array(out['scale'][l])
        e0 = truth[nm][l] - d['pk1'][l]; e1 = e0 - x @ np.array(out['beta'][l])
        r.append(((e0 ** 2).mean(), (e1 ** 2).mean()))
if r:
    print('відкладені, лише вихідний шар: MSE до %.4e після %.4e  ×%.4f' % (np.mean([a for a, _ in r]), np.mean([b for _, b in r]),
          np.mean([a for a, _ in r]) / np.mean([b for _, b in r])))
json.dump(out, open('/home/trim/wb2lean/v39fija/corr_table.json', 'w'))
