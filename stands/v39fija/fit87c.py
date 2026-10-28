"""E-A (18.10.2026, сесія a2): фіт корекції середнього з 5-кратним кросом по мережах (без окремих відкладених).
Дані: дампи v39 по одній мережі (a5: cd_m10..cd_m59 = mini 10..59), /home/trim/wblogs/corrdump/net_<time_ns>_000.npz.
Прив'язка файл → мережа: за збігом pk1 шару 0 з істиною (ланцюг на шарі 0 точний ~1e-5; чужа мережа ~0.5).
Цілі: y_l = truth_l − pk1_l (вільний прогін) і step_l = y_l − (y_{l−1} @ W_l)·Φ_l («свіжий крок», наближення X_TAB_FORCE e30_a).
Ridge по шару: стандартизовані колонки, вільний перетин, λ внутрішнім 5-блоковим кросом (як e30_a).
Оцінка на відкладеному блоці (фіт на 40, оцінка на 10, по колу):
  (1) виграш MSE шару в один крок; (2) вихідний коректор (лише шар 15, переноситься в бал прямо);
  (3) емуляція онлайн-корекції через лінійний відгук ланцюга: acc_l = (acc_{l−1} @ W_l)·Φ_l + c·d_l, вихід = pk1_15 + acc_15;
  контроль «чужа мережа» для (2); скан множника c; внесок свіжого кроку кожного шару у вихідну похибку (телескоп, пломба).
Підсумкові таблиці фітуються на всіх 50 мережах: /home/trim/wb2lean/v39fija/fit87c_tables.json (ключі <ціль>_<набір>).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (18.10, a2; перша мережа вже показала step/y 0.18 на виході):
  крос R² на шарі 15: на y 0.01–0.04, на step 0.03–0.12 (87 ознак); вихідний коректор ×1.01–×1.04;
  онлайн-емуляція, таблиці на step, 87 ознак: вихід ×1.05–×1.15, кращий c 0.75–1.0; таблиці на y в онлайні гірші за step.
  Шанс, що онлайн дасть ≥ ×1.08: ~35%; що вихідний коректор дасть ≥ ×1.05: ~15%.
Запуск: /home/trim/fs123/bin/python3 fit87c.py [n_networks=50]
"""
import glob, json, math, os, sys, time
import numpy as np
import pyarrow.parquet as pq

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 50
P = ('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/'
     'aa99830fdc09fad15407b10e8e3459d3e18bba0a/data')
OUT = os.environ.get('FIT_OUT', '/home/trim/wb2lean/v39fija/fit87c_tables.json')
DDIR = os.environ.get('FIT_DUMP_DIR', '/home/trim/wblogs/corrdump')
_erf = np.vectorize(math.erf)

truth = {}
for f in sorted(glob.glob(P + '/mini-*.parquet')):
    t = pq.read_table(f, columns=['mlp_name', 'all_layer_means'])
    for nm, alm in zip(t.column('mlp_name').to_pylist(), t.column('all_layer_means').to_pylist()):
        truth[nm] = np.array(alm, dtype=np.float64)
allnames = list(truth)

files = sorted(glob.glob(DDIR + '/net_*.npz'), key=lambda f: int(f.split('_')[-2]))[-NNET:]
dump, names = {}, []
for f in files:
    d = np.load(f)
    p0 = d['pk1'][0].astype(np.float64)
    errs = sorted((np.sqrt(((p0 - truth[nm][0]) ** 2).mean()), nm) for nm in allnames)
    (e1, nm1), (e2, nm2) = errs[0], errs[1]
    assert e1 < 1e-3 and e2 > 20 * e1, (f, e1, nm1, e2, nm2)
    assert nm1 not in dump, ('двічі', nm1)
    dump[nm1] = d
    names.append(nm1)
print('мереж %d; прив\'язка за шаром 0: макс rms своєї %.2e' % (len(names), max(
    np.sqrt(((dump[nm]['pk1'][0] - truth[nm][0]) ** 2).mean()) for nm in names)))
L = dump[names[0]]['pk1'].shape[0]
NF = 5
fold = {nm: i % NF for i, nm in enumerate(names)}


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
    return B, M, Phi


def feats(d, l, names_):
    B, M, _ = base(d, l)
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
assert len(F87) == 87 and len(F59) == 59
SETS = [('F2', F2), ('F13', F13), ('F20', F20), ('F59', F59), ('F87', F87)]
LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]
PHI = {nm: np.stack([base(dump[nm], l)[2] for l in range(L)]) for nm in names}
Y = {nm: np.stack([truth[nm][l] - dump[nm]['pk1'][l].astype(np.float64) for l in range(L)]) for nm in names}
g4s = np.mean([dump[nm]['g4'][L - 1].std() / (abs(dump[nm]['g4'][L - 1].mean()) + 1e-30) for nm in names])
print('g4 на шарі %d: відносний розкид по нейронах %.3f' % (L - 1, g4s))


def iter_weights(want):
    if os.environ.get('FIT_FAKE_W'):   # тест логіки без важкого parquet
        rng = np.random.default_rng(3)
        for nm in names:
            yield nm, (rng.normal(0, math.sqrt(2 / 1024), (16, 1024, 1024))).astype(np.float32)
        return
    for f in sorted(glob.glob(P + '/mini-*.parquet')):
        nms = pq.read_table(f, columns=['mlp_name']).column('mlp_name').to_pylist()
        if not any(nm in want for nm in nms):
            continue
        t = pq.read_table(f, columns=['weights'])
        vals = t.column('weights').combine_chunks().flatten().flatten().flatten()
        try:
            Wall = vals.to_numpy(zero_copy_only=True).reshape(len(nms), 16, 1024, 1024)
        except Exception:
            Wall = vals.to_numpy(zero_copy_only=False).reshape(len(nms), 16, 1024, 1024)
        for i, nm in enumerate(nms):
            if nm in want:
                yield nm, Wall[i]
        del t, vals, Wall


def tr(v, W, l):
    return v @ W[l].astype(np.float64)


# ---- прохід 1 по вагах: свіжий крок і внески шарів ----
t0 = time.time()
STEP, CON = {}, {}
for nm, W in iter_weights(set(names)):
    y = Y[nm]
    inh = np.zeros_like(y)
    inh[1:] = np.stack([(y[l - 1] @ W[l].astype(np.float64)) * PHI[nm][l] for l in range(1, L)])
    STEP[nm] = y - inh
    Tl = STEP[nm].copy()            # рядок k: свіжий крок шару k, перенесений до поточного шару
    for m in range(1, L):
        Tl[:m] = (Tl[:m] @ W[m].astype(np.float64)) * PHI[nm][m]
    CON[nm] = Tl
print('прохід 1: %.0f с' % (time.time() - t0))
print('\nчастки по шарах (сер. по мережах): ‖step‖²/‖y‖², corr(y, перенесене)')
for l in range(L):
    sy = sum((Y[nm][l] ** 2).sum() for nm in names)
    ss = sum((STEP[nm][l] ** 2).sum() for nm in names)
    inh = {nm: Y[nm][l] - STEP[nm][l] for nm in names}
    si = sum((inh[nm] ** 2).sum() for nm in names)
    c = sum((Y[nm][l] * inh[nm]).sum() for nm in names) / math.sqrt(sy * si + 1e-300)
    print('шар %2d  rms y %.3e  step/y %.3f  corr %.3f' % (l, math.sqrt(sy / len(names) / 1024), ss / sy, c))
tot = sum((Y[nm][L - 1] ** 2).sum() for nm in names)
seal = max(np.abs(CON[nm].sum(0) - Y[nm][L - 1]).max() / np.abs(Y[nm][L - 1]).max() for nm in names)
sh = [sum((CON[nm][k] ** 2).sum() for nm in names) / tot for k in range(L)]
print('внесок свіжого кроку шару у вихідну похибку ‖T_l‖²/‖y_15‖² (пломба телескопа %.1e):' % seal)
print(' '.join('%d:%.3f' % (k, v) for k, v in enumerate(sh)), '· сума %.3f' % sum(sh))


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


FX = {s: {nm: [feats(dump[nm], l, FN) for l in range(L)] for nm in names} for s, FN in SETS}
TG = {'y': Y, 'step': STEP}
# таблиці кросу: TABF[(ціль, набір, блок)][l] = (w, mu, sc, lam) з фітом без блоку
t0 = time.time()
TABF = {}
for tg in TG:
    for s, _ in SETS:
        for b in range(NF):
            trn = [nm for nm in names if fold[nm] != b]
            TABF[(tg, s, b)] = [fit_layer([FX[s][nm][l] for nm in trn], [TG[tg][nm][l] for nm in trn]) for l in range(L)]
print('\nфіт кросу: %.0f с' % (time.time() - t0))


def dvec(tab, s, nm, l):
    w, mu_, sc_, _ = tab[l]
    return ((FX[s][nm][l] - mu_) / sc_) @ w


print('\n(1) виграш MSE шару в один крок на відкладених блоках (SST/SSE), ціль = своя ціль фіту:')
for tg in TG:
    print('ціль %s:  шар  ' % tg + '  '.join('%7s' % s for s, _ in SETS))
    for l in range(L):
        row = []
        for s, _ in SETS:
            e0 = e1 = 0.0
            for nm in names:
                yv = TG[tg][nm][l]
                e0 += (yv ** 2).sum(); e1 += ((yv - dvec(TABF[(tg, s, fold[nm])], s, nm, l)) ** 2).sum()
            row.append(e0 / e1)
        print('              %3d  ' % l + '  '.join('%7.4f' % v for v in row))

m0 = np.mean([(Y[nm][L - 1] ** 2).mean() for nm in names])
print('\n(2) вихідний коректор (лише шар 15) на відкладених блоках, MSE виходу до %.4e:' % m0)
for tg in TG:
    for s, _ in SETS:
        res = {}
        for c in (0.5, 0.75, 1.0, 1.25):
            res[c] = np.mean([((Y[nm][L - 1] - c * dvec(TABF[(tg, s, fold[nm])], s, nm, L - 1)) ** 2).mean() for nm in names])
        better = sum(((Y[nm][L - 1] - dvec(TABF[(tg, s, fold[nm])], s, nm, L - 1)) ** 2).mean() < (Y[nm][L - 1] ** 2).mean()
                     for nm in names)
        fr = np.mean([((Y[nm][L - 1] - dvec(TABF[(tg, s, fold[nm])], s, names[(i + 1) % len(names)], L - 1)) ** 2).mean()
                      for i, nm in enumerate(names)])
        print('%-4s %-4s ' % (tg, s) + ' '.join('c=%.2f ×%.4f' % (c, m0 / v) for c, v in res.items())
              + ' · кращих %d/%d · чужа ×%.4f' % (better, len(names), m0 / fr))

# ---- прохід 2 по вагах: емуляція онлайн ----
print('\n(3) емуляція онлайн-корекції (усі шари з переносом лінійним відгуком), відкладені блоки:')
CS = (0.5, 0.75, 1.0)
LMINS = (0, 8, 12)
acc_res = {}
t0 = time.time()
VAR = [(tg, s, lmin, c) for tg in TG for s, _ in SETS for lmin in LMINS for c in CS]
for nm, W in iter_weights(set(names)):
    Dv = {}
    for tg in TG:
        for s, _ in SETS:
            Dv[(tg, s)] = np.stack([dvec(TABF[(tg, s, fold[nm])], s, nm, l) for l in range(L)])
    DD = np.stack([np.stack([(c * Dv[(tg, s)][l] if l >= lmin else 0.0 * Dv[(tg, s)][l]) for l in range(L)])
                   for (tg, s, lmin, c) in VAR])            # (V, L, n)
    acc = DD[:, 0].copy()
    allm = ((Y[nm][0][None] - acc) ** 2).mean(1)
    for l in range(1, L):
        acc = (acc @ W[l].astype(np.float64)) * PHI[nm][l][None] + DD[:, l]
        allm += ((Y[nm][l][None] - acc) ** 2).mean(1)
    fin = ((Y[nm][L - 1][None] - acc) ** 2).mean(1)
    for i, key in enumerate(VAR):
        acc_res.setdefault(key, []).append((fin[i], allm[i] / L, (Y[nm][L - 1] ** 2).mean(), (Y[nm] ** 2).mean()))
print('прохід 2: %.0f с' % (time.time() - t0))
for (tg, s, lmin, c), v in sorted(acc_res.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], kv[0][3])):
    v = np.array(v)
    better = int((v[:, 0] < v[:, 2]).sum())
    print('%-4s %-4s з шару %2d c=%.2f: вихід ×%.4f (кращих %2d/%d) · усі шари ×%.4f'
          % (tg, s, lmin, c, v[:, 2].mean() / v[:, 0].mean(), better, len(v), v[:, 3].mean() / v[:, 1].mean()))

# ---- підсумкові таблиці на всіх 50 ----
out = {}
for tg in TG:
    for s, FN in SETS:
        tab = [fit_layer([FX[s][nm][l] for nm in names], [TG[tg][nm][l] for nm in names]) for l in range(L)]
        out['%s_%s' % (tg, s)] = {'feats': FN, 'beta': [t[0].tolist() for t in tab], 'mean': [t[1].tolist() for t in tab],
                                  'scale': [t[2].tolist() for t in tab], 'lam': [t[3] for t in tab], 'nets': names}
json.dump(out, open(OUT, 'w'))
print('\nтаблиці (фіт на всіх %d): %s' % (len(names), OUT))
