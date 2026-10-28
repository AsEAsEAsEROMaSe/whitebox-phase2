"""E-A (18.10.2026, сесія a2): «свіжа» похибка кроку і емуляція онлайн-корекції через лінійний відгук ланцюга.
Навіщо: вільний прогін дає ціль y_l = truth_l − pk1_l = успадковане + крок. Успадковане (похибка середнього
шару l−1, перенесена шаром l) локальні ознаки не бачать, воно лише шумить фіт. Тут:
  inh_l  = (y_{l−1} @ W_l) * Phi_l          (лінійний відгук середнього: dE[relu]/dmu = P(p>0) ≈ Phi(alpha))
  step_l = y_l − inh_l                       (наближення до X_TAB_FORCE e30_a без прогону ланцюга)
Таблиці фітуються на step (і для порівняння на y), а ефект онлайн-корекції на вихідному шарі емулюється:
  acc_0 = d_0;  acc_l = (acc_{l−1} @ W_l) * Phi_l + d_l;  вихід = pk1_15 + acc_15
(корекція в ланцюзі міняє лише середнє, коваріацію ні; перший порядок по d ~1e-4 відносно).
Пломби: орієнтація W (E[relu] шару 0 = норми стовпців / √2π проти істини), acc з d=0 = база біт у біт,
емуляція «лише вихідний шар» = прямий вихідний коректор.
ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (18.10, a2): ‖step‖²/‖y‖² на глибоких шарах 0.2–0.5 (решта успадкована лінійно);
крос R² на step у 2–5 разів вищий, ніж на y; емуляція онлайн 87 ознак: вихід ×1.05–×1.15 на s0..s9;
таблиці на step кращі за таблиці на y в онлайн-емуляції. Шанс ≥ ×1.08: ~35%.
Запуск: /home/trim/fs123/bin/python3 fit87b.py <dump_dir> <run_json>
"""
import glob, json, math, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, '/home/trim/wb2lean/v39fija')
DUMP, RUNJ = sys.argv[1], sys.argv[2]
P = ('/home/trim/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026/snapshots/'
     'aa99830fdc09fad15407b10e8e3459d3e18bba0a/data')
files = sorted(glob.glob(DUMP + '/net_*.npz'), key=lambda f: int(f.split('_')[-2]))
names = [m['mlp_name'] for m in json.load(open(RUNJ))['results']['per_mlp']]
assert len(files) >= len(names), (len(files), len(names))
files = files[-len(names):]   # старі файли обірваних прогонів відкидаються (беремо найновіші за time_ns)
dump = {nm: np.load(f) for f, nm in zip(files, names)}
hold = names[:10]
fitn = names[10:]
_erf = np.vectorize(math.erf)
L = dump[names[0]]['pk1'].shape[0]


def herm(a, k):
    h0, h1 = np.ones_like(a), a
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, a * h1 - j * h0
    return h1


def state(d, l):
    mu, var = d['mu'][l].astype(np.float64), d['var'][l].astype(np.float64)
    s = np.sqrt(var); a = mu / s
    return mu, var, s, a


def Phi_of(d, l):
    _, _, _, a = state(d, l)
    return 0.5 * (1 + _erf(a / math.sqrt(2)))


def base(d, l):
    g = lambda k: d[k][l].astype(np.float64)
    mu, var, s, a = state(d, l)
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
F2 = ['1', 'pred']
SETS = [('F2', F2), ('F13', F13), ('F20', F20), ('F87', F87)]
LAMS = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]


def iter_weights(want):
    """(name, W[16,1024,1024] float32) для мереж з want, файл за файлом (~1 ГБ на файл)."""
    for f in sorted(glob.glob(P + '/mini-*.parquet')):
        t = pq.read_table(f, columns=['mlp_name'])
        nms = t.column('mlp_name').to_pylist()
        if not any(nm in want for nm in nms):
            continue
        t = pq.read_table(f, columns=['mlp_name', 'weights'])
        col = t.column('weights').combine_chunks()
        vals = col.storage.flatten().flatten().flatten() if hasattr(col, 'storage') else col.flatten().flatten().flatten()
        Wall = vals.to_numpy(zero_copy_only=False).reshape(len(nms), 16, 1024, 1024)
        for i, nm in enumerate(nms):
            if nm in want:
                yield nm, Wall[i]
        del t, col, vals, Wall


truth = {}
for f in sorted(glob.glob(P + '/mini-*.parquet')):
    t = pq.read_table(f, columns=['mlp_name', 'all_layer_means'])
    for nm, alm in zip(t.column('mlp_name').to_pylist(), t.column('all_layer_means').to_pylist()):
        truth[nm] = np.array(alm, dtype=np.float64)
Y = {nm: np.stack([truth[nm][l] - dump[nm]['pk1'][l].astype(np.float64) for l in range(L)]) for nm in names}
PHI = {nm: np.stack([Phi_of(dump[nm], l) for l in range(L)]) for nm in names}

# ---- прохід 1: успадковане і крок ----
t0 = time.time()
STEP, INH, WHOLD, CONTRIB = {}, {}, {}, {}
orient_done = False
TRANS = False
for nm, W in iter_weights(set(names)):
    if not orient_done:
        cn = np.linalg.norm(W[0].astype(np.float64), axis=0) / math.sqrt(2 * math.pi)
        rn = np.linalg.norm(W[0].astype(np.float64), axis=1) / math.sqrt(2 * math.pi)
        ec, er = np.sqrt(((cn - truth[nm][0]) ** 2).mean()), np.sqrt(((rn - truth[nm][0]) ** 2).mean())
        print('пломба орієнтації (шар 0, rms до істини): стовпці %.3e · рядки %.3e · pk1 ланцюга %.3e'
              % (ec, er, np.sqrt((Y[nm][0] ** 2).mean())))
        TRANS = er < ec          # True ⟹ p = W h (рядки), беремо W.T
        print('орієнтація: p = h @ %s' % ('W.T' if TRANS else 'W'))
        orient_done = True
    y = Y[nm]
    inh = np.zeros_like(y)
    for l in range(1, L):
        inh[l] = (y[l - 1] @ (W[l].T if TRANS else W[l]).astype(np.float64)) * PHI[nm][l]
    INH[nm] = inh
    STEP[nm] = y - inh
    # внесок свіжого кроку кожного шару у вихідну похибку: T_l = step_l J_{l->15}; Σ_l T_l = y_15 тотожно
    Tl = np.zeros((L, 1024))
    for k in range(L):
        v = STEP[nm][k].copy()
        for m in range(k + 1, L):
            v = (v @ (W[m].T if TRANS else W[m]).astype(np.float64)) * PHI[nm][m]
        Tl[k] = v
    CONTRIB[nm] = Tl
    if nm in hold:
        WHOLD[nm] = W.copy()
print('прохід 1: %.0f с, мереж %d' % (time.time() - t0, len(STEP)))
print('\nчастки по шарах (сер. по всіх 60): ‖step‖²/‖y‖², ‖inh‖²/‖y‖², corr(y, inh)')
for l in range(L):
    sy = sum((Y[nm][l] ** 2).sum() for nm in names)
    ss = sum((STEP[nm][l] ** 2).sum() for nm in names)
    si = sum((INH[nm][l] ** 2).sum() for nm in names)
    cyi = sum((Y[nm][l] * INH[nm][l]).sum() for nm in names) / math.sqrt(sy * si + 1e-300)
    print('шар %2d  y² %.3e  step %.3f  inh %.3f  corr %.3f' % (l, sy / len(names) / 1024, ss / sy, si / sy, cyi))


print('\nвнесок свіжого кроку шару l у вихідну похибку (T_l = step_l·J_{l→15}), частки енергії ‖T_l‖²/‖y_15‖²:')
tot = sum((Y[nm][L - 1] ** 2).sum() for nm in names)
seal = max(np.abs(CONTRIB[nm].sum(0) - Y[nm][L - 1]).max() / np.abs(Y[nm][L - 1]).max() for nm in names)
print('пломба телескопа Σ_l T_l = y_15: max відн. %.2e' % seal)
sh = [sum((CONTRIB[nm][k] ** 2).sum() for nm in names) / tot for k in range(L)]
print(' '.join('%d:%.3f' % (k, v) for k, v in enumerate(sh)), '· сума часток %.3f (1 = некогерентно)' % sum(sh))


def fit_layer(Xs, Ys):
    Xa = np.concatenate(Xs)
    mu_, sc_ = Xa.mean(0), Xa.std(0); sc_[sc_ < 1e-12] = 1.0; mu_[0] = 0.0; sc_[0] = 1.0
    Z = [(x - mu_) / sc_ for x in Xs]
    nb = 5; idx = np.arange(len(Xs)) % nb
    G = [(z.T @ z, z.T @ y) for z, y in zip(Z, Ys)]
    best = None
    for lam in LAMS:
        sse = sst = 0.0
        for b in range(nb):
            A = np.sum([G[i][0] for i in range(len(Xs)) if idx[i] != b], axis=0)
            yv = np.sum([G[i][1] for i in range(len(Xs)) if idx[i] != b], axis=0)
            R = lam * np.eye(A.shape[1]); R[0, 0] = 0
            w = np.linalg.solve(A + R, yv)
            for i in range(len(Xs)):
                if idx[i] == b:
                    sse += ((Ys[i] - Z[i] @ w) ** 2).sum(); sst += (Ys[i] ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, sst, lam)
    sse, sst, lam = best
    A = np.sum([g[0] for g in G], axis=0); yv = np.sum([g[1] for g in G], axis=0)
    R = lam * np.eye(A.shape[1]); R[0, 0] = 0
    return np.linalg.solve(A + R, yv), mu_, sc_, lam, sst / sse


FX = {s: {nm: [feats(dump[nm], l, FN) for l in range(L)] for nm in names} for s, FN in SETS}
TABS = {}
for tgt, TG in (('y', Y), ('step', STEP)):
    print('\nкрос на цілі %s (фіт 10..59), SST/SSE по шарах:' % tgt)
    print('шар  ' + '  '.join('%8s' % s for s, _ in SETS))
    for s, _ in SETS:
        TABS[(tgt, s)] = []
    for l in range(L):
        row = []
        for s, _ in SETS:
            w, mu_, sc_, lam, g = fit_layer([FX[s][nm][l] for nm in fitn], [TG[nm][l] for nm in fitn])
            TABS[(tgt, s)].append((w, mu_, sc_, lam))
            row.append(g)
        print('%3d  ' % l + '  '.join('%8.4f' % v for v in row))

# ---- прохід 2: емуляція онлайн на відкладених ----
print('\nемуляція на s0..s9, MSE вихідного шару (сер. по мережах); «вих» = лише шар 15, «онл» = усі шари з переносом')
m0 = np.mean([(Y[nm][L - 1] ** 2).mean() for nm in hold])
a0 = np.mean([(Y[nm] ** 2).mean() for nm in hold])
print('база: вихід %.4e · усі шари %.4e' % (m0, a0))
# пломба: d = 0 ⟹ acc = 0
for (tgt, s), tab in TABS.items():
    for c in (0.5, 0.75, 1.0):
        mo, mf, ma, nb_ = [], [], [], 0
        for nm in hold:
            W = WHOLD[nm]
            acc = np.zeros(1024)
            accs = []
            for l in range(L):
                w, mu_, sc_, lam = tab[l]
                dl = c * (((FX[s][nm][l] - mu_) / sc_) @ w)
                acc = (acc @ (W[l].T if TRANS else W[l]).astype(np.float64)) * PHI[nm][l] + dl if l > 0 else dl
                accs.append(acc)
                if l == L - 1:
                    dfin = dl
            e_on = Y[nm][L - 1] - acc
            e_out = Y[nm][L - 1] - dfin
            mo.append((e_on ** 2).mean()); mf.append((e_out ** 2).mean())
            ma.append(np.mean([((Y[nm][l] - accs[l]) ** 2).mean() for l in range(L)]))
            nb_ += (e_on ** 2).mean() < (Y[nm][L - 1] ** 2).mean()
        print('%-4s %-4s c=%.2f  онл: вихід ×%.4f (кращих %2d/10) усі ×%.4f  ·  вих: ×%.4f'
              % (tgt, s, c, m0 / np.mean(mo), nb_, a0 / np.mean(ma), m0 / np.mean(mf)))

out = {'%s_%s' % k: {'feats': dict(SETS)[k[1]], 'beta': [t[0].tolist() for t in v], 'mean': [t[1].tolist() for t in v],
                     'scale': [t[2].tolist() for t in v], 'lam': [t[3] for t in v]} for k, v in TABS.items()}
json.dump(out, open('/home/trim/wb2lean/v39fija/fit87b_tables.json', 'w'))
print('\nтаблиці: /home/trim/wb2lean/v39fija/fit87b_tables.json')
