"""19.10.2026 white-box-8c: нелінійний коректор (MLP, як LT у Susupro/logxio, MIT) проти ridge step_F87 на наших 90 дампах v39.
Протокол fit90: 5 блоків по 90 мережах; для блоку b обидва коректори вчаться на 72 мережах поза b, оцінка на 18 відкладених.
Ціль: свіжий крок STEP (як ridge). MLP спільна для всіх шарів: вхід = F87 стандартизовані ПО ШАРУ + one-hot шару (16),
ціль поділена на rms кроку шару; 2 приховані шари H, ReLU, Adam, рання зупинка на 8 внутрішніх мережах (поза b).
Оцінка: емуляція бічного потоку (як LT і як наш v41corr): acc_l = (acc_{l-1} @ W_l)·Φ_l + dv_l, вихід pk1_15 + acc_15.
Варіанти: ridge (c=1) · mlp · mlp+ridge (MLP на залишку ridge, вхід F87 + прогноз ridge).
ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ: на 72 мережах MLP проти ridge на виході ×0.98–×1.03 (ridge уже ×1.085, насичений g∞≈1.0965);
шанс, що mlp або mlp+ridge дасть ≥ ×1.02 понад ridge і ≥ 60% мереж: ~25%. Якщо так ⟹ варто дампити full (більше мереж).
Запуск: /home/trim/fs123/bin/python3 -I fitmlp.py (env MLP_H=64, MLP_EP=12, MLP_LR=1e-3)"""
import os, sys, math, time
import numpy as np

src = open('/home/trim/wb2lean/v39fija/fit90.py').read()
cut = src.index('t0 = time.time()\nDV = {}')
exec(compile(src[:cut], 'fit90_prefix', 'exec'))

H = int(os.environ.get('MLP_H', 64)); EP = int(os.environ.get('MLP_EP', 12)); LR = float(os.environ.get('MLP_LR', 1e-3))
BS = 4096
rng = np.random.default_rng(0)
X87 = {(nm, l): feats(nm, l, F87).astype(np.float32) for nm in names for l in range(L)}
print('ознаки: %.0f МБ' % (sum(x.nbytes for x in X87.values()) / 2**20), flush=True)


def layer_norm(trn):
    st = []
    for l in range(L):
        Xa = np.concatenate([X87[(nm, l)] for nm in trn])
        mu_, sc_ = Xa.mean(0), Xa.std(0); sc_[sc_ < 1e-12] = 1.0; mu_[0] = 0.0; sc_[0] = 1.0
        ys = math.sqrt(np.mean(np.concatenate([STEP[nm][l] for nm in trn]) ** 2))
        st.append((mu_, sc_, ys))
    return st


def design(nm, l, st, extra=None):
    mu_, sc_, ys = st[l]
    z = (X87[(nm, l)] - mu_) / sc_
    oh = np.zeros((z.shape[0], L), np.float32); oh[:, l] = 1
    cols = [z, oh] + ([extra[:, None].astype(np.float32) / ys] if extra is not None else [])
    return np.concatenate(cols, 1).astype(np.float32)


class MLP:
    def __init__(self, fin):
        g = lambda a, b: (rng.standard_normal((a, b)) * math.sqrt(2 / a)).astype(np.float32)
        self.p = [g(fin, H), np.zeros(H, np.float32), g(H, H), np.zeros(H, np.float32), g(H, 1) * 0.1, np.zeros(1, np.float32)]
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]; self.t = 0

    def fwd(self, x):
        W1, b1, W2, b2, W3, b3 = self.p
        h1 = np.maximum(x @ W1 + b1, 0); h2 = np.maximum(h1 @ W2 + b2, 0)
        return (h2 @ W3 + b3)[:, 0], (h1, h2)

    def step(self, x, y):
        W1, b1, W2, b2, W3, b3 = self.p
        out, (h1, h2) = self.fwd(x)
        g = (2.0 / len(y)) * (out - y)[:, None]
        gW3 = h2.T @ g; gb3 = g.sum(0); g2 = (g @ W3.T) * (h2 > 0)
        gW2 = h1.T @ g2; gb2 = g2.sum(0); g1 = (g2 @ W2.T) * (h1 > 0)
        gW1 = x.T @ g1; gb1 = g1.sum(0)
        self.t += 1
        for i, gr in enumerate([gW1, gb1, gW2, gb2, gW3, gb3]):
            self.m[i] = 0.9 * self.m[i] + 0.1 * gr; self.v[i] = 0.999 * self.v[i] + 0.001 * gr * gr
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= LR * mh / (np.sqrt(vh) + 1e-8)

    def pred(self, x):
        return np.concatenate([self.fwd(x[i:i + 65536])[0] for i in range(0, len(x), 65536)])


def train(trn, st, extra=None):
    val = trn[-8:]; fit = trn[:-8]
    def stack(nms):
        Xs = [design(nm, l, st, None if extra is None else extra[(nm, l)]) for nm in nms for l in range(L)]
        ys = [STEP[nm][l] / st[l][2] for nm in nms for l in range(L)]
        return np.concatenate(Xs), np.concatenate(ys).astype(np.float32)
    Xf, yf = stack(fit); Xv, yv = stack(val)
    net = MLP(Xf.shape[1]); best = (np.mean(yv ** 2), [x.copy() for x in net.p], 0)
    for ep in range(EP):
        perm = rng.permutation(len(yf))
        for i in range(0, len(perm), BS):
            j = perm[i:i + BS]; net.step(Xf[j], yf[j])
        e = np.mean((net.pred(Xv) - yv) ** 2)
        if e < best[0]:
            best = (e, [x.copy() for x in net.p], ep + 1)
    net.p = best[1]
    return net, best[2], best[0] / np.mean(yv ** 2)


DVR, DVM, DVMR = {}, {}, {}
t0 = time.time()
for b in range(NF):
    trn = [nm for nm in names if fold[nm] != b]; tst = [nm for nm in names if fold[nm] == b]
    rng.shuffle(trn)
    rid = {}
    for l in range(L):
        tab = fit_layer([X87[(nm, l)].astype(np.float64) for nm in trn], [STEP[nm][l] for nm in trn])
        for nm in names:
            rid[(nm, l)] = ((X87[(nm, l)] - tab[1]) / tab[2]) @ tab[0]
    st = layer_norm(trn)
    net, ep1, r1 = train(trn, st)
    # MLP на залишку ridge (ridge рахується in-sample на trn: на внутрішніх мережах трохи оптимістично)
    res_st = STEP
    STEP_bak = {nm: STEP[nm].copy() for nm in trn}
    for nm in trn:
        STEP[nm] = STEP[nm] - np.stack([rid[(nm, l)] for l in range(L)])
    net2, ep2, r2 = train(trn, st, extra=rid)
    for nm in trn:
        STEP[nm] = STEP_bak[nm]
    for nm in tst:
        for l in range(L):
            DVR[(nm, l)] = rid[(nm, l)]
            DVM[(nm, l)] = net.pred(design(nm, l, st)) * st[l][2]
            DVMR[(nm, l)] = rid[(nm, l)] + net2.pred(design(nm, l, st, rid[(nm, l)])) * st[l][2]
    print('блок %d: mlp епох %d val %.4f · mlp+ridge епох %d val %.4f · %.0f с' % (b, ep1, r1, ep2, r2, time.time() - t0), flush=True)

print('\nвиграш кроку на відкладених по шарах (SST/SSE): ridge · mlp · mlp+ridge')
for l in range(L):
    sst = sum((STEP[nm][l] ** 2).sum() for nm in names)
    print('%2d ' % l + ' · '.join('%.4f' % (sst / sum(((STEP[nm][l] - D[(nm, l)]) ** 2).sum() for nm in names)) for D in (DVR, DVM, DVMR)))

res = {k: [] for k in ('ridge', 'mlp', 'mlp+ridge')}
for nm in names:
    W = loadW(nm)
    accs = {k: D[(nm, 0)].astype(np.float64).copy() for k, D in zip(res, (DVR, DVM, DVMR))}
    for l in range(1, L):
        Wl = np.asarray(W[l], dtype=np.float64)
        for k, D in zip(res, (DVR, DVM, DVMR)):
            accs[k] = (accs[k] @ Wl) * PHI[nm][l] + D[(nm, l)]
    y0 = (Y[nm][L - 1] ** 2).mean()
    for k in res:
        res[k].append((((Y[nm][L - 1] - accs[k]) ** 2).mean(), y0))
print('\nемуляція бічного потоку, вихід (90 мереж, відкладені блоки):')
R = {k: np.array(v) for k, v in res.items()}
for k in res:
    print('%-9s ×%.4f проти ланцюга без корекції (%d/90)' % (k, R[k][:, 1].mean() / R[k][:, 0].mean(), int((R[k][:, 0] < R[k][:, 1]).sum())))
for k in ('mlp', 'mlp+ridge'):
    print('%-9s проти ridge: агрегат ×%.4f · кращих %d/90 · геом. ×%.4f' % (
        k, R['ridge'][:, 0].mean() / R[k][:, 0].mean(), int((R[k][:, 0] < R['ridge'][:, 0]).sum()),
        math.exp(np.mean(np.log(R['ridge'][:, 0] / R[k][:, 0])))))
