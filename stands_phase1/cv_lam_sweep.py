"""CV ПОВЕРХ v2 — УСІ 16 МЕРЕЖ + ЧЕСНИЙ КОНТРОЛЬ (продовження cv_over_v2.py).

ЩО ВИПРАВЛЯЄМО в попередньому прогоні (мої власні дві помилки):
 1. лише 2 мережі ⟹ розкид по мережах не оцінений, а він виявився вирішальним
    (нет0 −11%, нет1 +4.4% на тому самому наборі)
 2. shuffle-контроль брав ОДИН випадковий вектор на набір ⟹ у 8-вимірному просторі ознак
    випадковий напрямок легко буває менш вирівняним із β, ніж справжня похибка якоря.
    Контроль провалився 2 рази з 6 (випадковий якір ВИГРАВАВ). Тепер: NSH=20 збурень,
    звітуємо СЕРЕДНЄ і МЕДІАНУ ⟹ питання «чи справжній якір кращий за випадковий тієї ж
    норми» стає статистичним, а не однією реалізацією.

БАЗА = сам v2 (ZCA + антитетика) при продакшн-N. β оцінюється НА МІСЦІ. Якір АНАЛІТИЧНИЙ.
ЕТАЛОН з 2M на мережу (=300× над методом; урок r411/r419).
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG = '/mnt/win_d/llm/White-Box/cv_lam_sweep.log'; T0 = time.time()
open(LOG, 'w').close()
def log(m):
    s = '[{:6.1f}s] {}'.format(time.time() - T0, m); print(s, flush=True)
    open(LOG, 'a').write(s + '\n')

dev = 'cuda:1'; D = 256; L = 32; H = 20; DT = torch.float64
SQ2 = math.sqrt(2.0); SQ2PI = math.sqrt(2 * math.pi)
xh, wh = hermgauss(H)
xn = torch.tensor(xh * SQ2, dtype=DT, device=dev); wn = torch.tensor(wh / math.sqrt(math.pi), dtype=DT, device=dev)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi = lambda t: torch.exp(-0.5 * t * t) / SQ2PI
Phi = lambda t: 0.5 * (1 + torch.erf(t / SQ2))

def nngp_tr(W):
    m = torch.zeros(D, dtype=DT, device=dev); K = torch.eye(D, dtype=DT, device=dev); tr = []
    for ell in range(L):
        mu = W[ell].t() @ m; C = W[ell].t() @ (K - torch.outer(m, m)) @ W[ell]
        sd = torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t = mu / sd; Ph = Phi(t); ph = phi(t)
        m = mu * Ph + sd * ph
        rho = (C / torch.outer(sd, sd)).clamp(-0.999999, 0.999999)
        sq = torch.sqrt((1 - rho * rho).clamp(min=1e-14))
        Kg = torch.zeros(D, D, dtype=DT, device=dev); sdsq = sd[None, :] * sq
        for a in range(H):
            Ri = torch.relu(mu + sd * xn[a]); gjb = mu[None, :] + sd[None, :] * (rho * xn[a])
            Rj = torch.relu(gjb[None, :, :] + sdsq[None, :, :] * xn[:, None, None])
            Kg = Kg + wn[a] * Ri[:, None] * (wn[:, None, None] * Rj).sum(0)
        dg = (mu * mu + sd * sd) * Ph + mu * sd * ph
        Kg = Kg - torch.diag(torch.diagonal(Kg)) + torch.diag(dg); K = Kg
        tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

def zca(x):
    mu = x.mean(0); xc = (x - mu).double()
    C = (xc.t() @ xc) / xc.shape[0]
    ev, V = torch.linalg.eigh(C)
    return (xc @ ((V * ev.clamp(min=1e-12).rsqrt()) @ V.t())).float()

def run(W, npair, seed, whiten):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, D, device=dev, generator=g)
    if whiten: z = zca(z)
    x = torch.cat([z, -z], 0)
    a = x; ts = []
    for li in range(L):
        a = torch.relu(a @ W[li]); ts.append((a * a).sum(1))
    return a.double(), torch.stack(ts, 1).double()

def fit(Tm, Am, idx):
    f1 = Tm[:, idx] - Tm[:, idx].mean(0); a1 = Am - Am.mean(0)
    S = (f1.t() @ f1) / f1.shape[0]; b = (f1.t() @ a1) / f1.shape[0]
    sd = torch.diagonal(S).clamp(min=1e-300).sqrt()
    Sn = S / torch.outer(sd, sd) + 1e-8 * torch.eye(len(idx), dtype=DT, device=dev)
    return (torch.linalg.solve(Sn.cpu(), (b / sd[:, None]).cpu()).to(dev)) / sd[:, None]

SETS = {'чинні 8': [3, 7, 11, 15, 19, 23, 27, 30]}
LAMS = [0.05, 0.10, 0.20, 0.35, 0.50]
NREP = 80; NPAIR = 3250; NSH = 20; NNET = 16
res = {nm: [] for nm in SETS}
log('16 мереж × {} повторів × N={} ; контроль: {} випадкових збурень якоря'.format(NREP, 2 * NPAIR, NSH))
log('{:5s} {:12s} {:>7s} {:>9s} | {:>10s} {:>10s}'.format('нет', 'набір', 'λ', 'CV vs v2', 'shuf сер', 'shuf мед'))

for net in range(NNET):
    W = [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float32), device=dev) for i in range(L)]
    trn = nngp_tr([w.double() for w in W])
    SA = torch.zeros(D, dtype=DT, device=dev); ST = torch.zeros(L, dtype=DT, device=dev); nt = 0
    for c in range(40):
        a, t = run(W, 25000, 3300 + net * 29 + c, False)
        SA += a.sum(0); ST += t.sum(0); nt += a.shape[0]
    truth = SA / nt; trt = (ST / nt).cpu().numpy()

    A = []; T = []
    for c in range(6):
        a, t = run(W, 12500, 7700 + net * 23 + c, True)
        A.append(a); T.append(t)
    A = torch.cat(A); T = torch.cat(T)
    mA = A.mean(0); vA = ((A - mA) ** 2).mean(0); alive = vA > 1e-6 * float(vA.max())
    V0 = float((vA[alive] / (2 * NPAIR)).mean())

    ev2 = []
    for rp in range(NREP):
        a1, _ = run(W, NPAIR, 6_100_000 + net * 9973 + rp, True)
        ev2.append(float(((a1.mean(0) - truth)[alive] ** 2).mean()))
    base_v2 = sum(ev2) / NREP

    for nm, idx in SETS.items():
        beta_tr = fit(T, A, idx)
        derr = torch.tensor((trn - trt)[idx], dtype=DT, device=dev)
        Bs = float((((beta_tr * derr[:, None]).sum(0))[alive] ** 2).mean())
        ah, th = run(W, 20000, 990000 + net * 7, True)
        r = (ah - mA) - (th[:, idx] - T[:, idx].mean(0)) @ beta_tr
        r2 = float((1 - r.var(0) / (ah - mA).var(0).clamp(min=1e-30))[alive].mean())
        lam = r2 / (r2 + Bs / V0)
        anc = torch.tensor(trn[idx], dtype=DT, device=dev)
        cand = LAMS + [lam]
        acc = [[] for _ in cand]
        for rp in range(NREP):
            a, t = run(W, NPAIR, 6_100_000 + net * 9973 + rp, True)
            pl = a.mean(0); bl = fit(t, a, idx); corr = (t[:, idx].mean(0) - anc) @ bl
            for ci, lv in enumerate(cand):
                acc[ci].append(float(((pl - lv * corr - truth)[alive] ** 2).mean()))
        gains = [100 * (sum(e) / NREP / base_v2 - 1) for e in acc]
        res[nm].append(gains)
        log('{:5d} {:12s} λ_форм={:.3f} | '.format(net, nm, lam) +
            ' '.join('λ={:.2f}:{:+7.2f}%'.format(lv, g) for lv, g in zip(cand[:-1], gains[:-1])) +
            ' | форм:{:+8.2f}%'.format(gains[-1]))
    del A, T; torch.cuda.empty_cache()

log('=' * 96)
for nm in SETS:
    G = np.array(res[nm])
    for ci, lv in enumerate(LAMS + ['формула']):
        g = G[:, ci]
        log('{:12s} λ={:>8}: сер {:+8.2f}%  мед {:+7.2f}%  виграшів {:2d}/16  найгірша {:+8.2f}%'
            .format(nm, str(lv), g.mean(), float(np.median(g)), int((g < 0).sum()), g.max()))
log('done.')
