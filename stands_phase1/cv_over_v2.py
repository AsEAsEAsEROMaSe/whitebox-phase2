"""ВИРІШАЛЬНИЙ ТЕСТ: чи виживає CV від норм ‖a^ℓ‖² ПОВЕРХ ПОВНОГО v2 при продакшн-N?

ПЕРЕДІСТОРІЯ (С48). `tap_regime.py` показав: при N=6500 CV проти АНАЛІТИЧНОГО якоря дає
−31.4%/−25.3%/−12.0% (нет0) і −4.6%/−5.8%/−9.5% (нет1) — проти 0.10-0.25%, залогованих
при N=200000. Причина: b = зсув²/V₀, V₀ ∝ 1/N ⟹ b ∝ N ⟹ зміщена поправка виглядає тим
гіршою, чим більший N на стенді.

АЛЕ база там — ЧИСТА АНТИТЕТИКА (6.78e-6). v2 = антитетика + ZCA, і ZCA сама дає ~−48%.
Норми ‖a^ℓ‖² цілком можуть міряти те саме, що вже забирає вибілювання.
Рівно так загинув `arcsine_full.py`: ізольовано −2.1%, поверх ZCA +4…+12%.
І `combine.py` (+12.21%) теж міряв при N=200000 ⟹ його вердикт у тій самій пастці.

ТЕСТ: CV поверх ПОВНОГО v2 при N=6500, β оцінюється НА МІСЦІ, якір АНАЛІТИЧНИЙ (nngp_tr).
БАЗА = сам v2. Контролі:
  1. плоска антитетика (щоб бачити, скільки дає ZCA — очікуємо ~×0.5)
  2. λ=0 тотожно повертає базу (перевірка проводки)
  3. ЕТАЛОН з 3M = 460× над методом (урок r411/r419: еталон мусить бути точніший)
  4. shuffle-контроль: якір, зсунутий на випадковий вектор тієї ж норми ⟹ виграш має ЗНИКНУТИ
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG = '/mnt/win_d/llm/White-Box/cv_over_v2.log'; T0 = time.time()
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
    """ZCA-вибілювання вибірки (робить вибіркові моменти 1-2 точними) — як у v2"""
    mu = x.mean(0); xc = x - mu
    C = (xc.t() @ xc) / xc.shape[0]
    ev, V = torch.linalg.eigh(C.double())
    Wt = (V * ev.clamp(min=1e-12).rsqrt()) @ V.t()
    return (xc.double() @ Wt).float()

def run(W, npair, seed, whiten):
    """повертає (a^L, ознаки ‖a^ℓ‖²) для v2-подібної (whiten=True) або плоскої вибірки"""
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

SETS = {'ранні 0-7': list(range(0, 8)), 'чинні 8': [3, 7, 11, 15, 19, 23, 27, 30],
        'усі 32': list(range(L))}
NREP = 120; NPAIR = 3250

for net in (0, 1):
    W = [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float32), device=dev) for i in range(L)]
    log('=' * 104); log('######## НЕТ %d ########' % net)
    trn = nngp_tr([w.double() for w in W])

    SA = torch.zeros(D, dtype=DT, device=dev); ST = torch.zeros(L, dtype=DT, device=dev); nt = 0
    for c in range(60):
        a, t = run(W, 25000, 3300 + net * 29 + c, False)
        SA += a.sum(0); ST += t.sum(0); nt += a.shape[0]
    truth = SA / nt; trt = (ST / nt).cpu().numpy()
    vA = None

    # навчальна (лише щоб оцінити R²/λ; сам метод β бере НА МІСЦІ)
    A = []; T = []
    for c in range(8):
        a, t = run(W, 12500, 7700 + net * 23 + c, True)
        A.append(a); T.append(t)
    A = torch.cat(A); T = torch.cat(T)
    mA = A.mean(0); vA = ((A - mA) ** 2).mean(0); alive = vA > 1e-6 * float(vA.max())
    V0 = float((vA[alive] / (2 * NPAIR)).mean())
    log('  еталон 3M (=%.0f× над методом);  живих нейронів %d/256' % (nt / (2 * NPAIR), int(alive.sum())))

    # ---- база: плоска антитетика проти v2 ----
    ep = []; ev2 = []
    for rp in range(NREP):
        a0, _ = run(W, NPAIR, 6_100_000 + net * 9973 + rp, False)
        a1, _ = run(W, NPAIR, 6_100_000 + net * 9973 + rp, True)
        ep.append(float(((a0.mean(0) - truth)[alive] ** 2).mean()))
        ev2.append(float(((a1.mean(0) - truth)[alive] ** 2).mean()))
    base_flat = sum(ep) / NREP; base_v2 = sum(ev2) / NREP
    log('  КОНТРОЛЬ 1: плоска антитетика {:.4e} | v2 (ZCA+анти) {:.4e}  ⟹ ZCA дає {:+.1f}%'
        .format(base_flat, base_v2, 100 * (base_v2 / base_flat - 1)))

    log('  {:12s} {:>7s} | {:>11s} {:>9s} | {:>11s} {:>9s}'
        .format('набір', 'λ', 'CV поверх v2', 'зміна', 'shuffle-як', 'зміна'))
    for nm, idx in SETS.items():
        beta_tr = fit(T, A, idx)
        derr = torch.tensor((trn - trt)[idx], dtype=DT, device=dev)
        Bs = float((((beta_tr * derr[:, None]).sum(0))[alive] ** 2).mean())
        ah, th = run(W, 25000, 990000 + net * 7, True)
        r = (ah - mA) - (th[:, idx] - T[:, idx].mean(0)) @ beta_tr
        r2 = float((1 - r.var(0) / (ah - mA).var(0).clamp(min=1e-30))[alive].mean())
        lam = r2 / (r2 + Bs / V0)
        anc = torch.tensor(trn[idx], dtype=DT, device=dev)
        # shuffle-контроль: якір зсунуто випадковим вектором тієї ж норми
        gsh = torch.Generator(device=dev).manual_seed(555 + net)
        pert = torch.randn(len(idx), device=dev, generator=gsh, dtype=DT)
        anc_sh = anc + pert / pert.norm() * derr.norm()
        ecv = []; esh = []; ez = []
        for rp in range(NREP):
            a, t = run(W, NPAIR, 6_100_000 + net * 9973 + rp, True)
            pl = a.mean(0); bl = fit(t, a, idx)
            fb = t[:, idx].mean(0) - anc
            ecv.append(float(((pl - lam * (fb @ bl) - truth)[alive] ** 2).mean()))
            fs = t[:, idx].mean(0) - anc_sh
            esh.append(float(((pl - lam * (fs @ bl) - truth)[alive] ** 2).mean()))
            ez.append(float(((pl - 0.0 * (fb @ bl) - truth)[alive] ** 2).mean()))
        mc = sum(ecv) / NREP; ms = sum(esh) / NREP; mz = sum(ez) / NREP
        assert abs(mz / base_v2 - 1) < 1e-9, 'КОНТРОЛЬ 2 (λ=0) ПРОВАЛЕНО'
        log('  {:12s} {:7.3f} | {:11.4e} {:+8.2f}% | {:11.4e} {:+8.2f}%'
            .format(nm, lam, mc, 100 * (mc / base_v2 - 1), ms, 100 * (ms / base_v2 - 1)))
    del A, T; torch.cuda.empty_cache()
log('КОНТРОЛЬ 2 (λ=0 ≡ база) пройдено в усіх рядках.')
log('done.')
