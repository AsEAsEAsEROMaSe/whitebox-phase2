"""РЕЖИМ-ПЕРЕВІРКА CV-ЛІНІЇ: чи був вердикт «0.25%, мертво» артефактом великого N?

ЗНАХІДКА С48 (аудит). Модель усадки, яку вживають tap_tradeoff.py / shrink_cv.py:
    MSE(λ)/V₀ = 1 − λ(2−λ)R² + λ²·b ,   b = зсув²/V₀
    λ* = R²/(R²+b) ,   ВИГРАШ = R⁴/(R²+b)
Формула відтворює їхні логи: «усі 32» R²=0.5118, λ=0.005 ⟹ b=101.9 ⟹ 0.256% (виміряно 0.25%);
«ранні 0-7» R²=0.228, λ=0.004 ⟹ b=56.8 ⟹ 0.091% (виміряно 0.10%).

🔑 АЛЕ `b = зсув²/V₀`, де зсув якоря СИСТЕМАТИЧНИЙ (якір АНАЛІТИЧНИЙ — nngp_tr рахує сліди
з ваг, без семплів ⟹ від N не залежить), а V₀ ∝ 1/N.  ⟹ **b ∝ N**.
Стенди міряли при N=200000. Продакшн ≈ 6500 (арбітр: MSE 2.6e-6 проти 2.586e-7 тут) ⟹ b удесятеро
менше ⟹ виграш має бути ~10× більший. Той самий тип пастки, що «−29% при N=20k, +4.1% при N=200k».

ЧАСТИНА A: перерахунок формули по N на ТОЧНО оцінених R², β, derr (вони від N не залежать).
ЧАСТИНА B: ПРЯМИЙ емпіричний прогін при продакшн-N — бо формула це модель, не доказ.
   ⚠️ додається ефект, якого в моделі НЕМА: при малому N сам β оцінюється шумно (F/N).
   Тому β беремо з ОКРЕМОЇ навчальної вибірки, а потім МІРЯЄМО обидва варіанти:
     (i) β з великої навчальної (ідеалізація, збігається з моделлю)
     (ii) β оцінений НА ТИХ САМИХ N семплах (чесний продакшн) ⟸ вирішальний
   ⚠️ ЕТАЛОН мусить бути ТОЧНІШИЙ за метод (урок r411/r419): truth з 3M проти N=6500 ⟹ 460×.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG = '/mnt/win_d/llm/White-Box/tap_regime.log'; T0 = time.time()
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
relu = torch.relu

def nngp_tr(W):
    """АНАЛІТИЧНИЙ якір: сліди Tr(K^ℓ) з ваг, БЕЗ семплів"""
    m = torch.zeros(D, dtype=DT, device=dev); K = torch.eye(D, dtype=DT, device=dev); tr = []
    for ell in range(L):
        mu = W[ell].t() @ m; C = W[ell].t() @ (K - torch.outer(m, m)) @ W[ell]
        sd = torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t = mu / sd; Ph = Phi(t); ph = phi(t)
        m = mu * Ph + sd * ph
        rho = (C / torch.outer(sd, sd)).clamp(-0.999999, 0.999999)
        sq = torch.sqrt((1 - rho * rho).clamp(min=1e-14))
        Kg = torch.zeros(D, D, dtype=DT, device=dev); sdsq = sd[None, :] * sq
        for a in range(H):
            Ri = relu(mu + sd * xn[a]); gjb = mu[None, :] + sd[None, :] * (rho * xn[a])
            Rj = relu(gjb[None, :, :] + sdsq[None, :, :] * xn[:, None, None])
            Kg = Kg + wn[a] * Ri[:, None] * (wn[:, None, None] * Rj).sum(0)
        dg = (mu * mu + sd * sd) * Ph + mu * sd * ph
        Kg = Kg - torch.diag(torch.diagonal(Kg)) + torch.diag(dg); K = Kg
        tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

def sample(W, npair, seed):
    """антитетична вибірка -> (a^L, ознаки ‖a^ℓ‖²)"""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, D, device=dev, generator=g); x = torch.cat([z, -z], 0)
    a = x; ts = []
    for li in range(L):
        a = relu(a @ W[li]); ts.append((a * a).sum(1))
    return a.double(), torch.stack(ts, 1).double()

SETS = {'ранні 0-7': list(range(0, 8)), 'середні 12-19': list(range(12, 20)),
        'пізні 24-31': list(range(24, 32)), 'чинні 8': [3, 7, 11, 15, 19, 23, 27, 30],
        'усі 32': list(range(L))}
NS = [6500, 13000, 26000, 65000, 200000]

for net in (0, 1):
    W = [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float32), device=dev) for i in range(L)]
    log('=' * 100); log('######## НЕТ %d ########' % net)
    trn = nngp_tr([w.double() for w in W])

    # ---- ЕТАЛОН + істинні сліди (3M, тобто 460× над продакшн-N) ----
    SA = torch.zeros(D, dtype=DT, device=dev); ST = torch.zeros(L, dtype=DT, device=dev); nt = 0
    for c in range(60):
        a, t = sample(W, 25000, 3300 + net * 29 + c)
        SA += a.sum(0); ST += t.sum(0); nt += a.shape[0]
    truth = SA / nt; trt = (ST / nt).cpu().numpy()
    derr_all = trn - trt
    log('  еталон з %d семплів (=%.0f× над продакшн 6500)' % (nt, nt / 6500))

    # ---- НАВЧАЛЬНА вибірка для β (окрема від оцінювальних) ----
    A = []; T = []
    for c in range(8):
        a, t = sample(W, 12500, 7700 + net * 23 + c)
        A.append(a); T.append(t)
    A = torch.cat(A); T = torch.cat(T); ntr = A.shape[0]
    mA = A.mean(0); mT = T.mean(0)
    vA = ((A - mA) ** 2).mean(0); alive = vA > 1e-6 * float(vA.max())

    def fit(Tm, Am, idx):
        f1 = Tm[:, idx] - Tm[:, idx].mean(0); a1 = Am - Am.mean(0)
        S = (f1.t() @ f1) / f1.shape[0]; b = (f1.t() @ a1) / f1.shape[0]
        sd = torch.diagonal(S).clamp(min=1e-300).sqrt()
        Sn = S / torch.outer(sd, sd) + 1e-8 * torch.eye(len(idx), dtype=DT, device=dev)
        return (torch.linalg.solve(Sn.cpu(), (b / sd[:, None]).cpu()).to(dev)) / sd[:, None]

    # ---- ЧАСТИНА A: формула по N ----
    log('  A) ФОРМУЛА виграш = R⁴/(R²+b),  b = зсув²/V₀,  V₀ = Var/N  ⟹ b ∝ N')
    log('     {:14s} {:>8s} {:>11s} | {}'.format('набір', 'R²', 'зсув²',
        ' '.join('{:>9s}'.format('N='+str(x)) for x in NS)))
    STORE = {}
    for nm, idx in SETS.items():
        beta = fit(T, A, idx)
        # held-out R² на СВІЖІЙ вибірці
        ah, th = sample(W, 25000, 990000 + net * 7)
        r = (ah - mA) - (th[:, idx] - mT[idx]) @ beta
        r2 = float((1 - r.var(0) / (ah - mA).var(0).clamp(min=1e-30))[alive].mean())
        derr = torch.tensor(derr_all[idx], dtype=DT, device=dev)
        Bs = float((((beta * derr[:, None]).sum(0))[alive] ** 2).mean())
        gains = []
        for N in NS:
            V0 = float((vA[alive] / N).mean()); b = Bs / V0
            gains.append(r2 * r2 / (r2 + b))
        STORE[nm] = (idx, beta, r2, Bs)
        log('     {:14s} {:+8.4f} {:11.3e} | {}'.format(nm, r2, Bs,
            ' '.join('{:8.2f}%'.format(100 * g) for g in gains)))
    V0p = float((vA[alive] / 6500).mean())
    log('     стеля при R²=1: N=6500 → {:.2f}%   N=200000 → {:.2f}%   (найкращий якір «ранні»)'.format(
        100 / (1 + STORE['ранні 0-7'][3] / V0p),
        100 / (1 + STORE['ранні 0-7'][3] / float((vA[alive] / 200000).mean()))))

    # ---- ЧАСТИНА B: ПРЯМИЙ прогін при продакшн-N ----
    log('  B) ПРЯМИЙ ЕМПІРИЧНИЙ прогін, N=6500, 120 повторів (β з навчальної vs β на місці)')
    NREP = 120; NPAIR = 3250
    for nm in ('ранні 0-7', 'чинні 8', 'усі 32'):
        idx, beta, r2, Bs = STORE[nm]
        anc = torch.tensor(trn[idx], dtype=DT, device=dev)
        V0 = float((vA[alive] / 6500).mean()); lam = (r2 / (r2 + Bs / V0))
        e_pl = []; e_tr = []; e_lo = []
        for rp in range(NREP):
            a, t = sample(W, NPAIR, 4_100_000 + net * 9973 + rp)
            pl = a.mean(0); fb = t[:, idx].mean(0) - anc
            tr_ = pl - lam * (fb @ beta)                       # (i) β з навчальної
            bl = fit(t, a, idx)                                 # (ii) β НА ТИХ САМИХ семплах
            lo = pl - lam * (fb @ bl)
            e_pl.append(float(((pl - truth)[alive] ** 2).mean()))
            e_tr.append(float(((tr_ - truth)[alive] ** 2).mean()))
            e_lo.append(float(((lo - truth)[alive] ** 2).mean()))
        mp = sum(e_pl) / NREP; mt = sum(e_tr) / NREP; ml = sum(e_lo) / NREP
        log('     {:14s} λ={:.3f} | база {:.4e} | β-навч {:.4e} ({:+.2f}%) | β-на-місці {:.4e} ({:+.2f}%)'
            .format(nm, lam, mp, mt, 100 * (mt / mp - 1), ml, 100 * (ml / mp - 1)))
    del A, T; torch.cuda.empty_cache()
log('done.')
