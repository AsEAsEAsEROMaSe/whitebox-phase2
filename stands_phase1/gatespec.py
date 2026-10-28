"""СПЕКТР У КООРДИНАТАХ ЗАТВОРІВ (не входу).

a^L = x . prod_l (W_l diag(g_l))  -- МУЛЬТИЛІНІЙНА по затворах, степінь 1 у кожному шарі.
Ермітова сім'я по x насичується на ~1% (С44). Але це розклад у НЕ ТИХ координатах.

Стара перевірка "усі 20000 патернів затворів унікальні" відповідала на питання
"чи можна затвори ЗАПАМ'ЯТАТИ" (провал хешування) -- це ІНШЕ питання, ніж
"чи спадає спектр по СТЕПЕНІ". Функція може мати всі входи різні й тримати
90% маси на низьких степенях.

МІРЯЄМО (held-out, бо in-sample R2 бреше):
  1. степ.1 по затворах ОДНОГО шару (256 ознак) -- по всіх шарах
  2. накопичувально по кількох шарах
  3. степ.2 всередині шару (випадкові пари) -- чи додає
Порівняння з 1% стелею ермітової сім'ї по входу.
"""
import torch, math, time, numpy as np
LOG = '/mnt/win_d/llm/White-Box/gatespec.log'
T0 = time.time()
def log(m):
    line = '[{:6.1f}s] {}'.format(time.time()-T0, m)
    print(line, flush=True); open(LOG, 'a').write(line+'\n')
open(LOG, 'w').close()

dev = 'cuda:0'; D = 256; L = 32
relu = lambda x: torch.clamp(x, min=0.0)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS = [0, 1, 3, 7, 11, 15, 19, 23, 27, 31]

def fwd(W, n, seed):
    """повертає a^L (n,D) і затвори на TAPS (список (n,D) float32)"""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(n//2, D, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    a = x; G = []
    for li in range(L):
        h = a @ W[li]
        if li in TAPS: G.append((h > 0).float())
        a = relu(h)
    return a, G

def r2_holdout(F, Y, ridge=1e-6):
    """F (n,k) ознаки, Y (n,D) ціль. Половина на фіт, половина на тест.
    R2 = пояснена ПОСЕМПЛОВА дисперсія / посемплова дисперсія (не дисперсія середнього!)"""
    n = F.shape[0]; h = n//2
    F1, F2 = F[:h].double(), F[h:].double()
    Y1, Y2 = Y[:h].double(), Y[h:].double()
    f1 = F1 - F1.mean(0); y1 = Y1 - Y1.mean(0)
    k = F.shape[1]
    S = (f1.t() @ f1)/h + ridge*torch.eye(k, dtype=torch.float64, device=dev)
    b = (f1.t() @ y1)/h
    beta = torch.linalg.solve(S.cpu(), b.cpu()).to(dev)
    f2 = F2 - F1.mean(0); y2 = Y2 - Y1.mean(0)
    resid = y2 - f2 @ beta
    vY = y2.var(0); vR = resid.var(0)
    alive = vY > 1e-9*float(vY.max())
    return float((1 - vR[alive]/vY[alive]).clamp(-1, 1).mean()), int(alive.sum())

N = 200_000
for net in [0, 1]:
    W = [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float32), device=dev)
         for i in range(L)]
    log(''); log('######## НЕТ {}  (N={}, held-out) ########'.format(net, N))
    A, G = fwd(W, N, 5100+net)
    log('  частка відкритих затворів по шарах: ' +
        ' '.join('{:.3f}'.format(float(g.mean())) for g in G))

    log('  --- 1) СТЕПІНЬ 1, затвори ОДНОГО шару (256 ознак) ---')
    best = []
    for i, li in enumerate(TAPS):
        r2, nal = r2_holdout(G[i], A)
        best.append(r2)
        log('     шар {:2d}: R2 = {:+.5f}   (живих нейронів {})'.format(li, r2, nal))

    log('  --- 2) НАКОПИЧУВАЛЬНО (степ.1, кілька шарів разом) ---')
    for take in ([31], [27, 31], [23, 27, 31], [15, 19, 23, 27, 31], TAPS):
        idx = [TAPS.index(t) for t in take]
        F = torch.cat([G[i] for i in idx], 1)
        r2, _ = r2_holdout(F, A)
        log('     шари {:28s} ({:4d} ознак): R2 = {:+.5f}'.format(str(take), F.shape[1], r2))

    log('  --- 3) СТЕПІНЬ 2 всередині шару 31 (випадкові пари) ---')
    gg = torch.Generator(device=dev).manual_seed(77)
    i31 = TAPS.index(31); Gl = G[i31]
    for npair in (1000, 4000):
        ia = torch.randint(0, D, (npair,), device=dev, generator=gg)
        ib = torch.randint(0, D, (npair,), device=dev, generator=gg)
        F2 = torch.cat([Gl, Gl[:, ia]*Gl[:, ib]], 1)
        r2, _ = r2_holdout(F2, A)
        log('     степ.1+2 ({:4d} пар, {:4d} ознак): R2 = {:+.5f}'.format(
            npair, F2.shape[1], r2))
    del A, G
    torch.cuda.empty_cache()
log('')
log('ОРІЄНТИР: ермітова сім'"'"'я по ВХОДУ насичується на ~0.0100 (1%, С44).')
log('done.')
