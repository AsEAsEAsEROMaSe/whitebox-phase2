"""СПЕКТР ЛЯПУНОВА відображення по ГЛИБИНІ (стан = (m, K)).

Гіпотеза (сесія 46): критична ініціалізація 2/256 = точка БІФУРКАЦІЇ.
Похибка замикання не згасає і не вибухає, а накопичується як випадкове блукання
(compound.py: одношагова 1e-4, знаки випадкові, підсилення x6-11, разом sqrt(32)x1e-3).
Це підпис власного числа РІВНО 1 у відображення шар->шар.

ЩО МІРЯЄМО: показники Ляпунова лінеаризації кроку NNGP.
  - усі рівно 0 (множник 1)  -> накопичення неминуче, клас ідей закрито
  - є РОЗШАРУВАННЯ (частина < 0) -> у цих модах похибка ЗГАСАЄ,
    аналітичний якір по них не платить накопиченням -> тонке місце в стіні

Метод: P випадкових симетричних збурень, JVP крізь крок, QR-переортогоналізація
кожні kre шарів, лог росту -> показники.
"""
import torch, math, time, numpy as np
LOG = '/mnt/win_d/llm/White-Box/lyap.log'
T0 = time.time()
def log(m):
    line = '[{:6.1f}s] {}'.format(time.time()-T0, m)
    print(line, flush=True); open(LOG, 'a').write(line+'\n')
open(LOG, 'w').close()

dev = 'cuda:4'; D = 256; L = 32; H = 20
SQ2 = math.sqrt(2.0); SQ2PI = math.sqrt(2*math.pi)
DT = torch.float64                      # fp64: міряємо ЗАНУЛЕННЯ показника
relu = lambda x: torch.clamp(x, min=0.0)
from numpy.polynomial.hermite import hermgauss
xh, wh = hermgauss(H)
xn = torch.tensor(xh*SQ2, dtype=DT, device=dev)
wn = torch.tensor(wh/math.sqrt(math.pi), dtype=DT, device=dev)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi = lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi = lambda t: 0.5*(1+torch.erf(t/SQ2))

def step(m, K, W):
    """один шар NNGP: (m,K) -> (m',K').  диференційовний."""
    mu = W.t() @ m
    C  = W.t() @ (K - torch.outer(m, m)) @ W
    sd = torch.sqrt(torch.diagonal(C).clamp(min=1e-14))
    t  = mu/sd; Ph = Phi(t); ph = phi(t)
    m2 = mu*Ph + sd*ph
    rho = (C/torch.outer(sd, sd)).clamp(-0.999999, 0.999999)
    sq  = torch.sqrt((1-rho*rho).clamp(min=1e-14))
    sdsq = sd[None, :]*sq
    Kg = torch.zeros(D, D, dtype=DT, device=dev)
    for a in range(H):
        Ri  = relu(mu + sd*xn[a])
        gjb = mu[None, :] + sd[None, :]*(rho*xn[a])
        Rj  = relu(gjb[None, :, :] + sdsq[None, :, :]*xn[:, None, None])
        Kg  = Kg + wn[a]*Ri[:, None]*(wn[:, None, None]*Rj).sum(0)
    dg = (mu*mu + sd*sd)*Ph + mu*sd*ph
    Kg = Kg - torch.diag(torch.diagonal(Kg)) + torch.diag(dg)
    return m2, 0.5*(Kg+Kg.t())

def pack(dm, dK):   return torch.cat([dm, dK.reshape(-1)])
def unpack(v):
    dm = v[:D]; dK = v[D:].reshape(D, D)
    return dm, 0.5*(dK+dK.t())

P = 40          # кількість проб
KRE = 1         # переортогоналізація кожен шар (показники по глибині)
for net in [0, 1, 2]:
    W = [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float64), device=dev, dtype=DT)
         for i in range(L)]
    log(''); log('######## НЕТ {} (fp64, P={} проб) ########'.format(net, P))
    m = torch.zeros(D, dtype=DT, device=dev)
    K = torch.eye(D, dtype=DT, device=dev)
    g = torch.Generator(device=dev).manual_seed(4600+net)
    Q = torch.randn(D+D*D, P, dtype=DT, device=dev, generator=g)
    for p in range(P):                        # симетризувати K-частину
        dm, dK = unpack(Q[:, p]); Q[:, p] = pack(dm, dK)
    Q, _ = torch.linalg.qr(Q)
    lsum = torch.zeros(P, dtype=DT, device=dev)
    per_layer = []
    for ell in range(L):
        Wl = W[ell]
        cols = []
        for p in range(P):
            dm, dK = unpack(Q[:, p])
            (_, _), (om, oK) = torch.func.jvp(lambda a, b: step(a, b, Wl), (m, K), (dm, dK))
            cols.append(pack(om, oK))
        Y = torch.stack(cols, 1)
        Qn, R = torch.linalg.qr(Y)
        d = torch.diagonal(R)
        s = torch.sign(d); Qn = Qn*s[None, :]; d = d.abs()
        lsum = lsum + torch.log(d.clamp(min=1e-300))
        per_layer.append(d.cpu().numpy())
        Q = Qn
        m, K = step(m, K, Wl)
        if ell in (0, 7, 15, 23, 31):
            gr = d.cpu().numpy()
            log('  шар {:2d}: множники за КРОК  топ {:.4f}  мед {:.4f}  мін {:.4f}'.format(
                ell, gr[0], float(np.median(gr)), gr.min()))
    lam = (lsum/L).cpu().numpy()              # показник Ляпунова на шар
    mult = np.exp(lam)                        # середній геометр. множник за крок
    log('  ---- ПОКАЗНИКИ (множник за шар, exp(lambda)) ----')
    log('   топ-8 : ' + ' '.join('{:.4f}'.format(v) for v in mult[:8]))
    log('   мед   : {:.4f}    мін: {:.4f}'.format(float(np.median(mult)), mult.min()))
    log('   скільки мод з множником >0.99 : {} / {}'.format(int((mult > 0.99).sum()), P))
    log('   скільки мод з множником <0.90 : {} / {}'.format(int((mult < 0.90).sum()), P))
    log('   ПОВНЕ ЗГАСАННЯ за 32 шари (mult^32): топ {:.3e}  мед {:.3e}  мін {:.3e}'.format(
        mult[0]**L, float(np.median(mult))**L, mult.min()**L))
    np.save('/mnt/win_d/llm/White-Box/lyap_net%d.npy' % net,
            np.stack(per_layer))
log('done.')
