"""ЧИ СПРАВЖНІЙ НИЗЬКИЙ РАНГ ПОПРАВКИ ЗАМИКАННЯ? (аудит С48)

probe_closure_rank.py звітує PR(dM)=7.7/256 на шарі 31 і два майже рівні верхні власні
значення (1.000, 0.937) ⟹ «поправка майже рангу 2». ТРИ ПІДОЗРИ, жодна не перевірена:

  A) ДЗВІН. a^L>=0 + однорідність ⟹ головна вісь будь-якої матриці других моментів лягає
     на середнє. Рангу-2 могло б бути просто mu(x)mu^T. Перевірка: cos(верхні в.вектори, mu_a).
  B) КВАДРАТУРА. M_gauss рахується Гауссом-Ермітом H=24 у fp32 ЧЕРЕЗ ЗЛАМ ReLU — та сама
     сім'я багів, що в analytic_K.py. Похибка квадратури систематична й структурована
     ⟹ могла б сама бути низькорангова. Перевірка: H=24/fp32 проти H=64/fp64.
  C) MC-ШУМ. |dM|/|M| = 0.0017 при N=2e6, а 1/sqrt(N) = 7.1e-4 ⟹ сигнал лише ~2.4x над
     шумом. Перевірка: розділити вибірку навпіл, dM_noise=(M_A-M_B)/2 — прямий вимір підлоги.

ЧИТАННЯ. Якщо cos~0.99 -> це дзвін, гілка закрита. Якщо |dM| падає при кращій квадратурі ->
ранг був артефактом інтегрування. Якщо |dM| ~ |dM_noise| -> міряли власний шум.
Вижити ранг-2 може лише переживши всі три.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG = '/mnt/win_d/llm/White-Box/closure_rank_null.log'; T0 = time.time()
open(LOG, 'w').close()
def log(m):
    s = '[{:6.1f}s] {}'.format(time.time() - T0, m); print(s, flush=True)
    open(LOG, 'a').write(s + '\n')

dev = 'cuda:1'; D = 256; L = 32
d = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def quad(H, dt):
    xh, wh = hermgauss(H)
    return (torch.tensor(xh * math.sqrt(2), dtype=dt, device=dev),
            torch.tensor(wh / math.sqrt(math.pi), dtype=dt, device=dev))

def gauss_M(mu, C, H, dt):
    """E[relu(z_i) relu(z_j)] для гауса (mu,C) — квадратура H вузлів у типі dt"""
    xn, wn = quad(H, dt)
    mu = mu.to(dt); C = C.to(dt)
    sd = torch.sqrt(torch.diagonal(C).clamp(min=1e-12))
    rho = (C / torch.outer(sd, sd)).clamp(-0.999999, 0.999999)
    sq = torch.sqrt(1 - rho * rho); sdsq = sd[None, :] * sq
    M = torch.zeros(D, D, dtype=dt, device=dev)
    for a in range(H):
        Ri = torch.clamp(mu + sd * xn[a], min=0)
        gjb = mu[None, :] + sd[None, :] * (rho * xn[a])
        Rj = torch.clamp(gjb[None, :, :] + sdsq[None, :, :] * xn[:, None, None], min=0)
        M += wn[a] * Ri[:, None] * (wn[:, None, None] * Rj).sum(0)
    return M.double()

def spec(M):
    Ms = 0.5 * (M + M.t()).cpu()
    ev, V = torch.linalg.eigh(Ms)
    o = ev.abs().sort(descending=True).indices
    a = ev[o].abs()
    pr = float(a.sum() ** 2 / (a ** 2).sum())
    return pr, a, V[:, o]

for net in (0, 1):
    W = [torch.tensor(d['w_%d_%d' % (net, i)].astype(np.float32), device=dev) for i in range(L)]
    for ell in (16, 24, 31):
        log('=' * 96); log('МЕРЕЖА %d  ШАР %d' % (net, ell))
        N, CH = 2_000_000, 250_000
        acc = [dict(s1=torch.zeros(D, dtype=torch.float64, device=dev),
                    Szz=torch.zeros(D, D, dtype=torch.float64, device=dev),
                    r1=torch.zeros(D, dtype=torch.float64, device=dev),
                    Mrr=torch.zeros(D, D, dtype=torch.float64, device=dev), n=0) for _ in range(2)]
        for c in range(N // CH):
            z0 = torch.randn(CH // 2, D, device=dev, generator=torch.Generator(device=dev).manual_seed(1000 + c))
            x = torch.cat([z0, -z0], 0); zc = None
            for li in range(ell):
                h = x @ W[li]
                if li < ell - 1: x = torch.relu(h)
                else: zc = h
            A = acc[c % 2]                                   # половини для контролю шуму
            zd = zc.double(); a_ = torch.relu(zd)
            A['s1'] += zd.sum(0); A['Szz'] += zd.t() @ zd
            A['r1'] += a_.sum(0); A['Mrr'] += a_.t() @ a_; A['n'] += zc.shape[0]

        nt = acc[0]['n'] + acc[1]['n']
        mu = (acc[0]['s1'] + acc[1]['s1']) / nt
        C = (acc[0]['Szz'] + acc[1]['Szz']) / nt - torch.outer(mu, mu)
        M_true = (acc[0]['Mrr'] + acc[1]['Mrr']) / nt
        mu_a = (acc[0]['r1'] + acc[1]['r1']) / nt            # E[relu(z)] — «середнє»
        MA = acc[0]['Mrr'] / acc[0]['n']; MB = acc[1]['Mrr'] / acc[1]['n']

        fro = lambda M: float(torch.linalg.norm(M))
        fM = fro(M_true)
        Mg24 = gauss_M(mu, C, 24, torch.float32)
        Mg64 = gauss_M(mu, C, 64, torch.float64)
        dM24 = M_true - Mg24; dM64 = M_true - Mg64
        dN = (MA - MB) * 0.5                                  # MC-шум на повній вибірці

        p24, a24, V24 = spec(dM24)
        p64, a64, V64 = spec(dM64)
        pN, aN, _ = spec(dN)

        u = (mu_a / mu_a.norm()).cpu()
        cos = lambda V, k: abs(float(V[:, k] @ u))

        log('  |dM|/|M|      H=24 fp32 : {:.5f}    H=64 fp64 : {:.5f}'.format(fro(dM24) / fM, fro(dM64) / fM))
        log('  |MC-ШУМ|/|M|            : {:.5f}   ⟸ ПІДЛОГА (півсплітом)'.format(fro(dN) / fM))
        log('  |Mg24 − Mg64|/|M|       : {:.5f}   ⟸ похибка квадратури'.format(fro(Mg24 - Mg64) / fM))
        log('  PR:  dM(H24) {:5.1f}   dM(H64) {:5.1f}   MC-шум {:5.1f}'.format(p24, p64, pN))
        log('  топ-4 |eig| dM(H64): ' + ' '.join('{:.3f}'.format(float(v / a64[0])) for v in a64[:4]))
        log('  топ-4 |eig| MC-шум : ' + ' '.join('{:.3f}'.format(float(v / aN[0])) for v in aN[:4]))
        log('  🔔 cos(в.вектор dM(H64), mu_a):  #1 = {:.4f}   #2 = {:.4f}   #3 = {:.4f}'
            .format(cos(V64, 0), cos(V64, 1), cos(V64, 2)))
        log('     (контроль: cos випадкового вектора з mu_a ≈ 1/√256 = 0.0625)')
log('done.')
