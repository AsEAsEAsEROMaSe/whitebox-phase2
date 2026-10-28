"""КОНТРОЛЬНА ЗМІННА З ЯДРА АРККОСИНУСА — ознаки зі ЗНАЧЕНЬ шару 1, не зі знаків.

ЧОМУ ЦЕ НОВЕ:
  • [[finding-arcsine-family-mostly-zca-shadow]] брала sign(x·W₀) — лише ЗНАК, 1% понад ZCA.
    Значення a¹=relu(x·W₀) несуть ще й АМПЛІТУДУ.
  • середнє відоме ТОЧНО лише з ваг (ядро арккосинуса):
        E[a¹_i a¹_j] = (σ_iσ_j/2π)(sinθ + (π−θ)cosθ),  θ=arccos(ρ), ρ=K0/(σσᵀ), K0=W₀ᵀW₀
    ⟹ якір БЕЗКОШТОВНИЙ, [[finding-anchor-cost-identity]] не діє
  • relu не поліном ⟹ сім'я не лежить у ZCA цілком
  • ціна ~нуль: a¹ і так у форварді; проєкція a¹@V це 0.4% флопів

ОЗНАКА: p_k(x) = (a¹·v_k)² − v_kᵀ M v_k.  Середнє = 0 ТОЧНО.

ВИМІР ПРОДАКШН-МАТЧЕНИЙ ([[feedback-verify-before-failing-an-idea]]):
не стенд зі «зняттям квадратики», а СПРАВЖНІЙ v2 — антитетика + ZCA, N=6484, R повторів,
MSE з CV і без. ZCA реально застосований ⟹ «що стенд прибрав» не виникає.

КОНТРОЛІ:
  1) ПОЗИТИВНИЙ на M: точна матриця проти MC (норма Фробеніуса, не max-відносна —
     у M є елементи ≈0 при ρ→−1, там відносна похибка вибухає від шуму)
  2) ВИПАДКОВІ напрямки замість оптимальних
  3) β двома способами: ОРАКУЛ і IN-SAMPLE (як у продакшні)
  4) ДІАГНОСТИКА ЗСУВУ: ZCA-вибілена вибірка не строго N(0,I) ⟹ якір може «поїхати»;
     міряємо |серед.(est−TH)|² окремо від MSE
⚠️ V (напрямки) ОРАКУЛЬНІ — цей прогін встановлює ПОТЕНЦІАЛ, не метод.
"""
import torch, math, numpy as np, time

dev = 'cuda:4'; D = 256; L = 32; T0 = time.time()
NPAIRS = 3242          # продакшн: 6484 прогони
REPS = 80
NBIG = 120000          # велика вибірка для ОРАКУЛЬНИХ V і β
CH = 20000
CFG = [('оптим', k) for k in (1, 4, 16, 64)] + [('вип.', k) for k in (4, 64)]

def log(m): print('[{:6.1f}s] {}'.format(time.time() - T0, m), flush=True)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d' % (n, i)].astype(np.float32), device=dev) for i in range(L)]

def fwd_from_a1(W, a1):
    a = a1
    for i in range(1, L): a = torch.relu(a @ W[i])
    return a

def arccos_M(W0):
    """ТОЧНА E[relu(h)relu(h)ᵀ] для h = x·W0, x~N(0,I)."""
    K0 = (W0.t() @ W0).double()
    s = K0.diagonal().clamp(min=1e-30).sqrt()
    rho = (K0 / torch.outer(s, s)).clamp(-1.0, 1.0)
    th = torch.arccos(rho)
    return torch.outer(s, s) / (2 * math.pi) * (torch.sin(th) + (math.pi - th) * torch.cos(th))

def zca(x):
    C = (x.t() @ x) / x.shape[0]
    e, V = torch.linalg.eigh(C.double())
    return (V @ torch.diag(e.clamp(min=1e-8).rsqrt()) @ V.t()).float()

def feats(a1, Vk, mu_lin, mu_quad):
    pr = a1.double() @ Vk
    return torch.cat([pr - mu_lin.unsqueeze(0), pr * pr - mu_quad.unsqueeze(0)], 1)

for net in (0, 1):
    W = load(net); log('=' * 88); log('МЕРЕЖА %d' % net)

    # ---------- істина (2e6 семплів ⟹ власна похибка ~8e-9, знехтувана) ----------
    S = torch.zeros(D, dtype=torch.float64, device=dev); n = 0
    for c in range(10):
        g = torch.Generator(device=dev).manual_seed(990000 + c)
        z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
        S += fwd_from_a1(W, torch.relu(xx @ W[0])).double().sum(0); n += xx.shape[0]
    TH = S / n

    # ---------- КОНТРОЛЬ 1: точна M проти MC ----------
    M = arccos_M(W[0])
    MU1 = ((W[0].t() @ W[0]).double().diagonal().clamp(min=1e-30).sqrt()) / math.sqrt(2 * math.pi)
    Sm = torch.zeros(D, D, dtype=torch.float64, device=dev); m = 0
    for c in range(10):
        g = torch.Generator(device=dev).manual_seed(220000 + c)
        z = torch.randn(20000, D, device=dev, generator=g)
        a1 = torch.relu(z @ W[0]).double(); Sm += a1.t() @ a1; m += z.shape[0]
    Mmc = Sm / m
    log('  КОНТРОЛЬ M: ‖точна−MC‖/‖MC‖ = {:.4f}%   (очік. шум MC ≈ {:.3f}%)'
        .format(100 * float((M - Mmc).norm() / Mmc.norm()), 100 / math.sqrt(m)))

    # ---------- оракульні V з великої вибірки ----------
    A1 = []; YL = []
    for c in range(NBIG // CH):
        g = torch.Generator(device=dev).manual_seed(330000 + c)
        z = torch.randn(CH // 2, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
        a1 = torch.relu(xx @ W[0])
        A1.append(a1); YL.append(fwd_from_a1(W, a1))
    A1 = torch.cat(A1); YL = torch.cat(YL); nb = A1.shape[0]
    Yc = (YL.double() - YL.double().mean(0, keepdim=True))

    gg = torch.Generator(device=dev).manual_seed(4)
    Msum = torch.zeros(D, D, dtype=torch.float64, device=dev)
    for t in range(16):
        u = torch.randn(D, device=dev, generator=gg, dtype=torch.float64); u = u / u.norm()
        s = (Yc @ u).float()                                   # центроване за побудовою
        Cu = ((A1 * s.unsqueeze(1)).t() @ A1).double() / nb     # = Cov(a¹_i a¹_j , s)
        Cu = 0.5 * (Cu + Cu.t()); Msum += Cu @ Cu
    ev, V = torch.linalg.eigh(Msum); V = V.flip(1)
    VR, _ = torch.linalg.qr(torch.randn(D, D, device=dev, generator=gg, dtype=torch.float64))

    # ---------- оракульні β для кожної конфігурації (одноразово) ----------
    PRE = []
    for name, k in CFG:
        Vk = (V if name == 'оптим' else VR)[:, :k].contiguous()
        mu_q = ((Vk.t() @ M) * Vk.t()).sum(1)                   # v_aᵀ M v_a  (точно)
        mu_l = Vk.t() @ MU1                                     # v_a·E[a¹]   (точно)
        kk = 2 * k                                              # лінійні + квадратичні
        G = torch.zeros(kk, kk, dtype=torch.float64, device=dev)
        R = torch.zeros(kk, D, dtype=torch.float64, device=dev)
        sp = torch.zeros(kk, dtype=torch.float64, device=dev)
        for c0 in range(0, nb, CH):
            sl = slice(c0, min(c0 + CH, nb))
            p = feats(A1[sl], Vk, mu_l, mu_q); sp += p.sum(0)
            G += p.t() @ p; R += p.t() @ Yc[sl]
        G = G / nb - torch.outer(sp / nb, sp / nb); R = R / nb   # Yc уже центроване
        B = torch.linalg.solve(G + 1e-8 * float(G.diagonal().mean()) * torch.eye(kk, dtype=torch.float64, device=dev), R)
        PRE.append((name, k, Vk, mu_l, mu_q, B))
    del A1, YL, Yc
    torch.cuda.empty_cache()

    # ---------- продакшн-симуляція: ОДИН форвард на повтор ----------
    acc = {i: [[], [], [], []] for i in range(len(CFG))}       # e0, e_oracle, e_insample, dev_oracle
    for r in range(REPS):
        g = torch.Generator(device=dev).manual_seed(500000 + 977 * r + net)
        z = torch.randn(NPAIRS, D, device=dev, generator=g)
        x = torch.cat([z, -z], 0)
        x = x @ zca(x)
        a1 = torch.relu(x @ W[0])
        y = fwd_from_a1(W, a1).double()
        est0 = y.mean(0); e0 = float(((est0 - TH) ** 2).mean())
        for i, (name, k, Vk, mu_l, mu_q, B) in enumerate(PRE):
            p = feats(a1, Vk, mu_l, mu_q); pm = p.mean(0)
            est1 = est0 - pm @ B
            pc = p - pm
            Gi = (pc.t() @ pc) / p.shape[0]
            ri = (pc.t() @ (y - est0)) / p.shape[0]
            Bi = torch.linalg.solve(Gi + 1e-8 * float(Gi.diagonal().mean()) * torch.eye(2 * k, dtype=torch.float64, device=dev), ri)
            est2 = est0 - pm @ Bi
            acc[i][0].append(e0)
            acc[i][1].append(float(((est1 - TH) ** 2).mean()))
            acc[i][2].append(float(((est2 - TH) ** 2).mean()))
            acc[i][3].append((est1 - TH).cpu().numpy())
        del a1, y

    log('  {:>6s} {:>4s} {:>12s} {:>12s} {:>12s} {:>8s} {:>8s} {:>11s}'
        .format('напр.', 'k', 'MSE без CV', 'MSE ОРАКУЛ', 'MSE in-samp', 'виграш', 'in-samp', 'зсув²'))
    for i, (name, k, Vk, mu_l, mu_q, B) in enumerate(PRE):
        m0 = np.mean(acc[i][0]); m1 = np.mean(acc[i][1]); m2 = np.mean(acc[i][2])
        bias2 = float((np.mean(np.stack(acc[i][3]), axis=0) ** 2).mean())
        log('  {:>6s} {:4d} {:12.4e} {:12.4e} {:12.4e} {:7.3f}× {:7.3f}× {:11.3e}'
            .format(name, k, m0, m1, m2, m0 / m1, m0 / m2, bias2))
    log('  ⚠️ V оракульні ⟹ це ПОТЕНЦІАЛ. зсув² має бути ≪ MSE, інакше якір «поїхав».')
log('done.')
