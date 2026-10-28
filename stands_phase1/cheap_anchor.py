"""ДЕШЕВИЙ (не безкоштовний) ЯКІР: ознаки з проміжного шару ℓ.

ЛОГІКА, ЧОМУ ЦЕ ОБХОДИТЬ [[finding-anchor-cost-identity]]:
та тотожність виведена для якоря, що коштує СТІЛЬКИ Ж, скільки семпл. Щоб порахувати
E[g(a^ℓ)], потрібні лише ℓ шарів із 32 ⟹ якір дешевший у 32/ℓ разів. При ℓ=4 це ×8.
А a^ℓ — глибоко нелінійна функція x: несе 4-й порядок і вищі, на відміну від a¹
([[finding-arccos-cv-and-anchor-drift]] — там нуль саме тому, що a¹ занадто «близька» до x).

СХЕМА (усе продакшн-матчено):
  головна вибірка: N семплів, антитетика + ZCA, ПОВНИЙ прогін → y і a^ℓ
  якірна вибірка:  m НЕЗАЛЕЖНИХ батчів по N, кожен ЗІ СВОЇМ ZCA, прогін лише до шару ℓ
       ⟹ якірні семпли з ТОЧНО того самого розподілу, що й головні (важливо: ZCA
         залежить від розміру батча, тож батчі мусять бути однакові — інакше зсув)
  оцінка: est = ȳ − B·(p̄_гол − p̄_якір)
  БЮДЖЕТ: N·(1 + m·ℓ/32) = 6484  ⟹  N = 6484/(1 + m·ℓ/32)

КОНТРОЛІ:
  • базова лінія — той самий v2 на ПОВНОМУ N=6484 без CV (паровано за сідами)
  • ВИПАДКОВІ напрямки замість оптимальних
  • зсув² для базової лінії І для CV окремо (діагностика без базової лінії — не діагностика)
⚠️ напрямки V оракульні ⟹ це ПОТЕНЦІАЛ. Якщо є — окремо шукаємо дешевий спосіб їх дістати.
"""
import torch, math, numpy as np, time

dev = 'cuda:4'; D = 256; L = 32; T0 = time.time()
BUD = 6484
REPS = 60
NBIG = 120000
CH = 20000
KDIR = 32

def log(m): print('[{:6.1f}s] {}'.format(time.time() - T0, m), flush=True)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d' % (n, i)].astype(np.float32), device=dev) for i in range(L)]

def part(W, x, l):
    """a^l — прогін лише перших l шарів"""
    a = x
    for i in range(l): a = torch.relu(a @ W[i])
    return a
def rest(W, a, l):
    for i in range(l, L): a = torch.relu(a @ W[i])
    return a
def zca(x):
    C = (x.t() @ x) / x.shape[0]
    e, V = torch.linalg.eigh(C.double())
    return (V @ torch.diag(e.clamp(min=1e-8).rsqrt()) @ V.t()).float()
def whiten_batch(npair, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, D, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    return x @ zca(x)
def feats(a, Vk):
    pr = a.double() @ Vk
    return torch.cat([pr, pr * pr], 1)

for net in (0, 1):
    W = load(net); log('=' * 92); log('МЕРЕЖА %d' % net)
    S = torch.zeros(D, dtype=torch.float64, device=dev); n = 0
    for c in range(20):
        g = torch.Generator(device=dev).manual_seed(910000 + c)
        z = torch.randn(50000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
        S += rest(W, xx, 0).double().sum(0); n += xx.shape[0]
    TH = S / n

    for l in (2, 4, 8):
        # ---------- оракульні напрямки з великої вибірки ----------
        AL = []; YL = []
        for c in range(NBIG // CH):
            g = torch.Generator(device=dev).manual_seed(330000 + c)
            z = torch.randn(CH // 2, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
            a = part(W, xx, l); AL.append(a); YL.append(rest(W, a, l))
        AL = torch.cat(AL); YL = torch.cat(YL); nb = AL.shape[0]
        Yc = YL.double() - YL.double().mean(0, keepdim=True)
        gg = torch.Generator(device=dev).manual_seed(4)
        Ms = torch.zeros(D, D, dtype=torch.float64, device=dev)
        for t in range(16):
            u = torch.randn(D, device=dev, generator=gg, dtype=torch.float64); u = u / u.norm()
            s = (Yc @ u).float()
            Cu = ((AL * s.unsqueeze(1)).t() @ AL).double() / nb
            Ms += (0.5 * (Cu + Cu.t())) @ (0.5 * (Cu + Cu.t()))
        ev, V = torch.linalg.eigh(Ms); V = V.flip(1)
        VR, _ = torch.linalg.qr(torch.randn(D, D, device=dev, generator=gg, dtype=torch.float64))
        BB = {}
        for nm, VV in (('оптим', V), ('вип.', VR)):
            Vk = VV[:, :KDIR].contiguous()
            P = feats(AL, Vk); Pc = P - P.mean(0, keepdim=True)
            G = (Pc.t() @ Pc) / nb; R = (Pc.t() @ Yc) / nb
            BB[nm] = (Vk, torch.linalg.solve(
                G + 1e-8 * float(G.diagonal().mean()) * torch.eye(2 * KDIR, dtype=torch.float64, device=dev), R))
        del AL, YL, Yc, P, Pc
        torch.cuda.empty_cache()

        log('  ── шар ℓ={:d} (якір у {:.0f}× дешевший за семпл) ──'.format(l, 32.0 / l))
        # базова лінія — ОДИН раз на (мережа, шар), паровано за сідами
        eb = []
        for r in range(REPS):
            sd = 700000 + 7919 * r + 131 * l + net
            xb = whiten_batch(BUD // 2, sd)
            eb.append(float(((rest(W, xb, 0).double().mean(0) - TH) ** 2).mean()))
        m0 = np.mean(eb)
        log('  {:>6s} {:>3s} {:>6s} {:>12s} {:>12s} {:>8s} {:>11s}'
            .format('напр.', 'm', 'N', 'MSE база', 'MSE з CV', 'виграш', 'зсув² CV'))
        for mrep in (1, 2, 4, 8):
            N = int(BUD / (1.0 + mrep * l / 32.0)); npair = N // 2
            sc = mrep / (mrep + 1.0)                 # ОПТИМАЛЬНЕ масштабування під шум якоря
            res = {nm: ([], []) for nm in ('оптим', 'вип.')}
            for r in range(REPS):
                sd = 700000 + 7919 * r + 131 * l + net
                x = whiten_batch(npair, sd)
                a = part(W, x, l); est0 = rest(W, a, l).double().mean(0)
                anc = [part(W, whiten_batch(npair, sd + 100000 * (q + 1)), l) for q in range(mrep)]
                for nm in ('оптим', 'вип.'):
                    Vk, B = BB[nm]
                    pm = feats(a, Vk).mean(0)
                    pa = sum(feats(aa, Vk).mean(0) for aa in anc) / mrep
                    est1 = est0 - (pm - pa) @ (B * sc)
                    res[nm][0].append(float(((est1 - TH) ** 2).mean()))
                    res[nm][1].append((est1 - TH).cpu().numpy())
                del anc, a
            for nm in ('оптим', 'вип.'):
                m1 = np.mean(res[nm][0])
                b2 = float((np.mean(np.stack(res[nm][1]), axis=0) ** 2).mean())
                log('  {:>6s} {:3d} {:6d} {:12.4e} {:12.4e} {:7.3f}× {:11.3e}'
                    .format(nm, mrep, N, m0, m1, m0 / m1, b2))
log('done.')
