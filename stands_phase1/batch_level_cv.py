"""ДЕШЕВИЙ ЯКІР — версія 2, з ПРАВИЛЬНИМ підбором коефіцієнта.

ЧОМУ v1 (cheap_anchor.py) БУВ НЕДІЙСНИЙ:
B підбирався ПОСЕМПЛОВО на невибіленій вибірці. Але після ZCA вибіркове середнє будь-якої
КВАДРАТИЧНОЇ форми стає ТОЧНО істинним ⟹ квадратична частина ознаки не флуктуює зовсім.
Отже коефіцієнт, який працює на рівні СЕРЕДНІХ, визначається лише 4-м порядком і вище,
а посемпловий B підігнаний під 2-й. Підставляли коефіцієнт з іншої задачі.

ПРАВИЛЬНО: підбирати B на РІВНІ БАТЧІВ. Багато вибілених батчів; для кожного пара (p̄, ȳ);
регресія ȳ на p̄ по батчах. Це рівно той об'єкт, що працює в продакшні.

ЩО МІРЯЄМО (спершу лише ПОТЕНЦІАЛ, без плати за якір):
  R² out-of-sample: скільки дисперсії ȳ пояснює p̄ на СВІЖИХ батчах.
  Якщо R²≈0 — гілка закрита чесно. Якщо великий — рахуємо економіку якоря окремо.
КОНТРОЛІ: випадкові напрямки; фіт і тест на РІЗНИХ батчах; кілька шарів ℓ.
"""
import torch, math, numpy as np, time

dev = 'cuda:4'; D = 256; L = 32; T0 = time.time()
N = 6484; NPAIR = N // 2
NB_FIT = 300; NB_TEST = 200
KDIR = 24                      # ⟹ 2*KDIR = 48 ознак на 300 батчів фіту

def log(m): print('[{:6.1f}s] {}'.format(time.time() - T0, m), flush=True)
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d' % (n, i)].astype(np.float32), device=dev) for i in range(L)]
def zca(x):
    C = (x.t() @ x) / x.shape[0]
    e, V = torch.linalg.eigh(C.double())
    return (V @ torch.diag(e.clamp(min=1e-8).rsqrt()) @ V.t()).float()
def wbatch(seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(NPAIR, D, device=dev, generator=g)
    x = torch.cat([z, -z], 0)
    return x @ zca(x)

for net in (0, 1):
    W = load(net); log('=' * 78); log('МЕРЕЖА %d' % net)
    LAY = (2, 4, 8, 16, 24, 31)
    gg = torch.Generator(device=dev).manual_seed(4)
    VR, _ = torch.linalg.qr(torch.randn(D, D, device=dev, generator=gg))

    # ---------- один прохід: збираємо середні по батчах для ВСІХ шарів ----------
    YB = []                                     # NB × D
    PB = {l: [] for l in LAY}                   # NB × 2D  (лінійні + квадратичні по КООРДИНАТАХ)
    for b in range(NB_FIT + NB_TEST):
        x = wbatch(800000 + 7919 * b + net)
        a = x
        for i in range(L):
            a = torch.relu(a @ W[i])
            if i + 1 in LAY:
                ad = a.double()
                PB[i + 1].append(torch.cat([ad.mean(0), (ad * ad).mean(0)]))
        YB.append(a.double().mean(0))
        del x, a
    YB = torch.stack(YB)
    for l in LAY: PB[l] = torch.stack(PB[l])

    def oos_r2(Pfull, tag):
        """B на перших NB_FIT батчах, R² на решті. Pfull: NB × F"""
        P1 = Pfull[:NB_FIT]; P2 = Pfull[NB_FIT:]
        Y1 = YB[:NB_FIT]; Y2 = YB[NB_FIT:]
        mp = P1.mean(0); my = Y1.mean(0)
        A1 = P1 - mp; B1 = Y1 - my
        G = (A1.t() @ A1) / NB_FIT; R = (A1.t() @ B1) / NB_FIT
        Bc = torch.linalg.solve(G + 1e-4 * float(G.diagonal().mean()) * torch.eye(G.shape[0], dtype=torch.float64, device=dev), R)
        A2 = P2 - mp; res = (Y2 - my) - A2 @ Bc
        r2 = 1 - float((res * res).mean(0).sum()) / float(((Y2 - my) ** 2).mean(0).sum())
        return r2

    log('  {:>4s} {:>10s} {:>12s} {:>12s}'.format('шар', 'ознак', 'R² оптим', 'R² випадк.'))
    for l in LAY:
        Pl = PB[l]                                   # NB × 2D у КООРДИНАТНОМУ базисі
        # відбір напрямків: топ-KDIR координат за |кореляцією| з ȳ (на ФІТ-батчах)
        A1 = Pl[:NB_FIT] - Pl[:NB_FIT].mean(0)
        Y1 = YB[:NB_FIT] - YB[:NB_FIT].mean(0)
        sc = ((A1.t() @ Y1) ** 2).sum(1) / (A1 * A1).mean(0).clamp(min=1e-30)
        idx = torch.argsort(sc, descending=True)[:2 * KDIR]
        ridx = torch.randperm(2 * D, device=dev, generator=gg)[:2 * KDIR]
        log('  {:4d} {:10d} {:12.4f} {:12.4f}'
            .format(l, 2 * KDIR, oos_r2(Pl[:, idx], 'опт'), oos_r2(Pl[:, ridx], 'вип')))
    log('  R² — частка дисперсії БАТЧОВОГО СЕРЕДНЬОГО ȳ, пояснена середніми ознак p̄')
    log('  (це ПОТЕНЦІАЛ з ідеальним якорем; плата за якір рахується окремо)')
log('done.')
