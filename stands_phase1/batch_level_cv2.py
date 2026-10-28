"""ВИПРАВЛЕНА версія batch_level_cv.py — АУДИТ виявив систематичне заниження R².

ПОМИЛКА v1: F=48 ознак на n=300 батчів фіту ⟹ очікуваний зсув out-of-sample R² вниз
  R²_test ≈ R²_true − (1−R²_true)·F/n = −0.16 при R²_true=0.
Виміряне при ℓ=2 було −0.17 — збіг до 2-го знака ⟹ прилад мав роздільну здатність
рівно 0.16, а я міряв величини 0.17-0.82 і виводив «закон R²(ℓ)=ℓ/32».

ЩО ВИПРАВЛЕНО:
  1) НАБАГАТО більше батчів фіту (n=1500) ⟹ зсув падає до F/n
  2) СКАНУВАННЯ по числу ознак F: якщо R² росте з F і не виходить на полицю —
     значить набір ознак бідний, і «закон» був артефактом бідності
  3) ЯВНЕ виведення зсуву: друкуємо R² сирий і скоригований R²_true=(R²+F/n)/(1+F/n)
  4) КОНТРОЛЬ ЗБІЖНОСТІ: той самий вимір на половині батчів — якщо числа збігаються,
     n достатнє
  5) БАГАТШІ ОЗНАКИ: не лише mean і mean² по координатах, а й проєкції на топ-власні
     вектори коваріації активацій (позаоднодіагональна інформація)

ПИТАННЯ, НА ЯКЕ ВІДПОВІДАЄМО: чи справді R²(ℓ) ≈ ℓ/32 (тоді CV з дешевим якорем
неможлива), чи R² росте з якістю ознак і може перевищити ℓ/32 (тоді напрямок ЖИВИЙ).
Для виграшу при m=1 потрібно (1+ℓ/32)(1−R²/2) < 1, тобто R² > 2ℓ/(32+ℓ).
"""
import torch, math, numpy as np, time

dev = 'cuda:4'; D = 256; L = 32; T0 = time.time()
N = 6484; NPAIR = N // 2
NB_SEL = 500; NB_FIT = 1000; NB_TEST = 500
LAY = (2, 4, 8, 16, 24, 31)

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

def oos(P, Y, nfit, lam=1e-6):
    """B на перших nfit, R² на решті. Повертає (R²_сирий, R²_скоригований)"""
    F = P.shape[1]; nt = P.shape[0] - nfit
    P1, P2 = P[:nfit], P[nfit:]; Y1, Y2 = Y[:nfit], Y[nfit:]
    mp, my = P1.mean(0), Y1.mean(0)
    A1, B1 = P1 - mp, Y1 - my
    G = (A1.t() @ A1) / nfit; R = (A1.t() @ B1) / nfit
    Bc = torch.linalg.solve(G + lam * float(G.diagonal().mean()) * torch.eye(F, dtype=torch.float64, device=dev), R)
    res = (Y2 - my) - (P2 - mp) @ Bc
    r2 = 1 - float((res * res).mean(0).sum()) / float(((Y2 - my) ** 2).mean(0).sum())
    return r2, (r2 + F / nfit) / (1 + F / nfit)

for net in (0, 1):
    W = load(net); log('=' * 96); log('МЕРЕЖА %d' % net)
    gg = torch.Generator(device=dev).manual_seed(4)

    # ---- напрямки з ваг (БЕЗ доступу до y): власні вектори коваріації активацій ----
    EV = {}
    g = torch.Generator(device=dev).manual_seed(77)
    z = torch.randn(20000, D, device=dev, generator=g); xx = torch.cat([z, -z], 0)
    a = xx
    for i in range(L):
        a = torch.relu(a @ W[i])
        if i + 1 in LAY:
            ad = a.double(); mu = ad.mean(0)
            Cv = (ad.t() @ ad) / ad.shape[0] - torch.outer(mu, mu)
            e, Vv = torch.linalg.eigh(Cv); EV[i + 1] = Vv.flip(1).contiguous()
    del a, xx, z

    # ---- збір батчових середніх ----
    YB = []; PB = {l: [] for l in LAY}
    for b in range(NB_SEL + NB_FIT + NB_TEST):
        x = wbatch(800000 + 7919 * b + net)
        a = x
        for i in range(L):
            a = torch.relu(a @ W[i])
            if i + 1 in LAY:
                ad = a.double(); pr = ad @ EV[i + 1][:, :64]
                PB[i + 1].append(torch.cat([ad.mean(0), (ad * ad).mean(0),
                                            pr.mean(0), (pr * pr).mean(0)]))
        YB.append(a.double().mean(0))
        del x, a
    YB = torch.stack(YB)
    for l in LAY: PB[l] = torch.stack(PB[l])
    log('  зібрано {:d} батчів по {:d} семплів (відбір/фіт/тест = {}/{}/{})'
        .format(NB_SEL + NB_FIT + NB_TEST, N, NB_SEL, NB_FIT, NB_TEST))

    log('  {:>4s} {:>5s} {:>10s} {:>12s} {:>12s} {:>10s} {:>9s}'
        .format('шар', 'F', 'R² сир.', 'R² скориг.', 'потрібно', 'запас', 'збіжн.'))
    for l in LAY:
        Pl = PB[l]
        # ВІДБІР ознак — на ОКРЕМОМУ наборі, щоб корекція F/n лишалась чесною
        As = Pl[:NB_SEL] - Pl[:NB_SEL].mean(0)
        Ys = YB[:NB_SEL] - YB[:NB_SEL].mean(0)
        sc = ((As.t() @ Ys) ** 2).sum(1) / (As * As).mean(0).clamp(min=1e-30)
        order = torch.argsort(sc, descending=True)
        PF = Pl[NB_SEL:]; YF = YB[NB_SEL:]
        need = 2.0 * l / (32.0 + l)                     # поріг виграшу при m=1
        for F in (16, 48, 128, 320):
            idx = order[:F]
            r2, r2c = oos(PF[:, idx], YF, NB_FIT)
            # контроль збіжності: те саме на половині батчів фіту
            r2h, _ = oos(PF[:NB_FIT // 2 + NB_TEST][:, idx], YF[:NB_FIT // 2 + NB_TEST], NB_FIT // 2)
            log('  {:4d} {:5d} {:10.4f} {:12.4f} {:12.4f} {:10.4f} {:9.4f}'
                .format(l, F, r2, r2c, need, r2c - need, r2h))
    log('  «потрібно» = 2ℓ/(32+ℓ): поріг, вище якого CV з дешевим якорем ВИГРАЄ при m=1')
    log('  «збіжн.» = сирий R² на половині батчів фіту; має бути близько до сирого')
log('done.')
