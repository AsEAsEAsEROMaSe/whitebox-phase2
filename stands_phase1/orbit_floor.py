"""ПІДЛОГА МЕТОДУ ОРБІТ (аудит С48 пойнта finding-orbit-neighbour-transform).

ТЕЗА. У orbits.py немає контролю «незалежний семпл»; за нього прийняли ВИПАДКОВИЙ ПОВОРОТ
(rho=+0.058) і назвали це «спільна мода похибки». Але zQ НЕ незалежний від z: ортогональне
перетворення зберігає ||z|| ТОЧНО.

МЕХАНІЗМ. Мережа без зсувів однорідна степеня 1 ⟹ парна частина теж:
    r(z) = ||z|| * r(u),  u = z/||z||,  причому ||z|| ⊥ u для гаусіани.
⟹ E[r | ||z||] = ||z|| * mbar,  mbar = E_u[r(u)].
Ця РАДІАЛЬНА компонента інваріантна щодо БУДЬ-ЯКОГО ортогонального T (бо ||zT||=||z||),
отже вона входить у rho з коефіцієнтом РІВНО +1 для кожного T. Це підлога, не артефакт.

ПЕРЕДБАЧЕННЯ:  rho_floor = Var(||z||) * sum_j mbar_j^2 / sum_j Var(r_j)
Якщо воно збігається з +0.058 — підлога доведена і жоден орбітальний метод її не пройде.

КОНТРОЛІ:
  T=I               -> rho = +1.000   (прилад)
  радіальна f=||z|| -> rho = +1.000 для випадкового повороту (інваріант)
  безслідова квадратика z^T A z, tr A=0 -> rho ~ 0 (немає інваріантної частини)
"""
import torch, time, numpy as np
LOG = '/mnt/win_d/llm/White-Box/orbit_floor.log'; T0 = time.time()
open(LOG, 'w').close()
def log(m):
    s = '[{:6.1f}s] {}'.format(time.time() - T0, m); print(s, flush=True)
    open(LOG, 'a').write(s + '\n')

dev = 'cuda:1'; D = 256; L = 32
dat = np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d' % (net, i)].astype(np.float32), device=dev) for i in range(L)]

def even(W, x):
    a = x
    for w in W: a = torch.relu(a @ w)
    b = -x
    for w in W: b = torch.relu(b @ w)
    return (a + b) * 0.5

def randrot(seed):                      # ТОЙ САМИЙ seed 13, що в orbits.py
    g = torch.Generator(device=dev).manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(D, D, device=dev, generator=g))
    return Q

NCH, CH = 25, 2000                      # 50000 семплів — як в orbits.py
Q = randrot(13)

def rho_of(f, Tm, tag):
    """rho, зважена дисперсією по координатах — точно як в orbits.py"""
    z64 = lambda k: torch.zeros(k, dtype=torch.float64, device=dev)
    sx = sy = sxx = syy = sxy = None; n = 0
    for c in range(NCH):
        g = torch.Generator(device=dev).manual_seed(500000 + c)
        z = torch.randn(CH, D, device=dev, generator=g)
        a = f(z).double(); b = f(z @ Tm).double()
        if sx is None:
            k = a.shape[1]; sx, sy, sxx, syy, sxy = (z64(k) for _ in range(5))
        sx += a.sum(0); sy += b.sum(0); sxx += (a * a).sum(0); syy += (b * b).sum(0)
        sxy += (a * b).sum(0); n += CH
    vx = sxx / n - (sx / n) ** 2; vy = syy / n - (sy / n) ** 2
    cov = sxy / n - (sx / n) * (sy / n)
    r = float(cov.sum() / (vx.sum() * vy.sum()) ** 0.5)
    log('    {:52s} rho = {:+8.4f}'.format(tag, r))
    return r

# ---------------- контролі приладу (мережа не потрібна) ----------------
log('=' * 84); log('КОНТРОЛІ ПРИЛАДУ (випадковий поворот, seed 13)')
rho_of(lambda z: (z * z).sum(1, keepdim=True), Q, 'радіальна ||z||^2  (інваріант ⟹ має бути +1)')
g = torch.Generator(device=dev).manual_seed(77)
A = torch.randn(D, D, device=dev, generator=g); A = (A + A.t()) * 0.5
A -= torch.eye(D, device=dev) * (torch.diagonal(A).sum() / D)      # tr A = 0
log('    (перевірка: tr A = {:+.3e})'.format(float(torch.diagonal(A).sum())))
rho_of(lambda z: ((z @ A) * z).sum(1, keepdim=True), Q, 'безслідова квадратика (tr A=0 ⟹ має бути ~0)')

# ---------------- мережі ----------------
for net in (0, 1):
    W = load(net)
    log('=' * 84); log('МЕРЕЖА %d   (%d семплів)' % (net, NCH * CH))

    rho_I = rho_of(lambda z: even(W, z), torch.eye(D, device=dev), 'T = I  (контроль приладу)')
    rho_Q = rho_of(lambda z: even(W, z), Q, 'випадковий поворот  (orbits.py дав +0.058)')

    # --- радіальна частка: mbar = E_u[r(u)], Var(||z||), Var(r) ---
    s_m = torch.zeros(D, dtype=torch.float64, device=dev)
    s_r = torch.zeros(D, dtype=torch.float64, device=dev)
    s_rr = torch.zeros(D, dtype=torch.float64, device=dev)
    s_n = s_nn = 0.0; n = 0
    for c in range(NCH):
        g = torch.Generator(device=dev).manual_seed(500000 + c)
        z = torch.randn(CH, D, device=dev, generator=g)
        r = even(W, z).double()
        nz = z.double().norm(dim=1)
        s_m += (r / nz[:, None]).sum(0)          # r(u) = r(z)/||z||  (однорідність)
        s_r += r.sum(0); s_rr += (r * r).sum(0)
        s_n += float(nz.sum()); s_nn += float((nz * nz).sum()); n += CH
    mbar = s_m / n
    var_n = s_nn / n - (s_n / n) ** 2
    var_r = s_rr / n - (s_r / n) ** 2
    floor = float(var_n * (mbar * mbar).sum() / var_r.sum())

    # чесна перевірка однорідності (позитивний контроль механізму)
    g = torch.Generator(device=dev).manual_seed(4242)
    zt = torch.randn(512, D, device=dev, generator=g)
    h1 = even(W, zt); h2 = even(W, zt * 3.0)
    hom = float((h2 - 3.0 * h1).norm() / h1.norm())

    log('    ' + '-' * 74)
    log('    однорідність ||r(3z)-3r(z)||/||r(z)||        = {:.3e}  (має бути 0)'.format(hom))
    log('    Var(||z||)                                    = {:.5f}'.format(var_n))
    log('    ||mbar||^2 = ||E_u r(u)||^2                   = {:.5f}'.format(float((mbar * mbar).sum())))
    log('    sum_j Var(r_j)                                = {:.5f}'.format(float(var_r.sum())))
    log('    ПЕРЕДБАЧЕНА РАДІАЛЬНА ПІДЛОГА rho_floor       = {:+.4f}'.format(floor))
    log('    ВИМІРЯНА rho (випадковий поворот)             = {:+.4f}'.format(rho_Q))
    log('    відношення вимір/передбачення                 = {:.3f}'.format(rho_Q / floor if floor else float('nan')))
log('done.')
