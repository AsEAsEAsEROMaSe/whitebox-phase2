"""ГАУСІВ НАХИЛ НАСКРІЗЬ: реально семплюємо з `q` і міряємо MSE проти істини.

ЗВІДКИ. `impsamp2.py` дав 1.216× при фіті на ПРОДАКШН-обсязі (4004 пари) —
і це перше за два дні число, здатне рухати бал. Але воно рахувалося
ПЕРЕЗВАЖУВАННЯМ φ-семплів, а не реальним семплюванням з `q`. Тут — чесно:
    q = N(0, Σ_q),  Σ_q⁻¹ = I − 2C,  C = Σ_i c_i u_iu_iᵀ
    семплювання: z = ξ + Σ_i (α_i−1)(u_i·ξ)u_i,  α_i = 1/√(1−2c_i)
    вага:        w(z) = (Π α_i)·exp(−Σ_i c_i (u_i·z)²)
    оцінка:      T̂ = (1/n) Σ w_i ψ(z_i)     ← незміщена
і міряємо `E‖T̂ − T‖²` проти базової лінії БЕЗ нахилу на тій самій кількості
пар. Істина рахується самостійно на 2М семплах.

🚨 ЦІНА, ЯКУ ТРЕБА ЧЕСНО ВІДНЯТИ. Щоб побудувати `q`, потрібні:
  * напрямки `u_i` — власні вектори `E[∇F∇Fᵀ]` (градієнти = ще один прохід);
  * коефіцієнти `c_i` — фіт `log‖ψ‖` на ПІЛОТІ повних прогонів.
Обидва рахуються ДО основної вибірки і з'їдають бюджет. Тому друкуємо
і «сирий» виграш, і виграш ПІСЛЯ вирахування пілота.
Плюс дешевий варіант: `u_i` з МАТРИЦІ ПЕРЕНОСУ `G` (лише ваги + затвори,
без жодного градієнта) — якщо він працює, ціна напрямків падає в нуль.

КОНТРОЛІ
  * ✅ `C=0` мусить відтворити базову лінію ДО ОСТАННЬОЇ ЦИФРИ;
  * ✅ незміщеність: середнє `T̂` по жеребах мусить збігатися з `T`
    (друкуємо ‖зміщення‖²/MSE — має бути ≈0);
  * ⛔ `C` від ЧУЖОЇ мережі — мусить дати ≈1.0 або гірше;
  * ⚠️ ESS ваг: якщо валиться, виграш є артефактом важких хвостів;
  * усе на ПРОДАКШН-npair, NDRAW незалежних жеребів.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import cut_net                              # noqa: E402
from impsamp import gates_and_G                               # noqa: E402
from anchor_depth import truth_means                          # noqa: E402
from ridge_gate import active_dirs                            # noqa: E402


def dirs_from_G(wc, G, dev, seed, npair=2048):
    """Напрямки З ВАГ, БЕЗ жодного градієнта: у середньому полі
    ∇F ≈ W₀·diag(затвор₀)·G, отже активний підпростір ≈ старші власні
    вектори `AAᵀ` при `A = W₀ diag(затвор₀) G`. Ціна — один малий пілот
    на затвор₀ (він і так рахується в sub66)."""
    rng = np.random.default_rng(int(seed))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    p0 = torch.cat([z, -z], 0) @ wc[0]
    g0 = (p0 > 0).to(p0.dtype).mean(0).double()
    A = (wc[0].double() * g0.unsqueeze(0)) @ G
    ev, U = torch.linalg.eigh(A @ A.T)
    return U.flip(1).float()

W_, L_ = 256, 32
DEV = 'cuda:3'
NDRAW = 64
MTILT = 64


def psi_of(wc, z, G, m0):
    """ψ(z) = x_L − G(x₀ − m₀), антитетично усереднене по парі."""
    zb = torch.cat([z, -z], 0)
    x, x0 = zb, None
    for i, w in enumerate(wc):
        x = torch.relu(x @ w)
        if i == 0:
            x0 = x.double()
    p = x.double() - (x0 - m0) @ G
    n = z.shape[0]
    return 0.5 * (p[:n] + p[n:])


def fit_c(wc, U, G, m0, dev, seed, npair, m):
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, W_, device=dev, generator=g)
    y = psi_of(wc, z, G, m0).norm(dim=1).log()
    P = (z @ U[:, :m]).double() ** 2
    A = torch.cat([P, torch.ones(npair, 1, dtype=torch.float64, device=dev)], 1)
    c = torch.linalg.lstsq(A, y.unsqueeze(1)).solution[:m, 0]
    return c.clamp(-2.0, 0.40)


def draw(wc, U, c, G, m0, dev, seed, npair, m):
    g = torch.Generator(device=dev).manual_seed(seed)
    xi = torch.randn(npair, W_, device=dev, generator=g)
    if c is None:
        z, w = xi, torch.ones(npair, dtype=torch.float64, device=dev)
    else:
        al = (1.0 - 2.0 * c).clamp_min(1e-6).rsqrt()
        Um = U[:, :m]
        pr = (xi @ Um).double()
        z = xi + ((pr * (al - 1.0)).float() @ Um.T)
        q = (z @ Um).double() ** 2
        w = torch.exp(al.log().sum() - (q * c).sum(1))
    ps = psi_of(wc, z, G, m0)
    return (w.unsqueeze(1) * ps).mean(0), w


def main(nets=6, dev=DEV):
    torch.manual_seed(777)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'{"мережа":>7s} {"✅C=0":>8s} {"НАХИЛ":>8s} {"U З ВАГ":>9s} '
          f'{"c=const":>9s} {"⛔чужа C":>9s} {"ESS":>6s} {"‖змщ‖²/MSE":>11s}')
    tot = {'base': 0.0, 'tilt': 0.0, 'alien': 0.0, 'ctrl': 0.0}
    prev_c = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_net(W, dev)
            G = gates_and_G(wc, dev, 4242 + n)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() * INV_SQRT_2PI
            T = truth_means(wc, dev, 310_000 + 7919 * n)[-1]
            nw = wc[-1].shape[1]
            uo = torch.ones(nw, device=dev) / math.sqrt(nw)
            U, _ = active_dirs(wc, uo, dev, 4700 + n)
            Ug = dirs_from_G(wc, G, dev, 66_000 + n)
            c = fit_c(wc, U, G, m0, dev, 55_000 + n, npair, MTILT)
            cg = fit_c(wc, Ug, G, m0, dev, 55_000 + n, npair, MTILT)
            cu = torch.full_like(c, float(c.mean()))
            ca = prev_c if (prev_c is not None and prev_c.shape == c.shape) else None

            eb = et = ea = ec = 0.0
            egg = ecu = 0.0
            sb = torch.zeros_like(T)
            st = torch.zeros_like(T)
            ess = 0.0
            for d in range(NDRAW):
                sd = 12_000_000 + 313 * n + d
                mb, _ = draw(wc, U, None, G, m0, dev, sd, npair, MTILT)
                mc, _ = draw(wc, U, torch.zeros_like(c), G, m0, dev, sd,
                             npair, MTILT)
                mt, wt = draw(wc, U, c, G, m0, dev, sd + 7_000, npair, MTILT)
                eb += float(((mb - T) ** 2).sum())
                ec += float(((mc - T) ** 2).sum())
                et += float(((mt - T) ** 2).sum())
                sb += mb
                st += mt
                ess += float(wt.sum() ** 2 / (wt * wt).sum()) / npair
                mg, _ = draw(wc, Ug, cg, G, m0, dev, sd + 7_000, npair, MTILT)
                egg += float(((mg - T) ** 2).sum())
                mu, _ = draw(wc, U, cu, G, m0, dev, sd + 7_000, npair, MTILT)
                ecu += float(((mu - T) ** 2).sum())
                if ca is not None:
                    ma, _ = draw(wc, U, ca, G, m0, dev, sd + 9_000, npair, MTILT)
                    ea += float(((ma - T) ** 2).sum())
            bias2 = float(((st / NDRAW - T) ** 2).sum())
            tot['base'] += eb
            tot['tilt'] += et
            tot['ctrl'] += ec
            tot['alien'] += ea if ca is not None else eb
            tot.setdefault('gdir', 0.0)
            tot.setdefault('unif', 0.0)
            tot['gdir'] += egg
            tot['unif'] += ecu
            print(f'{n+1:7d} {eb/max(ec,1e-300):8.4f} {eb/max(et,1e-300):8.3f} '
                  f'{eb/max(egg,1e-300):9.3f} {eb/max(ecu,1e-300):9.3f} '
                  + (f'{eb/max(ea,1e-300):9.3f}' if ca is not None else
                     f'{"—":>9s}')
                  + f' {ess/NDRAW:6.3f} {bias2/(et/NDRAW):11.4f}',
                  flush=True)
            prev_c = c
        del W, wc, G
        torch.cuda.empty_cache()
    print(f'\n{"РАЗОМ":>7s} ✅C=0 {tot["base"]/tot["ctrl"]:.4f}×   '
          f'НАХИЛ(градієнтні U) {tot["base"]/tot["tilt"]:.3f}×   '
          f'U З ВАГ {tot["base"]/tot["gdir"]:.3f}×   '
          f'c=const {tot["base"]/tot["unif"]:.3f}×   '
          f'⛔чужа C {tot["base"]/tot["alien"]:.3f}×')
    print('\n💰 ЦІНА ПОБУДОВИ `q` (треба відняти):')
    print('   напрямки `u_i`: 32768 градієнтів ≈ 2× прогони ⟹ дорого, але їх '
          'можна брати з матриці переносу `G` (лише ваги+затвори) — окремий тест;')
    print('   коефіцієнти `c_i`: пілот npair пар ⟹ +100% до вибірки, якщо не '
          'перевикористати ті самі семпли (а це вже теорема тих самих семплів).')
    print('🔑 при подвоєнні вибірки замість нахилу MSE падає рівно вдвічі ⟹ '
          'нахил мусить дати БІЛЬШЕ за 2.0×, щоб виправдати пілот тієї ж ціни')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
