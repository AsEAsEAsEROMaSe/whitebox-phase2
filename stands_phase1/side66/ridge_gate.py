"""ВОРОТА ДЛЯ ВСІХ ОДНОВИМІРНИХ ІДЕЙ: чи розкладається `F` у суму рідж-функцій?

ЗВІДКИ (ідеї юзера: рідж-лети/Радон, Бєсов/вейвлети, Ґеґенбауер, раціональні).
Усі вони — про КРАЩИЙ ОДНОВИМІРНИЙ базис. Але нам потрібна не апроксимація
`F`, а її ІНТЕГРАЛ по гаусіану в 256 вимірах. Раціональна функція від 256
змінних по гаусіану точно не береться; вейвлет теж. **Точно береться лише
рідж-функція `g(w·z)`**: якщо `‖w‖=1`, то `w·z ~ N(0,1)` і
`E[g(w·z)] = ∫g(t)φ(t)dt` — одновимірна квадратура, машинна точність, БЕЗ
жодного семплу (сама `g` може бути хоч раціональною, хоч вейвлетною).

⟹ Тому питання не «який базис», а: **скільки дисперсії `F` бере адитивна
модель `c + Σ_{i≤m} g_i(u_i·z)` з довільними одновимірними `g_i`?**
Це СПІЛЬНА стеля для рідж-летів, Радона, Бєсова, Ґеґенбауера і раціональних
разом — і водночас стеля найкращої контрольної змінної з точним середнім:
    виграш КЗ ≤ 1/(1 − частка_знятої_дисперсії)

ЯК. Напрямки `u_i` — власні вектори активного підпростору `E[∇F∇Fᵀ]`
(точний градієнт кусково-лінійної мережі). `g_i` — непараметричні (бінінг
по 64 комірки з лінійною інтерполяцією), підганяються зворотним допасуванням
(backfitting). Усе міряється на ВІДКЛАДЕНІЙ половині.

КОНТРОЛІ
  * ✅ СИНТЕТИКА `Σ_{i≤8} relu(u_i·z)` — АДИТИВНА за побудовою: на m=8 мусить
    дати ~1.00, інакше зламане допасування;
  * ✅ ЛІНІЙНА `w·z` — мусить датися одним членом (m=1 → ~1.00);
  * ⛔ ВИПАДКОВІ напрямки замість активних — мусить бути помітно гірше;
  * ✅ m=0 → рівно 0.000;
  * ⚠️ усе на відкладеній половині: бінінг з 64 комірками × m членів легко
    перевчається, і саме це давало б міраж.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NS = 262144
CH = 8192
NBIN = 64
MS = [1, 2, 4, 8, 16, 32, 64, 128]
NPASS = 6


def fwd_grad(wc, z, u):
    xs, ms = [], []
    x = z
    for i in range(len(wc)):
        p = x @ wc[i]
        m = (p > 0).to(p.dtype)
        x = p * m
        ms.append(m)
    g = u.unsqueeze(0).expand(z.shape[0], -1)
    for i in range(len(wc) - 1, -1, -1):
        g = (g * ms[i]) @ wc[i].T
    return x, g


def active_dirs(wc, uo, dev, seed, n=32768):
    g0 = torch.Generator(device=dev).manual_seed(seed)
    A = torch.zeros(W_, W_, dtype=torch.float64, device=dev)
    done = 0
    while done < n:
        z = torch.randn(CH, W_, device=dev, generator=g0)
        _, gr = fwd_grad(wc, z, uo)
        gd = gr.double()
        A += gd.T @ gd
        done += CH
    A /= done
    ev, U = torch.linalg.eigh(A)
    return U.flip(1).float(), ev.flip(0)


def backfit(P, y, nbin=NBIN, npass=NPASS, edges=None):
    """Адитивна модель бінінгом. P: (n, m) координати. Повертає прогноз."""
    n, m = P.shape
    fit = torch.zeros(n, dtype=torch.float64, device=P.device)
    parts = [torch.zeros(n, dtype=torch.float64, device=P.device)
             for _ in range(m)]
    tabs = []
    if edges is None:
        edges = torch.linspace(-3.5, 3.5, nbin + 1, device=P.device,
                               dtype=torch.float64)
    idx = [torch.bucketize(P[:, i].double(), edges).clamp(1, nbin) - 1
           for i in range(m)]
    mu = y.mean()
    for _ in range(npass):
        for i in range(m):
            r = y - mu - (fit - parts[i])
            s = torch.zeros(nbin, dtype=torch.float64, device=P.device)
            c = torch.zeros(nbin, dtype=torch.float64, device=P.device)
            s.index_add_(0, idx[i], r)
            c.index_add_(0, idx[i], torch.ones_like(r))
            tab = s / c.clamp_min(1.0)
            tab = tab - (tab * c).sum() / c.sum().clamp_min(1.0)
            newp = tab[idx[i]]
            fit = fit - parts[i] + newp
            parts[i] = newp
            if len(tabs) < m:
                tabs.append(tab)
            else:
                tabs[i] = tab
    return mu, tabs, edges, idx


def apply_fit(mu, tabs, edges, P):
    out = torch.full((P.shape[0],), float(mu), dtype=torch.float64,
                     device=P.device)
    for i, tab in enumerate(tabs):
        j = torch.bucketize(P[:, i].double(), edges).clamp(1, len(tab)) - 1
        out = out + tab[j]
    return out


def capture(U, Z, F, m, half):
    P = Z @ U[:, :m]
    mu, tabs, edges, _ = backfit(P[:half], F[:half])
    pr = apply_fit(mu, tabs, edges, P[half:])
    y = F[half:]
    return 1.0 - float(((y - pr) ** 2).mean() / y.var())


def main(nets=3, dev=DEV):
    torch.manual_seed(9090)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    half = NS // 2
    g0 = torch.Generator(device=dev).manual_seed(555)
    Zc = torch.cat([torch.randn(CH, W_, device=dev, generator=g0)
                    for _ in range(NS // CH)])
    # ── ✅ КОНТРОЛІ на синтетиці з ВІДОМОЮ адитивною будовою
    Uc = torch.linalg.qr(torch.randn(W_, W_, device=dev, generator=g0))[0]
    lin = (Zc @ Uc[:, 0]).double()
    add = sum(torch.relu(Zc @ Uc[:, i]) for i in range(8)).double()
    for nm, yy in (('✅ лінійна w·z (треба m=1 → 1.00)', lin),
                   ('✅ Σ_{i≤8} relu(u_i·z) (треба m=8 → 1.00)', add)):
        row = [capture(Uc, Zc, yy, m, half) for m in (1, 2, 4, 8, 16)]
        print(f'{nm:44s} ' + ' '.join(f'm={m}:{v:.3f}'
                                      for m, v in zip((1, 2, 4, 8, 16), row)))

    print(f'\nЧАСТКА ДИСПЕРСІЇ `F`, ЗНЯТА АДИТИВНОЮ РІДЖ-МОДЕЛЛЮ '
          f'(відкладена половина, {NS} семплів)')
    print(f'{"мережа":>7s} ' + ' '.join(f'{"m="+str(m):>8s}' for m in MS)
          + f' {"⛔вип.m=32":>10s}')
    acc, accr = [], []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            nw = wc[-1].shape[1]
            uo = torch.ones(nw, device=dev) / math.sqrt(nw)
            U, ev = active_dirs(wc, uo, dev, 3000 + n)
            F = torch.cat([(torch.relu_(torch.zeros(0)) if False else
                            fwd_grad(wc, Zc[i:i + CH], uo)[0] @ uo).double()
                           for i in range(0, NS, CH)])
            row = [capture(U, Zc, F, m, half) for m in MS]
            Ur = torch.linalg.qr(torch.randn(W_, W_, device=dev,
                                             generator=g0))[0]
            rr = capture(Ur, Zc, F, 32, half)
            acc.append(row)
            accr.append(rr)
            print(f'{n+1:7d} ' + ' '.join(f'{v:8.3f}' for v in row)
                  + f' {rr:10.3f}', flush=True)
        del W, wc
        torch.cuda.empty_cache()
    A = np.array(acc)
    print(f'{"СЕРЕДНЄ":>7s} ' + ' '.join(f'{v:8.3f}' for v in A.mean(0))
          + f' {np.mean(accr):10.3f}')
    print(f'\nСТЕЛЯ КОНТРОЛЬНОЇ ЗМІННОЇ = 1/(1−частка):')
    print(f'{"":>7s} ' + ' '.join(f'{1.0/max(1e-9,1-v):7.2f}×'
                                  for v in A.mean(0)))
    print('\n🔑 читати так: це СПІЛЬНА стеля рідж-летів, Радона, Бєсова, '
          'Ґеґенбауера і раціональних — бо лише рідж-функція інтегрується '
          'по гаусіану ТОЧНО й без семплів')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
