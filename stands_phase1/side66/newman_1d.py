"""ТЕОРЕМА НЬЮМАНА НА НАШІЙ ФУНКЦІЇ: раціональне проти поліноміального.

ЗВІДКИ (ідея юзера). Ньюман 1964: поліноми наближають `|x|` зі швидкістю
`1/n`, раціональні — `exp(−c√n)`. Різниця в КЛАСІ швидкості. Якщо це
переноситься на нашу мережу, то ермітів спектр — діагноз ІНСТРУМЕНТА, а не
задачі, і неберучі 10–31% беруться іншим класом за ту саму ціну.

🚨 ПАСТКА, ЯКУ МІРЯЄМО ПЕРШОЮ. Ньюман — про ОДИН злам. Уздовж прямої в
32-шаровій мережі зламів стільки, скільки нейронів на ній перемикається.
Для `K` зламів раціональному наближенню потрібно ~`K` полюсів, і
`exp(−c√n)` стає `exp(−c√(n/K))`. Тому ПЕРШЕ число стенда — не похибка,
а **кількість зламів на гаусовій прямій**. Якщо їх тисячі — теорема не
працює на наших `n`, і це видно ДО будь-якої апроксимації.

ЩО РАХУЄМО
  1. `K` — кількість зламів `f(t) = F(t·u + w⊥)`, `t ~ N(0,1)`, точним
     підрахунком змін знаку ДОактивацій усіх нейронів уздовж прямої;
  2. похибка НАЙКРАЩОГО полінома степеня `n` у L²(гаусова вага) — через
     ермітову проєкцію (ортогональний базис ⟹ проєкція і є найкраща);
  3. похибка раціональної через **AAA** (Накацукаса-Сете-Трефетен 2018) —
     барицентричний вигляд, жадібний вибір опорних точок, ваги з SVD
     матриці Лоуднера. Параметрів у AAA із `m` опорними точками ≈ `2m`,
     тож порівнюємо при РІВНІЙ кількості параметрів.

КОНТРОЛІ
  * ✅ `|t|` — випадок Ньюмана: раціональне МУСИТЬ бити поліном на порядки,
    інакше зламана реалізація AAA;
  * ✅ гладка `exp(−t²/4)` — обидва мусять збігатися швидко, раціональне
    БЕЗ драматичної переваги;
  * ✅ кусково-лінійна з K зламами (синтетика, K=1,4,16,64,256) — прямо
    показує, як перевага тане з ростом K;
  * похибка міряється на ВІДКЛАДЕНИХ точках, зважених гаусіаном.
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
NGRID = 20001
TMAX = 5.0
DEGS = [4, 8, 16, 32, 64, 128]


def aaa(Z, F, mmax, tol=1e-13):
    """AAA-апроксимація (барицентрична). Повертає функцію-обчислювач."""
    Z = np.asarray(Z, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    M = Z.size
    J = list(range(M))
    zj, fj, wj = np.empty(0), np.empty(0), np.empty(0)
    R = np.full(M, F.mean())
    C = np.empty((M, 0))
    for _ in range(mmax):
        j = int(np.argmax(np.abs(F - R)))
        if j in set(np.where(np.isin(Z, zj))[0]):
            break
        zj = np.append(zj, Z[j])
        fj = np.append(fj, F[j])
        J = [i for i in J if i != j]
        with np.errstate(divide='ignore', invalid='ignore'):
            col = 1.0 / (Z - Z[j])
        col[~np.isfinite(col)] = 0.0
        C = np.hstack([C, col.reshape(-1, 1)])
        A = (F[J].reshape(-1, 1) * C[J]) - (C[J] * fj.reshape(1, -1))
        if A.shape[0] < A.shape[1]:
            break
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        wj = Vh[-1]
        num = C[:, :] @ (wj * fj)
        den = C[:, :] @ wj
        with np.errstate(divide='ignore', invalid='ignore'):
            R = num / den
        R[~np.isfinite(R)] = F[~np.isfinite(R)] if False else 0.0
        for k, zz in enumerate(zj):
            R[np.isclose(Z, zz)] = fj[k]
        if np.max(np.abs(F - R)) < tol * np.max(np.abs(F)):
            break

    def ev(x):
        x = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x)
        with np.errstate(divide='ignore', invalid='ignore'):
            CC = 1.0 / (x.reshape(-1, 1) - zj.reshape(1, -1))
        bad = ~np.isfinite(CC)
        CC[bad] = 0.0
        num = CC @ (wj * fj)
        den = CC @ wj
        with np.errstate(divide='ignore', invalid='ignore'):
            out = num / den
        for k, zz in enumerate(zj):
            out[np.isclose(x, zz)] = fj[k]
        out[~np.isfinite(out)] = 0.0
        return out
    return ev, zj.size


def herm_fit(t, f, wgt, deg):
    """Найкращий поліном степеня `deg` у зваженому L² (ортогоналізація QR)."""
    V = np.polynomial.hermite_e.hermevander(t, deg)
    sw = np.sqrt(wgt).reshape(-1, 1)
    Q, R = np.linalg.qr(V * sw)
    c = Q.T @ (f * np.sqrt(wgt))
    return (V * sw) @ np.linalg.solve(R, c) / np.sqrt(wgt)


def werr(f, g, wgt):
    return float(np.sum(wgt * (f - g) ** 2) / np.sum(wgt * (f - np.average(
        f, weights=wgt)) ** 2))


def study(name, fvals, t, wgt, kinks=None):
    print(f'\n── {name}' + (f'   (зламів K={kinks})' if kinks is not None else ''))
    print(f'{"параметрів":>11s} {"поліном L²":>12s} {"раціон. AAA":>13s} '
          f'{"перевага":>10s}')
    for d in DEGS:
        p = herm_fit(t, fvals, wgt, d)
        ep = werr(fvals, p, wgt)
        ev, mm = aaa(t, fvals, max(2, d // 2))
        er = werr(fvals, ev(t), wgt)
        adv = ep / max(er, 1e-300)
        print(f'{d+1:11d} {ep:12.3e} {er:13.3e} {adv:9.1f}×')


def main(nets=2, dev=DEV):
    t = np.linspace(-TMAX, TMAX, NGRID)
    wgt = np.exp(-0.5 * t * t)
    wgt = wgt / wgt.sum()

    study('✅ КОНТРОЛЬ |t| (випадок Ньюмана)', np.abs(t), t, wgt, kinks=1)
    study('✅ КОНТРОЛЬ гладка exp(−t²/4)', np.exp(-0.25 * t * t), t, wgt, kinks=0)
    rng = np.random.default_rng(7)
    for K in (4, 16, 64, 256):
        b = np.sort(rng.uniform(-3, 3, K))
        sl = rng.standard_normal(K + 1) * 0.5
        f = np.zeros_like(t)
        cur = 0.0
        prev = -TMAX
        idx = 0
        for i, bb in enumerate(list(b) + [TMAX]):
            m = (t >= prev) & (t < bb)
            f[m] = cur + sl[i] * (t[m] - prev)
            if m.any():
                cur = cur + sl[i] * (bb - prev)
            prev = bb
            idx = i
        study(f'✅ КОНТРОЛЬ кусково-лінійна', f, t, wgt, kinks=K)

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    for n, fpath in enumerate(fs):
        sd0 = int(np.load(fpath)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            g = torch.Generator(device=dev).manual_seed(1234 + n)
            u = torch.randn(W_, device=dev, generator=g)
            u = u / u.norm()
            wp = torch.randn(W_, device=dev, generator=g)
            wp = wp - (wp @ u) * u                     # w ⊥ u ⟹ t ~ N(0,1)
            tt = torch.tensor(t, dtype=torch.float32, device=dev)
            Z = tt.unsqueeze(1) * u.unsqueeze(0) + wp.unsqueeze(0)
            nk = 0
            x = Z
            for i, w in enumerate(wc):
                p = x @ w
                s = (p > 0)
                nk += int((s[1:] != s[:-1]).sum())
                x = torch.relu(p)
            nw = x.shape[1]
            uo = torch.ones(nw, device=dev) / math.sqrt(nw)
            fv = (x @ uo).double().cpu().numpy()
        study(f'МЕРЕЖА {n+1} уздовж гаусової прямої', fv, t, wgt, kinks=nk)
        del W, wc
        torch.cuda.empty_cache()

    print('\n🔑 читати так: якщо на контролі |t| перевага величезна і ПАДАЄ з '
          'ростом K, а на мережі K — тисячі, то теорема Ньюмана до нас не '
          'дотягується при наших n, і ермітів спектр — діагноз ЗАДАЧІ, не приладу')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 2)
