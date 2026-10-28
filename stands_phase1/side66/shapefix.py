"""ПОНЕЙРОННА ПОПРАВКА ФОРМИ В `m` — ЧИ ЗМЕНШУЄ `‖b‖` ЗАМИКАННЯ. (С86)

## ЗВІДКИ
`shapeviz` дав: похибка форми ПОНЕЙРОННО корелює з ВИМІРЯНОЮ `γ₄` того самого
нейрона на **+0.735** (стабільно 0.69–0.80 по всій глибині). Плюс зміряно, що
Еджворт `−γ₄/24` СИСТЕМАТИЧНО ЗАВИЩУЄ: відношення зміряне/Еджворт падає
0.942 (шар 2) → 0.529 (шар 32).

## 🔑 ЧИМ ЦЕ ВІДРІЗНЯЄТЬСЯ ВІД `edgeksig.py` (×0.875) І `edgesamp.py` (×0.65)
1. поправка йде **ТІЛЬКИ В `m`**, а не в рекурсію `C`
   (`m` — вузьке місце ×3.51, `C` — лише ×1.17);
2. масштаб КАЛІБРУЄТЬСЯ, і саме завищення в 1.9× раніше не знали.
🚨 Обидва пункти були причинами минулих провалів. Якщо цей стенд теж упаде —
   клас закривається остаточно, вже без «а якщо інакше».

## ВАРІАНТИ МАСШТАБУ `c` у поправці `E|h| ← σ·√(2/π)·(1 − c·γ₄/24)`
  c=0     ⛔ПЛОМБА: мусить дати РІВНО нинішнє замикання
  c=1     чистий Еджворт (має бути ГІРШЕ — відтворення старого провалу)
  c=lin   `1 − 1.15·⟨γ₄⟩_шару` — масштаб НА ШАР (понейронно ламається:
          при `γ₄>0.87` міняє знак, зміряно корел 0.040 на шарах 21–31)
  c=orac  ⚠️оракул: МНК-масштаб на шар проти істини (верхня межа гілки)
🚨 `α` (глобальний множник `m_deep`) ПЕРЕПІДБИРАЄТЬСЯ для КОЖНОГО варіанту —
   інакше нинішній має фору (пастка, на якій уже спіймалися в С83).

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ:
   c=0 пломба 0.00e+00. c=1 дасть ГІРШЕ за базу (0.85–0.95×) — відтворить
   старий провал. c=lin: 1.10–1.25× — головна ставка. c=orac: 1.3–1.5×.
   Ризик: поправка в `m` тягне за собою `C` через `d`-член, і накопичення
   може все з'їсти, як і раніше. Шанс, що c=lin дасть ≥1.10: **40%**.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/shapefix.py [cuda:N] [NETS] [LOG2N]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = 1 << 14
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж, вибірка 2^{LOG2N} = {NS}, {DEV}\n', flush=True)


def sample_g4(W, dev, ns, batch, seed):
    """`γ₄` передактивацій ПОНЕЙРОННО, з вибірки. Два проходи."""
    M1 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    M2 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    d = 0
    while d < ns:
        b = min(batch, ns - d)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]; hd = h.double()
            M1[i] += hd.sum(0); M2[i] += (hd * hd).sum(0)
            a = torch.relu(h)
        d += b
    MU = [m / ns for m in M1]
    SG = [(m2 / ns - m * m).clamp_min(1e-30).sqrt() for m, m2 in zip(MU, M2)]
    K4 = [torch.zeros(W_, dtype=torch.float64, device=dev) for _ in range(L_)]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    d = 0
    while d < ns:
        b = min(batch, ns - d)
        a = torch.randn(b, W_, device=dev, generator=g)
        for i in range(L_):
            h = a @ W[i]
            t = (h.double() - MU[i]) / SG[i]
            t2 = t * t
            K4[i] += (t2 * t2).sum(0)
            a = torch.relu(h)
        d += b
    return [(k / ns) - 3.0 for k in K4]


def closure(W, dev, G4=None, mode='none', cvec=None):
    """Замикання. `mode`: none | const (c=cvec скаляр) | perlayer (cvec[i]).

    Поправка ЛИШЕ в `m`: `m ← μΦ + σφ·(1 − c·γ₄/24)` — це рівно поправка до
    `E|h|`, бо саме `σφ`-член її несе, а `μΦ` точний.
    """
    n0 = W[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if mode == 'none' or G4 is None:
            fac = torch.ones_like(s)
        else:
            c = cvec if mode == 'const' else cvec[i]
            fac = 1.0 - c * G4[i] / 24.0
        m = mp * Ph + s * ph * fac
        if i < len(W) - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp)*torch.outer(ph/s, ph/s)
            d = (mp*mp + s*s) * Ph + mp * s * ph - m * m
            C = C + torch.diag(d - C.diagonal())
    return m


def bnorm(m, T1, T2):
    """‖b‖²/‖T‖² з КРОС-ЦІЛЛЮ, з ОПТИМАЛЬНИМ α для цього варіанта."""
    a = float((m * (T1 + T2) * 0.5).sum() / (m * m).sum().clamp_min(1e-300))
    d1, d2 = a * m - T1, a * m - T2
    return float((d1 * d2).sum() / (T1 * T2).sum().clamp_min(1e-300)), a


ROWS = []
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    b1, b2 = TRUTH2[sd0]
    T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
    T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
    with torch.no_grad():
        G4 = sample_g4(W, DEV, NS, BATCH, 555 + n)
        gbar = torch.stack([g.mean() for g in G4])
        m0 = closure(W, DEV)
        mZ = closure(W, DEV, G4, 'const', 0.0)
        seal = float((m0 - mZ).abs().max())
        r0, a0 = bnorm(m0, T1, T2)
        r1, _ = bnorm(closure(W, DEV, G4, 'const', 1.0), T1, T2)
        clin = [float(1.0 - 1.15 * gbar[i]) for i in range(L_)]
        rl, _ = bnorm(closure(W, DEV, G4, 'perlayer', clin), T1, T2)
        # свіп глобального множника
        best, bc = 1e30, 0.0
        for c in np.arange(0.0, 1.55, 0.05):
            rr, _ = bnorm(closure(W, DEV, G4, 'const', float(c)), T1, T2)
            if rr < best:
                best, bc = rr, float(c)
        ROWS.append((r0, r1, rl, best, bc, seal, a0))
    print(f'  … {n+1}/{len(FS)} seed {sd0} · ⛔пломба c=0: {seal:.2e} · '
          f'база {r0:.4e} · c=lin {rl:.4e} · найкращий c={bc:.2f}', flush=True)
    del W
    torch.cuda.empty_cache()

A = np.array(ROWS)
np.save('/mnt/win_d/llm/White-Box/arena62/logs/shapefix.npy', A)
print(f'\n⛔ПЛОМБА c=0 ≡ нинішнє замикання: max {A[:,5].max():.2e} '
      f'(мусить бути 0.00e+00)')
print(f'\n{"варіант":<26s} {"‖b‖²/‖T‖²":>12s} {"× до бази":>10s}')
print('─' * 52)
b = A[:, 0].mean()
for k, nm in ((0, '⛔база (нинішнє)'), (1, 'c=1 чистий Еджворт'),
              (2, '🎯c=lin  1−1.15⟨γ₄⟩_шару'),
              (3, '⚠️оракул: найкращий c на мережу')):
    v = A[:, k].mean()
    print(f'{nm:<26s} {v:12.4e} {b/v:10.4f}')
print('─' * 52)
print(f'найкращий c по мережах: {A[:,4].mean():.3f} ± {A[:,4].std():.3f}   '
      f'(c=1 — чистий Еджворт, c=0.5 — вдвічі слабший)')
print(f'α по мережах: {A[:,6].mean():.5f} ± {A[:,6].std():.5f}   '
      f'(у сабміті 1+_ALPHA = 0.99249)')
print('\n🎯 ГЕЙТ: c=lin ≥ 1.10× ⟹ несемо в естиматор. Інакше клас ЗАКРИТО.')
