"""ЕДЖВОРТ НА РОБОЧІЙ ВИБІРЦІ: чи це виграш, чи переформулювання задачі (С78).

🚨 ПАСТКА, ЯКУ ЦЕЙ СТЕНД МАЄ РОЗВ'ЯЗАТИ. `field.py` дав стелю 151.5×, але там
`m, s, κ` бралися з 2^18 пар — ТОЧНІ. «Знати моменти точно» — це і є вся наша
задача, тож стеля могла виявитись не новим шляхом, а її переформулюванням.

Тут обидві оцінки будуються з ОДНІЄЇ І ТІЄЇ САМОЇ робочої вибірки (4096 пар,
антитетика, ZCA — рівно як у сабміті), і порівнюються з крос-істиною:

  A  пряма:      m̂ = mean(relu(p₃₁))            ← те, що робить сабміт
  B₃..B₆ Еджворт: Ê(m̂p, ŝ, κ̂₃…κ̂₆) на p₃₁        ← гладка функція 6 моментів

Обидві незміщені (з точністю до обрізання ряду), обидві з тих самих даних ⟹
різниця ЛИШЕ в дисперсії. Якщо Ê виграє — це прямий, майже безкоштовний бал
(кілька редукцій на шар). Якщо ні — клас Еджворта закритий на практиці попри
стелю, і причина буде названа числом.

📣 ПЕРЕДБАЧЕНО: `m̂p₃₁ = mean(x₃₀)·W₃₁` несе ТОЙ САМИЙ накопичений шум, що й
   `mean(relu(p₃₁))`. Еджворт моделює краще лише ОСТАННІЙ крок relu, а той
   дає ~2% MSE (`gcsplit.py`). Ставлю на виграш ≲1.02×. Якщо так — стеля 151.5×
   належить класу «точні моменти», тобто це та сама ціль, що й точна голова
   з [[finding-error-is-additive-across-all-32-layers]], а не нова.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
W_, D, CH = 256, 32, 1 << 14
S2PI = float(1.0 / np.sqrt(2 * np.pi))
SQ2 = float(np.sqrt(2.0))

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _run(W, seed, npairs, whiten=False):
    """Моменти 1..6 ОСТАННЬОГО шару і пряме mean(relu) там само."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = torch.zeros(7, W_, dtype=torch.float64, device=DEV)
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        x = torch.cat([z, -z], 0)
        for L in range(D - 1):
            x = torch.relu(x @ W[L])
        p = x @ W[D - 1]
        pd = p.double()
        q = pd * pd
        Z[0] += pd.sum(0); Z[1] += q.sum(0); Z[2] += (q * pd).sum(0)
        Z[3] += (q * q).sum(0); Z[4] += (q * q * pd).sum(0)
        Z[5] += (q * q * q).sum(0)
        Z[6] += torch.relu(pd).sum(0)
        done += b
    N = float(2 * npairs)
    a1, a2, a3, a4, a5, a6 = (Z[i] / N for i in range(6))
    s2 = (a2 - a1 * a1).clamp_min(1e-30)
    mu3 = a3 - 3 * a1 * a2 + 2 * a1 ** 3
    mu4 = a4 - 4 * a1 * a3 + 6 * a1 * a1 * a2 - 3 * a1 ** 4
    mu5 = a5 - 5 * a1 * a4 + 10 * a1 * a1 * a3 - 10 * a1 ** 3 * a2 + 4 * a1 ** 5
    mu6 = (a6 - 6 * a1 * a5 + 15 * a1 * a1 * a4 - 20 * a1 ** 3 * a3
           + 15 * a1 ** 4 * a2 - 5 * a1 ** 6)
    return dict(m=a1, s=s2.sqrt(), k3=mu3, k4=mu4 - 3 * s2 * s2,
                k5=mu5 - 10 * s2 * mu3,
                k6=mu6 - 15 * s2 * mu4 - 10 * mu3 * mu3 + 30 * s2 ** 3,
                direct=Z[6] / N)


def _edge(M):
    """Ряд Еджворта до 4-го порядку ⟹ список оцінок E[relu] (гаус, κ₃…κ₆)."""
    m, s = M['m'], M['s']
    t = m / s
    s2 = s * s; s3 = s2 * s; s4 = s2 * s2; s5 = s4 * s; s7 = s5 * s2
    Ph = 0.5 * (1.0 + torch.erf(t / SQ2))
    ph = torch.exp(-0.5 * t * t) * S2PI
    t2 = t * t; t3 = t2 * t; t4 = t2 * t2; t6 = t4 * t2; t8 = t4 * t4
    He3 = t3 - 3 * t
    He4 = t4 - 6 * t2 + 3
    He5 = t3 * t2 - 10 * t3 + 15 * t
    He6 = t6 - 15 * t4 + 45 * t2 - 15
    He7 = t6 * t - 21 * t3 * t2 + 105 * t3 - 105 * t
    He8 = t8 - 28 * t6 + 210 * t4 - 420 * t2 + 105
    He10 = (t8 * t2 - 45 * t8 + 630 * t6 - 3150 * t4 + 4725 * t2 - 945)
    k3, k4, k5, k6 = M['k3'], M['k4'], M['k5'], M['k6']
    o0 = m * Ph + s * ph
    o1 = o0 - k3 * t * ph / (6 * s2)
    o2 = (o1 + k4 * ph * (t2 - 1) / (24 * s3)
          + k3 * k3 * ph * He4 / (72 * s5))
    o3 = (o2 - k5 * ph * He3 / (120 * s4)
          - k3 * k4 * ph * He5 / (144 * s5 * s)
          - k3 ** 3 * ph * He7 / (1296 * s4 * s4))
    o4 = (o3 + k6 * ph * He4 / (720 * s5)
          + k3 * k5 * ph * He6 / (720 * s7)
          + k4 * k4 * ph * He6 / (1152 * s7)
          + k3 * k3 * k4 * ph * He8 / (1728 * s7 * s2)
          + k3 ** 4 * ph * He10 / (31104 * s7 * s4))
    return [o0, o1, o2, o3, o4]


NAMES = ('гаус(m,s)', '+κ₃', '+κ₄', '+κ₅', '+κ₆')
acc = np.zeros(6)          # пряма + 5 порядків
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        tA = _run(W, 91_000 + 13 * n, 1 << LOGN)['direct']
        tB = _run(W, 40_000 + 29 * n, 1 << LOGN)['direct']
        for d in range(DRAWS):
            M = _run(W, 55_000 + 7 * n + 1013 * d, NPAIR, whiten=True)
            cand = [M['direct']] + _edge(M)
            for i, c in enumerate(cand):
                acc[i] += float(((c - tA) * (c - tB)).sum())    # КРОС-ЦІЛЬ
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}  пряма={acc[0]:.4e}  '
          f'+κ₆={acc[5]:.4e}  ⟹ {acc[0]/max(acc[5],1e-30):.4f}×', flush=True)

print(f'\n{NETS} мереж × {DRAWS} жеребів по {2*NPAIR} семплів (відношення СУМ)\n')
print(f'{"оцінка":>14s} {"MSE":>14s} {"проти прямої":>14s}')
print('-' * 46)
print(f'{"A пряма":>14s} {acc[0]:14.5e} {1.0:13.4f}×')
for i, nm in enumerate(NAMES):
    print(f'{"B " + nm:>14s} {acc[i+1]:14.5e} {acc[0]/max(acc[i+1],1e-30):13.4f}×')
best = acc[1:].min()
verdict = ('🟢 ВИГРАШ' if best < acc[0] else
           '⛔ ПРОГРАШ — стеля належить класу «точні моменти», не новому методу')
print(f'\n🔑 НАЙКРАЩИЙ ЕДЖВОРТ: {acc[0]/max(best,1e-30):.4f}×  ({verdict})')
print('📊 БАЛ: 1.954e-07 — це прайс каналу, не сабміт')
