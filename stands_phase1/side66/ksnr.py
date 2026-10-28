"""ЯКИЙ ПОРЯДОК КУМУЛЯНТА ВЗАГАЛІ ВИМІРНИЙ З РОБОЧОЇ ВИБІРКИ (С78).

`field.py` дав ОРАКУЛЬНУ стелю: гаус 1.04% → +κ₃ 0.16% → +κ₄ 0.040% → +κ₅ 0.017%
(запас 64.6×, нижче лідерових 0.02%). Але там κ бралися з 2^18 пар. У сабміті
вибірка — 4096 пар, і `κ̂` тим шумніші, чим вищий порядок.

🔑 ЦЕЙ СТЕНД МІРЯЄ S/N НАПРЯМУ: ‖κ_істина‖ / ‖κ̂ − κ_істина‖ на робочій вибірці.
   Порядок із S/N < 1 у поправку брати НЕ МОЖНА — він додасть більше шуму, ніж
   зніме зміщення (це рівно те, що наші `λ` роблять автоматично, ставлячи вагу
   в нуль; див. [[finding-lambda-absorbs-any-mechanism-you-fix]]).

📣 ПЕРЕДБАЧЕНО з арифметики моментів гаусіана при N=8192:
   κ₃: γ₃~1/√256=0.063, шум √(15/N)=0.043 ⟹ S/N ≈ 1.5 — ледь вимірно
   κ₄: γ₄~1/256=0.0039, шум √(96/N)=0.108 ⟹ S/N ≈ 0.04 — НЕ вимірно
   κ₅: γ₅~256^-1.5=2.4e-4, шум √(945/N)=0.34 ⟹ S/N ≈ 7e-4 — безнадійно
   Якщо так — жива лише поправка κ₃, а її стеля (0.16%) майже дорівнює нашій
   поточній (0.18%) ⟹ клас дає ~нуль на ПРАКТИЦІ попри стелю 64.6×.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 18
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
W_ = 256
D = 32
CH = 1 << 14

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _k(W, seed, npairs, whiten=False):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(6, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            pd = p.double()
            q = pd * pd
            Z[L][0] += pd.sum(0); Z[L][1] += q.sum(0)
            Z[L][2] += (q * pd).sum(0); Z[L][3] += (q * q).sum(0)
            Z[L][4] += (q * q * pd).sum(0); Z[L][5] += (q * q * q).sum(0)
            x = torch.relu(p)
        done += b
    N = float(2 * npairs)
    K3, K4, K5, K6, SS = [], [], [], [], []
    for L in range(D):
        a1, a2, a3, a4, a5, a6 = (Z[L][i] / N for i in range(6))
        s2 = (a2 - a1 * a1).clamp_min(1e-30)
        mu3 = a3 - 3.0 * a1 * a2 + 2.0 * a1 ** 3
        mu4 = a4 - 4.0 * a1 * a3 + 6.0 * a1 * a1 * a2 - 3.0 * a1 ** 4
        mu5 = (a5 - 5.0 * a1 * a4 + 10.0 * a1 * a1 * a3
               - 10.0 * a1 ** 3 * a2 + 4.0 * a1 ** 5)
        mu6 = (a6 - 6.0 * a1 * a5 + 15.0 * a1 * a1 * a4
               - 20.0 * a1 ** 3 * a3 + 15.0 * a1 ** 4 * a2 - 5.0 * a1 ** 6)
        m4c = mu4 - 3.0 * s2 * s2
        K3.append(mu3); K4.append(m4c)
        K5.append(mu5 - 10.0 * s2 * mu3)
        K6.append(mu6 - 15.0 * s2 * mu4 - 10.0 * mu3 * mu3 + 30.0 * s2 ** 3)
        SS.append(s2.sqrt())
    return K3, K4, K5, K6, SS


rows = []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        T3, T4, T5, T6, S = _k(W, 91_000 + 13 * n, 1 << LOGN)      # істина
        acc = np.zeros((4, 2))                                      # сигнал², шум²
        for d in range(DRAWS):
            H3, H4, H5, H6, _ = _k(W, 55_000 + 7*n + 1013*d, NPAIR, whiten=True)
            for i, (T, H) in enumerate(((T3,H3),(T4,H4),(T5,H5),(T6,H6))):
                # безрозмірні: κ_k / s^k
                for L in range(D):
                    sk = S[L] ** (i + 3)
                    t, h = T[L] / sk, H[L] / sk
                    acc[i, 0] += float((t * t).sum())
                    acc[i, 1] += float((h - t).pow(2).sum())
        rows.append(acc)
    del W
    torch.cuda.empty_cache()
    a = rows[-1]
    print(f'  … мережа {n+1}/{NETS}  S/N: κ₃={np.sqrt(a[0,0]/a[0,1]):.3f} '
          f'κ₄={np.sqrt(a[1,0]/a[1,1]):.3f} κ₅={np.sqrt(a[2,0]/a[2,1]):.4f} '
          f'κ₆={np.sqrt(a[3,0]/a[3,1]):.4f}',
          flush=True)

R = np.array(rows).sum(0)
print(f'\n{NETS} мереж × {DRAWS} жеребів по {2*NPAIR} семплів (відношення СУМ)\n')
print(f'{"кумулянт":>10s} {"‖сигнал‖":>12s} {"‖шум‖":>12s} {"S/N":>9s} {"вирок":>16s}')
print('-' * 64)
for i, nm in enumerate(('κ₃', 'κ₄', 'κ₅', 'κ₆')):
    sn = np.sqrt(R[i, 0] / R[i, 1])
    v = '🟢 вимірний' if sn > 1.0 else ('🟡 межа' if sn > 0.5 else '⛔ шум')
    print(f'{nm:>10s} {np.sqrt(R[i,0]):12.4e} {np.sqrt(R[i,1]):12.4e} '
          f'{sn:9.4f} {v:>16s}')
print(f'\n🔑 Оракульні стелі з `field.py`: +κ₃ 12.5× потребує κ₄, 64.6× потребує κ₅.')
print('   Порядок із S/N<1 у поправку брати НЕ МОЖНА.')
print('📊 БАЛ: 1.954e-07 — вимір прайсує клас Еджворта на ПРАКТИЦІ')
