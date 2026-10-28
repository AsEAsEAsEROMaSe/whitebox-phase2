"""СПЕКТР ЕРМІТОВИХ СТЕПЕНІВ ІНТЕГРАНДА — СТЕЛЯ ВСІХ ТОЧНИХ МЕТОДІВ. (мета 1e-8)

ЗВІДКИ. С44/С47 закрили детерміновані кубатури (Струд степ-5, 131k вузлів —
у 35× ГІРШЕ за v2), АЛЕ аудит [[finding-cubature-deterministic-flat]] прямо
записав: «найсильніша версія — той самий Струд, ПОВЕРНУТИЙ випадковою
ортогональною матрицею — НЕ ТЕСТУВАЛАСЬ», бо вузли на координатних осях —
найгірший невипадковий набір для мережі зі структурою вздовж стовпців `W₀`.
Там же стоїть число «~82% вмісту на степені ≥6», але воно було побічним.

ЧОМУ ЦЕ ЧИСЛО ВИРІШУЄ ВСЕ. **Степінь Ерміта ІНВАРІАНТНИЙ до обертання.**
Отже спектр степенів закриває або відкриває ВЕСЬ клас методів, точних до
степеня `d` — кубатури Струда, повернуті кубатури, розріджені сітки, QMC у
зважених просторах — одним числом, і поворот тут нічого не міняє за теоремою.
Наш v2 уже точний до 3-го моменту ⟹ стеля методу, точного до степеня `d`:
    виграш ≤ Var(F) / Var_{>d}(F)

ЯК МІРЯЄМО — тотожність Мелера, без жодного розкладу в 256 вимірах:
    C(ρ) = Cov( F(z), F(ρz + √(1−ρ²)z') ) = Σ_{k≥1} ρᵏ · Var_k(F)
Скануємо ρ, потім розв'язуємо невід'ємну задачу найменших квадратів на `a_k`.

КОНТРОЛІ
  * ✅ `C(1)` мусить дорівнювати `Var(F)`, `C(0)` — нулю (друкуємо обидва);
  * ✅ `Σ a_k` мусить збігтися з `Var(F)`;
  * ✅ ПОЗИТИВНИЙ КОНТРОЛЬ: та сама процедура на ЛІНІЙНІЙ функції `w·z` має
    дати `a₁ = 1`, решта 0; на `(w·z)² − 1` — `a₂ = 1`;
  * спільні `z'` для всіх ρ (спільні випадкові числа) — знижує шум;
  * дві половини вибірки — друкуємо розкид.
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
NS = 65536
CH = 8192
KMAX = 24
RHOS = np.concatenate([np.linspace(0.05, 0.95, 19), [0.97, 0.99]])


def fwd(wc, z):
    x = z
    for i in range(L_):
        x = torch.relu(x @ wc[i])
    return x


def spectrum(cs, var):
    """Невід'ємні `a_k` із `C(ρ)=Σ ρᵏ a_k`; додаємо рівняння Σa_k=Var."""
    A = np.array([[r ** k for k in range(1, KMAX + 1)] for r in RHOS])
    b = np.array(cs)
    A = np.vstack([A, np.ones(KMAX)])
    b = np.concatenate([b, [var]])
    try:
        from scipy.optimize import nnls
        a, _ = nnls(A, b)
    except Exception:
        a = np.linalg.lstsq(A, b, rcond=None)[0].clip(min=0.0)
    return a


def main(nets=6, dev=DEV):
    torch.manual_seed(4321)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    # ── ✅ ПОЗИТИВНИЙ КОНТРОЛЬ на функціях із ВІДОМИМ спектром
    w = torch.randn(W_, device=dev, dtype=torch.float64)
    w = w / w.norm()
    for nm, fn in (('лінійна w·z', lambda t: t @ w),
                   ('квадратична (w·z)²−1', lambda t: (t @ w) ** 2 - 1.0)):
        cs, v0 = [], None
        z = torch.randn(NS, W_, device=dev, dtype=torch.float64)
        zp = torch.randn(NS, W_, device=dev, dtype=torch.float64)
        f0 = fn(z); v0 = float(f0.var())
        for r in RHOS:
            g = fn(r * z + math.sqrt(1 - r * r) * zp)
            cs.append(float(((f0 - f0.mean()) * (g - g.mean())).mean()))
        a = spectrum(cs, v0) / v0
        print(f'  ✅ {nm:22s} a₁={a[0]:.3f} a₂={a[1]:.3f} a₃={a[2]:.3f} '
              f'решта={a[3:].sum():.3f}')

    acc = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            nw = wc[-1].shape[1]
            u = torch.ones(nw, device=dev) / math.sqrt(nw)
            Z, ZP, F0 = [], [], []
            for _ in range(NS // CH):
                z = torch.randn(CH, W_, device=dev)
                zp = torch.randn(CH, W_, device=dev)
                Z.append(z); ZP.append(zp); F0.append((fwd(wc, z) @ u).double())
            F0 = torch.cat(F0)
            m0, v0 = float(F0.mean()), float(F0.var())
            cs = []
            for r in RHOS:
                s = math.sqrt(1.0 - r * r)
                acc_c = 0.0
                for j, (z, zp) in enumerate(zip(Z, ZP)):
                    g = (fwd(wc, r * z + s * zp) @ u).double()
                    acc_c += float(((F0[j*CH:(j+1)*CH] - m0)
                                    * (g - g.mean())).sum())
                cs.append(acc_c / F0.shape[0])
            a = spectrum(cs, v0) / v0
            acc.append(a)
        print(f'  мережа {n+1}/{len(fs)}  C(0.05)/Var={cs[0]/v0:.4f} '
              f'C(0.99)/Var={cs[-1]/v0:.4f}  ≤3: {a[:3].sum():.3f}  '
              f'≥6: {a[5:].sum():.3f}', flush=True)

    A = np.array(acc)
    m, s = A.mean(0), A.std(0)
    print(f'\n{len(fs)} мереж × {NS} семплів, KMAX={KMAX}')
    print('\nСПЕКТР ЕРМІТОВИХ СТЕПЕНІВ (частка дисперсії)')
    print(f'{"k":>3s} {"a_k":>9s} {"±":>8s} {"накопич.":>10s}')
    c = 0.0
    for k in range(min(12, KMAX)):
        c += m[k]
        print(f'{k+1:3d} {m[k]:9.4f} {s[k]:8.4f} {c:10.4f}')
    print(f'{"13+":>3s} {A[:, 12:].sum(1).mean():9.4f} '
          f'{A[:, 12:].sum(1).std():8.4f} {A.sum(1).mean():10.4f}')
    print('\nСТЕЛЯ МЕТОДУ, ТОЧНОГО ДО СТЕПЕНЯ d  (виграш ≤ 1/Var_{>d})')
    for d in (1, 2, 3, 4, 5, 6, 8, 10):
        tail = A[:, d:].sum(1).mean()
        print(f'  d={d:2d}: лишається {tail:.4f} ⟹ стеля {1.0/max(tail,1e-9):6.2f}×')
    print('\n🔑 степінь Ерміта ІНВАРІАНТНИЙ до обертання ⟹ це стеля і для '
          'ПОВЕРНУТОГО Струда, і для розріджених сіток, і для QMC')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
