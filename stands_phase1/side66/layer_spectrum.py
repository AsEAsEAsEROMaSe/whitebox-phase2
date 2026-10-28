"""ДЕ НАРОДЖУЄТЬСЯ ХВІСТ ВИСОКИХ СТЕПЕНІВ — ПОШАРОВИЙ СПЕКТР.

ЗВІДКИ. M37 §B: неберучі 10–16% дисперсії — це ЗЛАМИ ReLU (softplus β=4
обвалює хвіст у 3–7 разів). Питання юзера, яке лишилось невиміряним:
**на якому шарі цей хвіст з'являється?** Якщо він народжується в одному-двох
шарах — поправка локальна, і бюджет може зійтись там, де раніше не сходився.
Якщо росте рівномірно — жодної локалізації нема, і це ще один мур.

ЩО МІРЯЄМО. Для КОЖНОГО шару ℓ — спектр ермітових степенів по ВХОДУ `z`
сукупної активації, без вибору напрямку:
    C_ℓ(ρ) = Σ_j Cov( x_{ℓ,j}(z), x_{ℓ,j}(ρz+√(1−ρ²)z') )
нормовано на Σ_j Var(x_{ℓ,j}). Це інваріант шару, а не одного функціонала.
Далі — та сама процедура з РОЗДІЛЕННЯМ ПАРНОСТІ, що виграла суд у
`spec_referee.py` 5:0.

ЕФЕКТИВНО: обидва прогони (z і z_ρ) йдуть ОДНИМ батчем, накопичуються лише
суми по компонентах ⟹ пам'ять O(ширина), а не O(семпли×шари×ширина).

КОНТРОЛІ
  * ✅ шар 0 має ТОЧНО відомий спектр: `relu(w·z)` — одновимірна функція від
    гаусової змінної, її ермітів розклад класичний (a₁ = 1/2 рівно для
    центрованого relu стандартної нормалі, далі парні спадають як
    a_{2m} ~ [(2m−3)!!]²/(2m)!). Друкуємо поруч із виміром;
  * ✅ той самий прогін із softplus β=4 — хвіст мусить бути НИЖЧИМ на КОЖНОМУ
    шарі, інакше вимір не той;
  * ⚠️ `C_ℓ(0.99)/Var` друкується — мусить бути близько 1.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402
from degree_spectrum2 import spectrum_parity, RHOS, RP        # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NS = 131072
CH = 8192
TAPS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 28, 31]


def relu_exact_spectrum(kmax=14):
    """Ермітів спектр `relu(u)`, u~N(0,1), точно.

    relu = ½u + ½|u|;  |u| має лише ПАРНІ коефіцієнти.
    c_k = E[relu(u)·He_k(u)]/k!;  Var_k = c_k²·k!.
    Рахуємо c_k чисельно-точно через рекурсію моментів — але для
    контролю досить прямої квадратури Гауса-Ерміта високого порядку.
    """
    n = 400
    x, w = np.polynomial.hermite_e.hermegauss(n)
    f = np.maximum(x, 0.0)
    w = w / w.sum()
    he = [np.ones_like(x), x]
    for k in range(2, kmax + 1):
        he.append(x * he[-1] - (k - 1) * he[-2])
    m = float((w * f).sum())
    a = []
    for k in range(1, kmax + 1):
        c = float((w * f * he[k]).sum()) / math.factorial(k)
        a.append(c * c * math.factorial(k))
    v = float((w * f * f).sum()) - m * m
    return np.array(a) / v


def run(wc, dev, seed, act='relu', beta=4.0):
    g = torch.Generator(device=dev).manual_seed(seed)
    nl = len(wc)
    sa = {t: None for t in TAPS}
    sab = {t: 0.0 for t in TAPS}
    saa = {t: 0.0 for t in TAPS}
    sb = {t: None for t in TAPS}
    out = {t: [] for t in TAPS}

    def act_f(h):
        return (torch.relu(h) if act == 'relu'
                else torch.nn.functional.softplus(h, beta=beta))

    # ── крок 1: одномоментні статистики на z (спільні для всіх ρ)
    zs = []
    for _ in range(NS // CH):
        zs.append(torch.randn(CH, W_, device=dev, generator=g))
    zps = [torch.randn(CH, W_, device=dev, generator=g) for _ in zs]
    for z in zs:
        x = z
        for i in range(nl):
            x = act_f(x @ wc[i])
            if i in sa:
                xd = x.double()
                sa[i] = xd.sum(0) if sa[i] is None else sa[i] + xd.sum(0)
                saa[i] += float((xd * xd).sum())
    N = float(NS)
    ma = {t: sa[t] / N for t in TAPS}
    var = {t: saa[t] / N - float((ma[t] * ma[t]).sum()) for t in TAPS}

    # ── крок 2: перехресні моменти для кожного ρ
    cs = {t: [] for t in TAPS}
    for r in RHOS:
        s = math.sqrt(1.0 - r * r)
        cab = {t: 0.0 for t in TAPS}
        sbv = {t: None for t in TAPS}
        for z, zp in zip(zs, zps):
            xa, xb = z, r * z + s * zp
            for i in range(nl):
                xa = act_f(xa @ wc[i])
                xb = act_f(xb @ wc[i])
                if i in cab:
                    ad, bd = xa.double(), xb.double()
                    cab[i] += float((ad * bd).sum())
                    sbv[i] = bd.sum(0) if sbv[i] is None else sbv[i] + bd.sum(0)
        for t in TAPS:
            mb = sbv[t] / N
            cs[t].append(cab[t] / N - float((ma[t] * mb).sum()))
    res = {}
    for t in TAPS:
        a, rr = spectrum_parity(cs[t], var[t])
        res[t] = (a / var[t], cs[t][-1] / var[t], rr)
    return res


def main(nets=4, dev=DEV):
    torch.manual_seed(6060)
    ex = relu_exact_spectrum()
    print('✅ ТОЧНИЙ спектр relu(w·z) (шар 0 мусить це відтворити):')
    print('   a₁..a₆ = ' + ' '.join(f'{ex[k]:.3f}' for k in range(6)) +
          f'   ≥7={ex[6:].sum():.3f}   ≤3={ex[:3].sum():.3f}\n')

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    acc = {t: [] for t in TAPS}
    soft = {}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            r = run(wc, dev, 7000 + n)
            for t in TAPS:
                acc[t].append(r[t][0])
            if n == 0:
                soft = run(wc, dev, 7000 + n, act='soft', beta=4.0)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … мережа {n+1}/{nets}', flush=True)

    print(f'\nПОШАРОВИЙ СПЕКТР, {nets} мереж × {NS} семплів')
    print(f'{"шар":>4s} {"a₁":>7s} {"a₂":>7s} {"a₃":>7s} {"a₄":>7s} {"a₅":>7s} '
          f'{"≤3":>7s} {"ХВІСТ≥6":>9s} {"±":>7s} {"стеля d=3":>10s}')
    for t in TAPS:
        A = np.array(acc[t])
        m = A.mean(0)
        tail = A[:, 5:].sum(1)
        t3 = A[:, 3:].sum(1).mean()
        print(f'{t:4d} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f} {m[3]:7.3f} '
              f'{m[4]:7.3f} {m[:3].sum():7.3f} {tail.mean():9.3f} '
              f'{tail.std():7.3f} {1.0/max(t3,1e-9):9.2f}×')

    print(f'\n✅ КОНТРОЛЬ softplus β=4 на мережі 1 — хвіст мусить бути НИЖЧИМ '
          f'на кожному шарі:')
    print(f'{"шар":>4s} {"relu ≥6":>9s} {"softplus ≥6":>12s}')
    for t in TAPS:
        rr = np.array(acc[t])[0][5:].sum()
        ss = soft[t][0][5:].sum()
        print(f'{t:4d} {rr:9.3f} {ss:12.3f}')
    print('\n🔑 читати так: якщо ХВІСТ≥6 стрибає на 1-2 шарах — негладкість '
          'ЛОКАЛЬНА і туди можна цілити бюджет; якщо росте рівномірно — '
          'локалізації нема, це ще один мур')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
