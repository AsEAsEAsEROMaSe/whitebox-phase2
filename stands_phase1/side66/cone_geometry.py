"""ЯКА ГЕОМЕТРІЯ КОНУСА, У ЯКОМУ ЛЕЖИТЬ СЕМПЛ. (бічний стенд, ідея юзера «вести з кінця»)

ПИТАННЯ. Мережа без зсувів кусково-лінійна: для фіксованої маски затворів
`a = z·A_M`, де `A_M = W₀D₀W₁D₁…W₃₁D₃₁`. Якщо йти ВІД КІНЦЯ, треба знати,
скільки свободи має прообраз і чи широкий конус. Два числа вирішують:

1. РАНГ `A_M`. Скільки вимірів `z` узагалі впливають на вихід. Решта —
   нуль-простір: там `a` СТАЛЕ, і по них можна інтегрувати аналітично,
   а не семплити. Друкуємо і формальний ранг, і стабільний (participation).

2. ШИРИНА КОНУСА вздовж цих «порожніх» напрямків: як далеко можна відійти,
   поки не перемкнеться ПЕРШИЙ затвор. Рахуємо точно: рухаючись `z + t·v`,
   доактиваційне `p` рухається як `p + t·q`, де `q` — той самий прогін для `v`
   зі ЗАМОРОЖЕНИМИ масками ⟹ перемикання при `t* = min |p| / |q|`.
   Якщо `t*` велике проти масштабу `‖z‖=√256=16` — конус широкий, є що
   інтегрувати. Якщо мале — семпл сидить майже на межі, і «плоских» ділянок
   для аналітики просто нема.

КОНТРОЛІ
  * порівнюємо напрямок із НУЛЬ-ПРОСТОРУ проти ВИПАДКОВОГО і проти напрямку
    з ГОЛОВНОГО підпростору — якщо всі три дають одну ширину, геометрія
    конуса не має «зручних» напрямків узагалі;
  * перевіряємо ТОТОЖНІСТЬ `a(z+tv) = a(z)` прямим прогоном при `t = 0.9·t*`
    (мусить збігтися до машинного нуля) і при `t = 1.1·t*` (мусить зламатись)
    — це доводить, що `t*` знайдено правильно;
  * контроль Ейлера `z·A_M = a(z)` — перевірка, що `A_M` зібрано правильно.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
from transport_r2 import cut_net                              # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'


def forward(wc, z):
    """Прогін; повертає активації, доактиваційні `p` і маски."""
    x, ps, ds = z, [], []
    for i in range(len(wc)):
        p = x @ wc[i]
        ps.append(p)
        ds.append((p > 0).to(p.dtype))
        x = torch.relu(p)
    return x, ps, ds


def linear_dir(wc, v, ds):
    """Той самий прогін для напрямку `v` зі ЗАМОРОЖЕНИМИ масками ⟹ dp/dt."""
    x, qs = v, []
    for i in range(len(wc)):
        q = x @ wc[i]
        qs.append(q)
        x = q * ds[i]
    return qs


def amat(wc, ds):
    """`A_M = W₀D₀ … W₃₁D₃₁` для фіксованої маски."""
    A = wc[0] * ds[0]
    for i in range(1, len(wc)):
        A = A @ (wc[i] * ds[i])
    return A


def main(nets=4, nsamp=24, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    rnk_f, rnk_s, wid, eul, chk = [], [], {'нуль': [], 'випадк': [], 'головн': []}, [], []
    for k, f in enumerate(fs):
        s = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[s]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            gen = torch.Generator(device=dev)
            gen.manual_seed(4242 + k)
            for _ in range(nsamp):
                z = torch.randn(1, W_, generator=gen, device=dev)
                a, ps, ds = forward(wc, z)
                A = amat(wc, [d[0] for d in ds]).double()
                eul.append(float((z.double() @ A - a.double()).abs().max()
                                 / a.double().abs().max()))
                U, S, Vh = torch.linalg.svd(A)
                smax = float(S[0])
                rnk_f.append(int((S > 1e-7 * smax).sum()))
                rnk_s.append(float(S.sum() ** 2 / (S * S).sum()))

                dirs = {'нуль': U[:, -1], 'головн': U[:, 0],
                        'випадк': torch.randn(W_, generator=gen, device=dev)}
                for nm, v in dirs.items():
                    v = (v / v.norm()).float().reshape(1, W_)
                    qs = linear_dir(wc, v, [d for d in ds])
                    # 🚨 до межі наближають ЛИШЕ ті затвори, де p і q різних знаків;
                    #    решту рух ВІДДАЛЯЄ ⟹ їх у мінімум брати не можна
                    t = min(float(torch.where(p * q < 0,
                                              p.abs() / q.abs().clamp_min(1e-30),
                                              torch.full_like(p, float('inf'))).min())
                            for p, q in zip(ps, qs))
                    wid[nm].append(t)
                    if nm == 'нуль':
                        a1, _, d1 = forward(wc, z + 0.9 * t * v)
                        a2, _, d2 = forward(wc, z + 1.1 * t * v)
                        same = all(bool((x == y).all()) for x, y in zip(ds, d1))
                        chk.append((float((a1 - a).abs().max() / a.abs().max()),
                                    same,
                                    all(bool((x == y).all()) for x, y in zip(ds, d2))))
        print(f'  мережа {k+1}/{len(fs)}', flush=True)

    print(f'\n{len(fs)} мереж × {nsamp} семплів')
    print(f'контроль Ейлера |z·A_M − a|/|a| : {np.mean(eul):.2e} (мусить бути ~1e-7)')
    print(f'формальний ранг A_M            : {np.mean(rnk_f):6.1f} із 256')
    print(f'стабільний ранг (participation): {np.mean(rnk_s):6.2f}')
    print(f'\nШИРИНА КОНУСА (масштаб ‖z‖ = {math.sqrt(W_):.1f})')
    print(f'{"напрямок":>10s} {"медіана t*":>11s} {"середнє":>9s} {"t*/‖z‖":>9s}')
    print('-' * 44)
    for nm in ('нуль', 'головн', 'випадк'):
        v = np.array(wid[nm])
        print(f'{nm:>10s} {np.median(v):11.4f} {v.mean():9.4f} '
              f'{np.median(v)/math.sqrt(W_):9.5f}')
    c = np.array([x[0] for x in chk])
    print(f'\nТОТОЖНІСТЬ a(z+0.9t*v) = a(z): {c.mean():.2e} (мусить бути ~0)')
    print(f'маска ціла при 0.9·t*: {np.mean([x[1] for x in chk])*100:.0f}% '
          f'· при 1.1·t*: {np.mean([x[2] for x in chk])*100:.0f}% (мусить бути ~0%)')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
         nsamp=int(sys.argv[2]) if len(sys.argv) > 2 else 24)
