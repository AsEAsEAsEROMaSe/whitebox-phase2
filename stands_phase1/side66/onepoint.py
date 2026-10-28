"""ІДЕЯ ЮЗЕРА: БЕКПРОПОМ ЗНАЙТИ ВХІД `z*`, ЧИЙ ВИХІД = ТОЧНЕ СЕРЕДНЄ.

ПИТАННЯ. Чи існує ОДНА точка `z*` з `relu(h₃₁(z*)) = m_іст`? Якщо так — які в
неї властивості, і чи вона хоч теоретично передбачувана з ваг (бо на сабміті
`m_іст` невідоме — ми ним КОРИСТУЄМОСЬ, щоб її знайти).

🔑 СТРУКТУРА ЗАДАЧІ: вхід `n` чисел, вихід `n` чисел ⟹ рівнянь стільки ж,
скільки невідомих ⟹ розв'язок МАЄ існувати локально. Тому цікаве не «чи
знайдеться», а ЯКИМ він буде.

ОЧІКУВАННЯ ДО ПРОГОНУ: збіжність <1% відносної похибки · `‖z*‖ ~ √n` ·
з РІЗНИХ стартів РІЗНІ `z*` (множина розв'язків, не точка) ⟹ саме це вирішує,
чи можна його передбачати.

ЩО МІРЯЄМО (аналіз, а не виграш):
  1. відносна похибка `‖relu(h(z*))−m‖/‖m‖`;
  2. `‖z*‖/√n` — чи типова це точка для `N(0,I)`;
  3. косинуси МІЖ розв'язками з різних стартів — одна точка чи многовид;
  4. частка активних нейронів шару 1 при `z*` (для гаусового входу ≈0.5);
  5. ⚠️ГОЛОВНЕ: чи `relu(h(z*))` для СУСІДНЬОЇ мережі лишається близьким до її
     `m` — тобто чи `z*` несе щось про ЗАДАЧУ, а не лише про конкретну мережу.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget                              # noqa: E402
from tilt_wide import truth_last                                  # noqa: E402

L_ = 32
DEV = 'cuda:3'
STEPS = 3000


def fwd(z, wc):
    x = z
    for i, w in enumerate(wc):
        p = x @ w
        x = torch.relu(p)
    return x


def solve(wc, tgt, seed, dev, steps=STEPS, lr=0.05):
    n0 = wc[0].shape[0]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    z = torch.randn(1, n0, generator=g, device=dev, dtype=torch.float32)
    z = z.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    tg = tgt.float().unsqueeze(0)
    for it in range(steps):
        opt.zero_grad()
        out = fwd(z, wc)
        loss = ((out - tg) ** 2).sum()
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = fwd(z, wc)
        rel = float((out - tg).norm() / tg.norm())
    return z.detach()[0], rel


def main(nets=4, starts=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets + 1]

    print(f'{"мережа":>7s} {"старт":>6s} {"відн.похибка":>13s} '
          f'{"‖z*‖/√n":>9s} {"актив.шар1":>11s} {"cos з №0":>9s}')
    print('-' * 62)
    keep = []
    for n in range(nets):
        sd0 = int(np.load(fs[n])['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _npair = cut_and_budget(W, dev)
            T = truth_last(wc, dev, 510_000 + 7919 * n).float()
        n0 = wc[0].shape[0]
        zs = []
        for st in range(starts):
            z, rel = solve(wc, T, 1000 + 137 * st, dev)
            with torch.no_grad():
                act = float(((z.unsqueeze(0) @ wc[0]) > 0).float().mean())
                cs = (float(torch.nn.functional.cosine_similarity(
                    z, zs[0], dim=0)) if zs else 1.0)
            zs.append(z)
            print(f'{n:7d} {st:6d} {rel:13.5f} {float(z.norm())/np.sqrt(n0):9.3f} '
                  f'{act:11.3f} {cs:9.3f}')
        keep.append((wc, T, zs[0]))
        del W
        torch.cuda.empty_cache()

    # ⚠️ ГОЛОВНЕ: чи `z*` однієї мережі щось означає для ІНШОЇ
    print('\n⚠️ ПЕРЕНОС `z*` НА ІНШУ МЕРЕЖУ (мусить розвалитись, якщо `z*` — '
          'властивість МЕРЕЖІ, а не задачі):')
    print(f'{"z* з мережі":>12s} {"застосовано до":>15s} {"відн.похибка":>13s}')
    print('-' * 45)
    for a in range(min(2, nets)):
        for b in range(min(2, nets)):
            wc_b, T_b, _ = keep[b]
            if keep[a][2].shape[0] != wc_b[0].shape[0]:
                print(f'{a:12d} {b:15d} {"розмір різний":>13s}')
                continue
            with torch.no_grad():
                out = fwd(keep[a][2].unsqueeze(0), wc_b)
                rel = float((out - T_b.unsqueeze(0)).norm() / T_b.norm())
            print(f'{a:12d} {b:15d} {rel:13.5f}')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
