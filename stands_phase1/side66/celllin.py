"""ІДЕЯ ЮЗЕРА: ЧИ МОЖНА З ДВОХ БЛИЗЬКИХ СЕМПЛІВ ЕКСТРАПОЛЮВАТИ РЕШТУ.

МЕХАНІЗМ, ЩО ДАЄ ІДЕЇ ШАНС. Мережа з `relu` КУСКОВО-ЛІНІЙНА: поки патерн
активацій не змінився, `f(z) = z @ J` ТОЧНО, де `J` — добуток ваг із
зафіксованими воротами. Арифметика на користь:
  * прогін одного семпла = 31 матвектор ≈ `62n²`;
  * побудова `J` = 31 матмул ≈ `62n³` (дорого, разово);
  * АЛЕ потім кожна точка комірки коштує `2n²` — у 31 раз дешевше.
⟹ якщо в комірку влазить >~256 ІНФОРМАТИВНО РІЗНИХ точок, це виграш.

ЧОМУ МОЖЕ НЕ СПРАЦЮВАТИ: комірка мала (8192 нейронів, кожен може перемкнутися),
а точки всередині неї майже тотожні ⟹ НЕ несуть нової інформації для
усереднення. Але `relu` НЕПЕРЕРВНИЙ: при перемиканні стрибає лише ПОХІДНА, не
значення ⟹ лінійне наближення деградує ПЛАВНО, і питання в ФОРМІ деградації.

ОЧІКУВАННЯ ДО ПРОГОНУ: точна лінійність до `δ≈1e-3‖z‖`; для НОВОЇ інформації
треба `δ~1`; на цій відстані похибка наближення буде порядку самого приросту.

ЩО МІРЯЄМО по сітці `δ` (у частках ‖z‖):
  1. частка ПЕРЕМКНЕНИХ нейронів (усі 32 шари);
  2. `‖Δf‖/‖f‖` — чи змінився вихід узагалі (чи точка «нова»);
  3. `‖Δf − δ·uJ‖ / ‖Δf‖` — ВІДНОСНА похибка лінійного наближення приросту.
🔑 ВИРІШУЄ ПУНКТ 3 ПРИ `δ`, ДЕ ПУНКТ 2 ПОРІВНЯННИЙ З РОЗКИДОМ МІЖ СЕМПЛАМИ.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget                              # noqa: E402

L_ = 32
DEV = 'cuda:3'
DELTAS = [1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 2.0]


def fwd_gates(z, wc):
    """Прогін + патерн воріт кожного шару."""
    x, g = z, []
    for w in wc:
        p = x @ w
        gg = (p > 0)
        g.append(gg)
        x = torch.relu(p)
    return x, g


def fwd_fixed(u, wc, g):
    """ЛІНІЙНЕ поширення `u` крізь ЗАФІКСОВАНІ ворота = `u @ J` точно."""
    x = u
    for w, gg in zip(wc, g):
        x = (x @ w) * gg.to(x.dtype)
    return x


def main(nets=3, npt=64, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    for n in range(nets):
        sd0 = int(np.load(fs[n])['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            n0 = wc[0].shape[0]
            g0 = torch.Generator(device=dev); g0.manual_seed(31 + n)
            z = torch.randn(npt, n0, generator=g0, device=dev)
            f0, gates = fwd_gates(z, wc)
            ntot = sum(int(x.numel()) for x in gates) / npt
            # РОЗКИД між НЕЗАЛЕЖНИМИ семплами — масштаб «нової інформації»
            spread = float((f0 - f0.mean(0, keepdim=True)).norm(dim=1).mean()
                           / f0.norm(dim=1).mean())
            print(f'\n=== мережа {n}: n={n0}, нейронів у ланцюгу {ntot:.0f}, '
                  f'розкид між НЕЗАЛЕЖНИМИ семплами ‖Δf‖/‖f‖ = {spread:.3f}')
            print(f'{"δ/‖z‖":>8s} {"перемкн.":>9s} {"‖Δf‖/‖f‖":>10s} '
                  f'{"похибка ЛІН":>12s} {"частка розкиду":>15s}')
            print('-' * 60)
            for d in DELTAS:
                u = torch.randn(npt, n0, generator=g0, device=dev)
                u = u / u.norm(dim=1, keepdim=True) * z.norm(dim=1,
                                                            keepdim=True)
                z2 = z + d * u
                f1, g1 = fwd_gates(z2, wc)
                sw = float(sum((a != b).double().sum() for a, b in
                               zip(gates, g1)) / (npt * ntot))
                df = f1 - f0
                lin = fwd_fixed(d * u, wc, gates)
                rel = float((df - lin).norm(dim=1).mean()
                            / df.norm(dim=1).mean().clamp_min(1e-30))
                mag = float(df.norm(dim=1).mean() / f0.norm(dim=1).mean())
                print(f'{d:8.4f} {sw:9.4f} {mag:10.4f} {rel:12.4f} '
                      f'{mag/spread:15.3f}')
        del W, wc
        torch.cuda.empty_cache()
    print('\n🎯 ЧИТАТИ ТАК: ідея жива ЛИШЕ якщо є рядок, де «частка розкиду» '
          '≈1 (точка справді НОВА), а «похибка ЛІН» ≪1 (наближення ще працює).')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
