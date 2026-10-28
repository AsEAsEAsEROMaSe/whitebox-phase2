"""ІДЕЯ ЮЗЕРА: ТОЧНЕ ІНТЕГРУВАННЯ ВЗДОВЖ ПРЯМОЇ ЗАМІСТЬ ОКРЕМИХ ТОЧОК.

ВІДМІННІСТЬ ВІД ⛔`celllin.py`: там була ЕКСТРАПОЛЯЦІЯ (одна лінійна ділянка) і
вона провалилась. Тут — КУСКОВО-лінійне інтегрування: уздовж прямої `z(t)`
функція `f(t)` складається з відрізків, і ЗНАЙШОВШИ ВСІ ВУЗЛИ, інтеграл
`∫f(t)φ(t)dt` береться ТОЧНО. Це не наближення, а справжня «нескінченність
точок уздовж однієї осі» — тобто саме те, що просить юзер.

МАТЕМАТИКА (щоб оцінка лишалась НЕЗМІЩЕНОЮ). Розкладаємо `z = z_⊥ + (z·u)u`.
Умовно на `z_⊥` координата вздовж `u` є `N(0,1)` ⟹
    `E[f] = E_{z_⊥}[ ∫ f(z_⊥ + t·u) φ(t) dt ]`.
Внутрішній інтеграл беремо точно ⟹ це РАО-БЛЕКВЕЛЛ по одному напрямку:
дисперсія падає рівно на ту частку, яку пояснює вісь `u`.

ОЧІКУВАННЯ ДО ПРОГОНУ: для ВИПАДКОВОГО `u` знімається ~`1/256` дисперсії
(1.004×), а ціна прямої — сотні прогонів через кількість вузлів ⟹ програш
на ДВА порядки. ПОРІГ: якщо «еквівалент точок» > «ціна в точках» — ідея жива.

ЩО МІРЯЄМО:
  1. `перемикань` уздовж прямої — ЦІНА точного проходу (кожен вузол вимагає
     оновлення стану, ≈`n²` флопів проти `62n²` за повний прогін);
  2. `Var(g)/Var(f)` — скільки дисперсії ЛИШИЛОСЬ після точного інтегрування
     по `u` (беремо 64 вузли Гаусса–Ерміта як сурогат точного інтеграла);
  3. те саме для `u` = ГОЛОВНИЙ напрямок адьойнта (еф.ранг ≈8) — найкращий
     мислимий вибір осі.
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
NZ = 256          # скільки `z_⊥`
NQ = 64           # вузлів Гаусса–Ерміта на вісь


def fwd(z, wc):
    x = z
    for w in wc:
        x = torch.relu(x @ w)
    return x


def gates_of(z, wc):
    x, g = z, []
    for w in wc:
        p = x @ w
        g.append(p > 0)
        x = torch.relu(p)
    return g


def main(nets=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    xh, wh = np.polynomial.hermite_e.hermegauss(NQ)
    xh = torch.tensor(xh, dtype=torch.float32, device=dev)
    wh = torch.tensor(wh / wh.sum(), dtype=torch.float64, device=dev)

    for n in range(nets):
        sd0 = int(np.load(fs[n])['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            n0 = wc[0].shape[0]
            ntot = sum(int(w.shape[1]) for w in wc)
            g = torch.Generator(device=dev); g.manual_seed(11 + n)

            # ── 1. ЦІНА: скільки перемикань уздовж прямої ────────────────
            z1 = torch.randn(8, n0, generator=g, device=dev)
            z2 = torch.randn(8, n0, generator=g, device=dev)
            NT = 512
            ts = torch.linspace(0, 1, NT, device=dev)
            sw = 0
            prev = None
            for k in range(NT):
                zz = z1 + ts[k] * (z2 - z1)
                gg = gates_of(zz, wc)
                if prev is not None:
                    sw += int(sum((a != b).sum() for a, b in zip(gg, prev)))
                prev = gg
            sw_per_line = sw / 8.0

            # ── 2. СКІЛЬКИ ДАЄ вісь: Rao-Blackwell по одному напрямку ────
            z = torch.randn(NZ, n0, generator=g, device=dev)
            f0 = fwd(z, wc).double()
            var_f = float(f0.var(0, unbiased=True).sum())

            def var_after(u):
                u = u / u.norm()
                zp = z - (z @ u).unsqueeze(1) * u.unsqueeze(0)
                acc = torch.zeros(NZ, wc[-1].shape[1], dtype=torch.float64,
                                  device=dev)
                for q in range(NQ):
                    acc += wh[q] * fwd(zp + xh[q] * u.unsqueeze(0),
                                       wc).double()
                return float(acc.var(0, unbiased=True).sum())

            u_rand = torch.randn(n0, generator=g, device=dev)
            v_rand = var_after(u_rand)
            # ГОЛОВНИЙ напрямок адьойнта: перший сингулярний вектор `E[∇f]`
            zg = torch.randn(64, n0, generator=g, device=dev,
                             requires_grad=False)
            with torch.enable_grad():
                zz = zg.clone().requires_grad_(True)
                out = fwd(zz, wc).sum()
                gr, = torch.autograd.grad(out, zz)
            u_top = gr.mean(0)
            v_top = var_after(u_top)

        cost_line = sw_per_line / 62.0          # у ПОВНИХ прогонах
        eq_rand = var_f / max(v_rand, 1e-300)
        eq_top = var_f / max(v_top, 1e-300)
        print(f'\n=== мережа {n}: n={n0}, нейронів {ntot}')
        print(f'  перемикань уздовж прямої: {sw_per_line:.0f} '
              f'⟹ ціна ≈ {cost_line:.1f} повних прогонів')
        print(f'  ВИПАДКОВА вісь: лишилось дисперсії {v_rand/var_f:.4f} '
              f'⟹ еквівалент {eq_rand:.3f} точок')
        print(f'  ГОЛОВНА вісь адьойнта: лишилось {v_top/var_f:.4f} '
              f'⟹ еквівалент {eq_top:.3f} точок')
        print(f'  🎯 ВЕРДИКТ: пряма дає {eq_top:.2f} точок за ціну '
              f'{cost_line:.1f} ⟹ {eq_top/max(cost_line,1e-9):.4f}× від '
              f'простого семплювання')
        del W, wc
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
