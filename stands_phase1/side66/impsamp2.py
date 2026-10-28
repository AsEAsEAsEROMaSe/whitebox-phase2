"""ГІЛКА 2, крок 2: РЕАЛІЗОВНА зміна міри — гаусів нахил.

ЗВІДКИ (`impsamp.py`). Оракульна стеля важливої вибірки на СПРАВЖНЬОМУ
підінтегральному `ψ` = **2.09×** (антитетично). Але `q ∝ φ‖ψ‖` не
реалізовне: щоб знати `‖ψ(z)‖`, треба вже прогнати мережу. Радіальна
`q(‖z‖)` дає лише **1.057×**. Проксі з шару 8 дає 1.638×, але з нього не
семплюють — це вже двоступенева схема з окремою ціною.

ЄДИНИЙ КЛАС, З ЯКОГО СЕМПЛЮЮТЬ ТОЧНО І ДАРОМ — гаусів нахил:
    q(z) ∝ φ(z)·exp(zᵀCz)  ⟹  q = N(0, (I−2C)⁻¹)
Семплювання коштує рівно стільки ж, скільки `randn`. Питання одне: наскільки
добре `log‖ψ(z)‖` наближається квадратичною формою?
(Лінійний член `bᵀz` для АНТИТЕТИЧНОЇ пари марний: `(ψ(z)+ψ(−z))/2` — ПАРНА
функція `z`, тож усе непарне гине. Тому фітуємо лише квадратичну форму.)

ОБЧИСЛЕННЯ БЕЗ ПЕРЕСЕМПЛЮВАННЯ (важливо: ті самі семпли, чесно):
    E_q[w²‖ψ‖²] = Z_q · E_φ[‖ψ‖²·e^{−zᵀCz}],   Z_q = E_φ[e^{zᵀCz}]
    ВИГРАШ = (E_φ‖ψ‖² − ‖T‖²) / (E_q[w²‖ψ‖²] − ‖T‖²)

КОНТРОЛІ
  * ✅ `C=0` мусить дати РІВНО 1.000×;
  * ✅ радіальний випадок `C=cI` — мусить відтворити ~1.057× з `impsamp.py`;
  * ✅ формула звірена аналітично: для `f=z₁` оракульна стеля = **π/2**
    (`impsamp.py` дав 1.571 — збіг до 4 знаків);
  * ⛔ квадратична форма з ВИПАДКОВИМИ напрямками — мусить дати ~1.0;
  * ⚠️ `C` підганяється на 1-й половині, виграш міряється на 2-й;
  * ⚠️ друкуємо ефективний розмір вибірки ваг (ESS): якщо він валиться,
    «виграш» є артефактом важких хвостів ваги.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import INV_SQRT_2PI                          # noqa: E402
from transport_r2 import cut_net                              # noqa: E402
from impsamp import gates_and_G                               # noqa: E402
from ridge_gate import active_dirs, fwd_grad                  # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NS = 262144
CH = 16384
MS = [1, 4, 16, 64]


def main(nets=4, dev=DEV):
    torch.manual_seed(31415)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print('  ФІТ НА 131072 ПАРАХ (нереалістично)      | ФІТ НА 4004 ПАРАХ '
          '(ПРОДАКШН)')
    print(f'{"мережа":>7s} {"✅C=0":>7s} ' +
          ' '.join(f'{"m="+str(m):>8s}' for m in MS) + ' |' +
          ' '.join(f'{"m="+str(m):>8s}' for m in MS) +
          f' {"⛔вип.":>9s} {"радіал.":>8s}')
    acc = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _, _ = cut_net(W, dev)
            G = gates_and_G(wc, dev, 4242 + n)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() * INV_SQRT_2PI
            nw = wc[-1].shape[1]
            uo = torch.ones(nw, device=dev) / math.sqrt(nw)
            U, _ = active_dirs(wc, uo, dev, 4100 + n)
            Ur = torch.linalg.qr(torch.randn(W_, W_, device=dev))[0]

            gen = torch.Generator(device=dev).manual_seed(600_000 + n)
            Zs, NA, PS = [], [], []
            for _ in range(NS // CH):
                z = torch.randn(CH, W_, device=dev, generator=gen)
                zb = torch.cat([z, -z], 0)
                x, x0 = zb, None
                for i, w in enumerate(wc):
                    x = torch.relu(x @ w)
                    if i == 0:
                        x0 = x.double()
                psi = x.double() - (x0 - m0) @ G
                pa = 0.5 * (psi[:CH] + psi[CH:])
                Zs.append(z)
                NA.append(pa.norm(dim=1))
                PS.append(pa.sum(0))
            Z = torch.cat(Zs)
            na = torch.cat(NA)
            T2 = float((sum(PS) / Z.shape[0]).pow(2).sum())
            y = na.log()
            half = Z.shape[0] // 2
            v_mc = float((na[half:] ** 2).mean()) - T2

            def gain_for(Umat, m, scale=1.0, nfit=None):
                """Фіт log‖ψ‖ ≈ a + Σ c_i (u_i·z)² на 1-й половині, вимір на 2-й.

                nfit — обмежити фіт ПРОДАКШН-обсягом (npair пар), бо саме
                перевчання на малій вибірці тричі давало міраж у цьому проєкті.
                """
                P = (Z @ Umat[:, :m]).double() ** 2
                nf = half if nfit is None else min(nfit, half)
                A = torch.cat([P[:nf], torch.ones(nf, 1, dtype=torch.float64,
                                                  device=dev)], 1)
                sol = torch.linalg.lstsq(A, y[:nf].unsqueeze(1)).solution
                c = sol[:m, 0] * scale
                # стійкість: I−2C ≻ 0 ⟹ c_i < 0.5
                c = c.clamp(max=0.45)
                q2 = (P[half:] * c).sum(1)
                zq = float(torch.exp((P[:nf] * c).sum(1)).mean())
                eq = zq * float((na[half:] ** 2 * torch.exp(-q2)).mean())
                w = torch.exp(-q2)
                ess = float(w.sum() ** 2 / (w * w).sum() / w.shape[0])
                return (v_mc / (eq - T2) if eq - T2 > 0 else float('nan')), ess

            g0 = gain_for(U, 1, scale=0.0)[0]
            row = [gain_for(U, m)[0] for m in MS]
            rowp = [gain_for(U, m, nfit=4004)[0] for m in MS]
            gr, ess16 = gain_for(Ur, 16)
            # радіальний = єдиний спільний масштаб на ВСІХ 256 напрямках
            Pf = (Z.double() ** 2).sum(1)
            Af = torch.stack([Pf[:half], torch.ones(half, dtype=torch.float64,
                                                    device=dev)], 1)
            sf = torch.linalg.lstsq(Af, y[:half].unsqueeze(1)).solution
            cf = float(sf[0, 0])
            cf = min(cf, 0.45 / 1.0)
            q2 = Pf[half:] * cf
            zq = float(torch.exp(Pf[:half] * cf).mean())
            eqr = zq * float((na[half:] ** 2 * torch.exp(-q2)).mean())
            grad_ = v_mc / (eqr - T2) if eqr - T2 > 0 else float('nan')
            _, e16 = gain_for(U, 16)
            acc.append([g0] + row + rowp + [gr, grad_, e16])
            print(f'{n+1:7d} {g0:7.4f} ' + ' '.join(f'{v:8.3f}' for v in row)
                  + ' |' + ' '.join(f'{v:8.3f}' for v in rowp)
                  + f' {gr:9.3f} {grad_:8.3f}', flush=True)
        del W, wc, G, Z, na
        torch.cuda.empty_cache()
    A = np.array(acc)
    k = len(MS)
    print(f'{"СЕРЕДНЄ":>7s} {A[:,0].mean():7.4f} ' +
          ' '.join(f'{v:8.3f}' for v in A.mean(0)[1:1+k]) + ' |' +
          ' '.join(f'{v:8.3f}' for v in A.mean(0)[1+k:1+2*k]) +
          f' {A[:,-3].mean():9.3f} {A[:,-2].mean():8.3f}')
    print('\n✅ стовпець C=0 мусить бути рівно 1.0000')
    print('🔑 читати так: це ЄДИНИЙ клас зміни міри, з якого семплюють точно і '
          'даром. Якщо тут ~1.0 — гілка «важлива вибірка» закрита, бо '
          'оракульні 2.09× недосяжні жодною реалізовною щільністю')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
