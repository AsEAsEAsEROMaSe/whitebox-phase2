"""БАГАТОРІВНЕВИЙ МОНТЕ-КАРЛО: чи є дешевий сурогат, що ділить ЗЛАМИ з мережею?

ЗВІДКИ (ідея юзера). Стеля 9.49× доведена для класу «точний до степеня d +
МК на решті». Вихід із класу — контрольна змінна, яка НЕ поліном: сама мережа,
обчислена дешевше. Тоді вона має злами приблизно в тих самих місцях і корелює
з `F` на ВСІХ степенях, включно з непіддатливими 10.5%. Формальна назва —
MLMC (Гілз): `E[F] = E[F̃] + E[F−F̃]`, дешевий рівень жене багато семплів,
дорогий — жменю, бо різниця має малу дисперсію.

АРИФМЕТИКА, ЯКУ МІРЯЄМО (оптимальний розподіл Гілза при фіксованій вартості C):
    Var_min = (√(V₀c₀) + √(V₁c₁))² / C,   база = Var(F)·c₁/C
    ⟹ ВИГРАШ = V / (√(V·γ) + √(V₁))²,  γ = c₀/c₁,  V₁ = Var(F − cF̃)
Тобто все залежить від ДВОХ чисел на сурогат: γ (частка флопів) і
ε = V₁/V = 1 − ρ². Для виграшу 2× треба √γ+√ε ≤ 0.707; для 56× — ≤ 0.134.

⚠️ ЧОМУ ЦЕ МОЖЕ ВМЕРТИ АПРІОРІ (і чому все одно міряємо): ваги випадкові,
у випадкових матриць спектр ПЛАСКИЙ (Марченко-Пастур) ⟹ зрізання рангу губить
енергію лінійно. Але проєкт уже тричі ховав апріорну оцінку об стенд, тож
міряємо САМЕ ТЕ ВІДНОШЕННЯ, на якому стоїть ставка.

СУРОГАТИ
  S1 ранг-r зрізання кожної `W` (SVD): γ = 2r/N — злами зсунуті, структура та сама
  S2 прорідження нейронів до m найважливіших + перемасштаб: γ = (m/N)²
  S3 префікс глибини k + ЛІНІЙНА голова, підігнана МНК: γ ≈ k/L
  S4 ✅ КОНТРОЛЬ: сама `F` як сурогат (ε=0, γ=1) ⟹ виграш мусить бути рівно 1.00×
  S5 ✅ КОНТРОЛЬ: незалежна ЧУЖА мережа (ε≈1) ⟹ виграш мусить бути ≈ нуль-ефект

Плюс тест юзера: СПЕКТР СТЕПЕНІВ різниці `F − cF̃`. Якщо різниця сидить на
низьких k — її бере наша вже наявна точність до 3-го степеня, і це не MLMC,
а ще краще: зрощення двох механізмів.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from transport_r2 import cut_net                              # noqa: E402
from degree_spectrum2 import spectrum_parity, RHOS            # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:0'
NS = 65536
CH = 8192


def flops(ws):
    return sum(2.0 * a * b for a, b in ws)


def cost_full(wc):
    return flops([(w.shape[0], w.shape[1]) for w in wc])


def fwd(wc, z):
    x = z
    for w in wc:
        x = torch.relu(x @ w)
    return x


def fwd_lr(uv, z):
    """Ранг-r: x ← relu((x U) V)."""
    x = z
    for U, V in uv:
        x = torch.relu((x @ U) @ V)
    return x


def make_lowrank(wc, r):
    uv, c = [], 0.0
    for w in wc:
        U, S, Vh = torch.linalg.svd(w.double(), full_matrices=False)
        rr = min(r, S.shape[0])
        A = (U[:, :rr] * S[:rr]).float()
        B = Vh[:rr].float()
        uv.append((A, B))
        c += 2.0 * (w.shape[0] * rr + rr * w.shape[1])
    return uv, c


def make_prune(wc, m):
    """Прорідження за важливістю ‖стовпець‖·‖наступний рядок‖ + перемасштаб."""
    keep = []
    for i in range(len(wc) - 1):
        imp = wc[i].norm(dim=0) * wc[i + 1].norm(dim=1)
        mm = min(m, imp.shape[0])
        keep.append(torch.topk(imp, mm).indices.sort().values)
    out, c = [], 0.0
    for i, w in enumerate(wc):
        ww = w
        if i > 0:
            ww = ww[keep[i - 1], :]
        if i < len(wc) - 1:
            ww = ww[:, keep[i]]
            ww = ww * (w.shape[1] / ww.shape[1]) ** 0.5
        out.append(ww.contiguous())
        c += 2.0 * ww.shape[0] * ww.shape[1]
    return out, c


def fwd_prefix(wc, z, k):
    x = z
    for w in wc[:k]:
        x = torch.relu(x @ w)
    return x


def opt_c_and_eps(F, G):
    """Оптимальний масштаб і залишкова частка дисперсії."""
    f = F - F.mean()
    g = G - G.mean()
    vg = float((g * g).mean())
    if vg <= 0:
        return 0.0, 1.0
    c = float((f * g).mean()) / vg
    return c, float(((f - c * g) ** 2).mean()) / float((f * f).mean())


def gain(gam, eps):
    """Виграш MLMC за оптимальним розподілом Гілза (V₀≈V)."""
    d = (math.sqrt(gam) + math.sqrt(max(eps, 0.0))) ** 2
    return 1.0 / max(d, 1e-12)


def main(nets=6, dev=DEV, do_spec=True):
    torch.manual_seed(20260729)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets + 1]

    def load(f):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        wc, _, _ = cut_net(W, dev)
        return wc

    alien = load(fs[-1])                       # для контролю S5
    rows = {}
    spec_acc = {}
    for n, f in enumerate(fs[:nets]):
        with torch.no_grad():
            wc = load(f)
            nw = wc[-1].shape[1]
            uo = torch.ones(nw, device=dev) / math.sqrt(nw)
            c1 = cost_full(wc)
            g = torch.Generator(device=dev).manual_seed(31337 + n)
            Z = [torch.randn(CH, W_, device=dev, generator=g)
                 for _ in range(NS // CH)]
            F = torch.cat([(fwd(wc, z) @ uo).double() for z in Z])

            cands = []
            for r in (16, 32, 64, 128):
                uv, c0 = make_lowrank(wc, r)
                G = torch.cat([(fwd_lr(uv, z)
                                @ torch.ones(uv[-1][1].shape[1], device=dev)
                                / math.sqrt(uv[-1][1].shape[1])).double()
                               for z in Z])
                cands.append((f'S1 ранг-{r}', G, c0 / c1))
                del uv
            for m in (64, 128, 192):
                pw, c0 = make_prune(wc, m)
                mw = pw[-1].shape[1]
                up = torch.ones(mw, device=dev) / math.sqrt(mw)
                G = torch.cat([(fwd(pw, z) @ up).double() for z in Z])
                cands.append((f'S2 ширина-{m}', G, c0 / c1))
                del pw
            for k in (8, 16, 24):
                H = torch.cat([fwd_prefix(wc, z, k).double() for z in Z])
                nh = H.shape[1]
                half = NS // 2
                A = torch.cat([H[:half], torch.ones(half, 1, device=dev,
                                                    dtype=torch.float64)], 1)
                sol = torch.linalg.lstsq(A, F[:half].unsqueeze(1)).solution
                Ga = torch.cat([H, torch.ones(NS, 1, device=dev,
                                              dtype=torch.float64)], 1) @ sol
                c0 = flops([(wc[i].shape[0], wc[i].shape[1])
                            for i in range(k)]) + 2.0 * nh
                cands.append((f'S3 глибина-{k}+МНК', Ga.squeeze(1), c0 / c1))
                del H, A, Ga
            cands.append(('✅ S4 сама F (ε=0)', F.clone(), 1.0))
            aw = alien[-1].shape[1]
            ua = torch.ones(aw, device=dev) / math.sqrt(aw)
            cands.append(('✅ S5 ЧУЖА мережа',
                          torch.cat([(fwd(alien, z) @ ua).double() for z in Z]),
                          cost_full(alien) / c1))

            for nm, G, gam in cands:
                cc, eps = opt_c_and_eps(F, G)
                rows.setdefault(nm, []).append((gam, eps, gain(gam, eps)))
            if do_spec and n < 2:
                for nm, G, gam in cands:
                    if not (nm.startswith('S1 ранг-128')
                            or nm.startswith('S2 ширина-192')
                            or nm.startswith('S3 глибина-24')):
                        continue
                    cc, _ = opt_c_and_eps(F, G)
                    R = F - cc * G
                    spec_acc.setdefault(nm, []).append(
                        resid_spectrum(wc, uo, Z, R, dev, cc, nm, wc))
            del wc, Z, F
            torch.cuda.empty_cache()
        print(f'  … мережа {n+1}/{nets} готова', flush=True)

    print(f'\n{"сурогат":22s} {"γ фл.":>8s} {"ε=1−ρ²":>9s} {"ρ²":>7s} '
          f'{"виграш MLMC":>12s}')
    print('  ' + '-' * 62)
    for nm, v in rows.items():
        a = np.array(v)
        print(f'  {nm:22s} {a[:,0].mean():8.4f} {a[:,1].mean():9.4f} '
              f'{1-a[:,1].mean():7.4f} {a[:,2].mean():11.2f}×')
    print('\n🔑 для виграшу 2× треба √γ+√ε ≤ 0.707;  для 56× ≤ 0.134')
    print('✅ читання контролів: S4 мусить дати рівно 1.00× (γ=1, ε=0); '
          'S5 — близько 1.00× знизу (γ≈1, ε≈1) ⟹ ≈0.25×')


def resid_spectrum(wc, uo, Z, R, dev, cc, nm, wref):
    """Заглушка-місток: спектр різниці рахує окремий прогін (див. spec_diff)."""
    return None


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         do_spec=False)
