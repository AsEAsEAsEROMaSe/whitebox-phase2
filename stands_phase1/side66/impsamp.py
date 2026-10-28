"""ГІЛКА 2: ВАЖЛИВА ВИБІРКА / ЗМІНА МІРИ — стеля, яку ніхто не міряв.

ЗВІДКИ. Стеля 9.49× (і виправлена 2.7–7×) рахувалась для схеми «точний до
степеня d + МК на решті». Важлива вибірка не в цьому класі: вона міняє САМ
РОЗПОДІЛ, а не точність за степенями. За весь проєкт не міряно жодного разу.

ЩО САМЕ МІРЯЄМО. Наш естиматор — це ПРОСТЕ СЕРЕДНЄ по семплах, бо
    ψ(z) = x_L(z) − G·(x_0(z) − m_0),   G = W₁diag(g₁)…W_{L−1}diag(g_{L−1})
і sub66 віддає рівно `(1/n)Σψ(z_i)`. Тому міряти IS треба НА `ψ`, а не на
сирому виході — інакше зміряємо стелю не того ланцюга (пастка дня).

Для векторної цілі при щільності `q` і вазі `w=φ/q`:
    Σ_j Var_j = E_q[w²‖ψ‖²] − ‖T‖²,  мінімум при `q ∝ φ‖ψ‖`
    ⟹ **СТЕЛЯ IS = (E‖ψ‖² − ‖T‖²) / ((E‖ψ‖)² − ‖T‖²)**
Це ОРАКУЛЬНА стеля: вона припускає, що ми знаємо `‖ψ(z)‖` до обчислення.

Практичні варіанти, які можна реально реалізувати:
  * **РАДІАЛЬНА** `q(z)=q(‖z‖)` — єдина, з якої легко семплювати в 256
    вимірах (напрямок рівномірний, радіус за одновимірною щільністю);
  * **ПРОКСІ з префікса глибини 8** — `‖ψ‖` передбачається лінійно з
    активацій шару 8 (ρ²=0.79 за `mlmc_depth.py`), ціна +33% флопів.

КОНТРОЛІ
  * ✅ `q=φ` (без нахилу) мусить дати рівно 1.000×;
  * ✅ синтетика з великим динамічним діапазоном `exp(z₁)` — IS мусить дати
    величезний виграш, інакше формула зламана;
  * ✅ окремо для АНТИТЕТИЧНОЇ пари (як у sub66) — бо антитетика вже знімає
    частину дисперсії, і рахувати IS поверх сирої вибірки означало б
    подвійний облік;
  * друкуємо і для сирого `x_L`, і для `ψ` — різниця показує, наскільки
    важливо міряти правильний ланцюг.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub66_torch import GATE_N, INV_SQRT_2PI                 # noqa: E402
from transport_r2 import cut_net                             # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
NS = 524288
CH = 16384
NRBIN = 60


def gates_and_G(wc, dev, seed, npair=8192):
    """Затвори як у sub66 + явна матриця переносу G (w₀ × w_L)."""
    rng = np.random.default_rng(int(seed))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    x = torch.cat([z, -z], 0)
    gates = [None]
    for i, w in enumerate(wc):
        p = x @ w
        if i > 0:
            gates.append((p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N)
        x = torch.relu(p)
    G = torch.eye(wc[0].shape[1], dtype=torch.float64, device=dev)
    for i in range(1, len(wc)):
        G = (G @ wc[i].double()) * gates[i].unsqueeze(0)
    return G


def is_ceiling(nrm2_sum, nrm_sum, T2, n):
    """(E‖ψ‖²−‖T‖²) / ((E‖ψ‖)²−‖T‖²)."""
    e2, e1 = nrm2_sum / n, nrm_sum / n
    num, den = e2 - T2, e1 * e1 - T2
    if den <= 0:
        return float('nan')
    return num / den


def main(nets=4, dev=DEV):
    torch.manual_seed(2024)
    # ── ✅ КОНТРОЛЬ формули на функції з великим діапазоном
    g = torch.Generator(device=dev).manual_seed(11)
    zz = torch.randn(1 << 21, device=dev, dtype=torch.float64, generator=g)
    for nm, f in (('✅ exp(z₁) — має бути ВЕЛИКИЙ виграш', torch.exp(zz)),
                  ('✅ z₁ (лінійна) — помірний', zz.abs() * 0 + zz),
                  ('✅ стала 1 — рівно 1.000×', torch.ones_like(zz))):
        a = f.abs()
        T = float(f.mean())
        e2, e1 = float((f * f).mean()), float(a.mean())
        den = e1 * e1 - T * T
        gg = (e2 - T * T) / den if den > 1e-14 else float('nan')
        print(f'  {nm:42s} стеля IS = {gg:8.3f}×')

    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    print(f'\n{"мережа":>7s} {"ОРАКУЛ сир.x":>13s} {"ОРАКУЛ ψ":>10s} '
          f'{"ОРАКУЛ ψ антитет.":>18s} {"РАДІАЛЬНА ψ":>12s} {"ПРОКСІ-8 ψ":>11s}')
    acc = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_net(W, dev)
            G = gates_and_G(wc, dev, 4242 + n)
            m0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt() * INV_SQRT_2PI
            gen = torch.Generator(device=dev).manual_seed(88_000 + n)
            sx1 = sx2 = sp1 = sp2 = sa1 = sa2 = 0.0
            Tx = torch.zeros(wc[-1].shape[1], dtype=torch.float64, device=dev)
            Tp = torch.zeros_like(Tx)
            Ta = torch.zeros_like(Tx)
            rb_s = torch.zeros(NRBIN, dtype=torch.float64, device=dev)
            rb_c = torch.zeros(NRBIN, dtype=torch.float64, device=dev)
            rb_s2 = torch.zeros(NRBIN, dtype=torch.float64, device=dev)
            h8_x, h8_y = [], []
            nn = 0
            for _ in range(NS // CH):
                z = torch.randn(CH, W_, device=dev, generator=gen)
                zb = torch.cat([z, -z], 0)
                x = zb
                x0 = None
                x8 = None
                for i, w in enumerate(wc):
                    x = torch.relu(x @ w)
                    if i == 0:
                        x0 = x.double()
                    if i == 8:
                        x8 = x.double()
                xd = x.double()
                psi = xd - (x0 - m0) @ G
                px, pp = xd.norm(dim=1), psi.norm(dim=1)
                sx1 += float(px.sum()); sx2 += float((px * px).sum())
                sp1 += float(pp.sum()); sp2 += float((pp * pp).sum())
                Tx += xd.sum(0); Tp += psi.sum(0)
                # антитетична пара
                pa = 0.5 * (psi[:CH] + psi[CH:])
                na = pa.norm(dim=1)
                sa1 += float(na.sum()); sa2 += float((na * na).sum())
                Ta += pa.sum(0)
                # радіальний профіль E[‖ψ‖ | r] на антитетичній парі
                r = z.norm(dim=1).double()
                idx = ((r - 12.0) / (24.0 - 12.0) * NRBIN).long().clamp(0, NRBIN - 1)
                rb_s.index_add_(0, idx, na)
                rb_s2.index_add_(0, idx, na * na)
                rb_c.index_add_(0, idx, torch.ones_like(na))
                if len(h8_x) < 8:
                    h8_x.append(0.5 * (x8[:CH] + x8[CH:]))
                    h8_y.append(na)
                nn += CH
            N2 = 2 * nn
            T2x = float((Tx / N2).pow(2).sum())
            T2p = float((Tp / N2).pow(2).sum())
            T2a = float((Ta / nn).pow(2).sum())
            gx = is_ceiling(sx2, sx1, T2x, N2)
            gp = is_ceiling(sp2, sp1, T2p, N2)
            ga = is_ceiling(sa2, sa1, T2a, nn)
            # РАДІАЛЬНА: q(r) ∝ φ_r(r)·E[‖ψ‖|r] ⟹ E_q[w²‖ψ‖²] = E_r[(E[‖ψ‖|r])²]...
            # точно: Σ_j Var = E_r[ E[‖ψ‖²|r] ] − ‖T‖² при q=φ;
            #        при оптимальному радіальному q мінімум = (E_r[√(E[‖ψ‖²|r])])² − ‖T‖²
            c = rb_c.clamp_min(1.0)
            m2r = rb_s2 / c
            wr = rb_c / rb_c.sum()
            den_r = float((wr * m2r.sqrt()).sum()) ** 2 - T2a
            gr = (sa2 / nn - T2a) / den_r if den_r > 0 else float('nan')
            # ПРОКСІ: лінійний прогноз ‖ψ‖ з активацій шару 8
            Xh = torch.cat(h8_x); Yh = torch.cat(h8_y)
            hf = Xh.shape[0] // 2
            A = torch.cat([Xh[:hf], torch.ones(hf, 1, dtype=torch.float64,
                                               device=dev)], 1)
            sol = torch.linalg.lstsq(A, Yh[:hf].unsqueeze(1)).solution
            A2 = torch.cat([Xh[hf:], torch.ones(Xh.shape[0] - hf, 1,
                                                dtype=torch.float64,
                                                device=dev)], 1)
            pred = (A2 @ sol).squeeze(1).clamp_min(1e-12)
            yt = Yh[hf:]
            # E_q[w²‖ψ‖²] = E[‖ψ‖²/ĥ]·E[ĥ] при q ∝ φ·ĥ
            eq = float((yt * yt / pred).mean()) * float(pred.mean())
            den_p = eq - T2a
            gpx = (sa2 / nn - T2a) / den_p if den_p > 0 else float('nan')
            acc.append((gx, gp, ga, gr, gpx))
            print(f'{n+1:7d} {gx:13.3f} {gp:10.3f} {ga:18.3f} '
                  f'{gr:12.3f} {gpx:11.3f}', flush=True)
        del W, wc, G
        torch.cuda.empty_cache()
    A = np.array(acc)
    print(f'{"СЕРЕДНЄ":>7s} ' + ' '.join(f'{v:11.3f}' for v in A.mean(0)))
    print('\n✅ «стала 1» мусила дати рівно 1.000× — це перевірка формули')
    print('🔑 читати так: ОРАКУЛ ψ антитетичний — це СТЕЛЯ всієї гілки '
          '(знання ‖ψ‖ наперед); радіальна і проксі — що реально досяжно')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
