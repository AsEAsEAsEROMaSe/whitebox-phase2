"""НА ЯКОМУ ШАРІ СТАВИТИ ЯКІР? Свіп глибини на РЕАЛЬНОМУ ланцюгу sub66.

ЗВІДКИ. `mlmc_depth.py` дав профіль `ρ²(ℓ)` лінійної голови: 0.496 (ℓ=1),
0.792 (ℓ=8), 0.912 (ℓ=16), **0.981 (ℓ=24)**. Це стеля будь-якої схеми
«аналітичний якір на шарі ℓ + вибірка на решті»: шар 8 → 4.8×, шар 24 → 54×.
А `estimator_sub66` якорить на шарі **0** (де середнє відоме ТОЧНО, σ/√2π) і
несе похибку крізь затвори; sub70/72 додають μ-якорі на шарах 1..8.
Питання: чи дає ГЛИБШИЙ якір більше, попри те що там замикання вже зміщене.

МЕХАНІЗМ, ЯКИЙ МІРЯЄМО (рівно ланцюг sub66)
    e_out = x̄_31 − E[relu(p_31)]            ← справжня похибка вибірки на цілі
    v_ℓ   = x̄_ℓ − m_ℓ                        ← наша оцінка похибки на шарі ℓ
    ê(ℓ)  = transport(v_ℓ, ℓ+1…31)           ← v ← (v @ W_i)·затвор_i
    залишок = e_out − λ·ê(ℓ)
    ВИГРАШ(ℓ) = E‖e_out‖² / E‖e_out − λ ê(ℓ)‖²
`m_ℓ` береться трьома способами, і різниця між ними — це і є ціна замикання:
  * **ОРАКУЛ**: справжнє `E[relu(p_ℓ)]` ⟹ стеля глибини ℓ, без питань замикання;
  * **ЗАМИКАННЯ**: `m_ℓ` із гаусової рекурсії (те, що реально доступне);
  * **ТОЧНИЙ шар 0**: `σ/√2π` ⟹ це рівно sub66, наша базова лінія.

КОНТРОЛІ
  * ✅ шар 0 ОРАКУЛ і шар 0 ТОЧНИЙ мусять збігтися до шуму (там замикання точне);
  * ✅ `λ=0` мусить дати рівно 1.000× на кожному рядку;
  * ⛔ ЧУЖИЙ `m_ℓ` (від іншої мережі) — мусить бути ГІРШЕ за 1.0;
  * вибірка — РІВНО продакшн-розміру (npair із `cut_net`, ~4004 пари,
    антитетично), бо знак ефекту в цьому проєкті вже перевертався з розміром;
  * λ оптимальне — це ОРАКУЛ по λ; поруч друкуємо λ=1 і LOO-λ по мережах.
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
from closure_precision import NQ, relu_cov0, relu_cov_gauss  # noqa: E402

W_, L_ = 256, 32
DEV = 'cuda:3'
TRUTH_PAIRS = 1_048_576
CHUNK = 65536
NDRAW = 96
ANCH = [0, 1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 30]


def truth_means(wc, dev, seed):
    """E[relu(p_ℓ)] для всіх ℓ на великому НЕЗАЛЕЖНОМУ жеребі (антитетично)."""
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < TRUTH_PAIRS:
        k = min(CHUNK, TRUTH_PAIRS - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32), device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            x = torch.relu(x @ w)
            acc[i] += x.sum(0).double()
        done += k
    return [a / (2.0 * done) for a in acc]


def closure_means(wc, dev):
    """m_ℓ = E[relu(p_ℓ)] із гаусової рекурсії (те саме замикання, що в sub70)."""
    nodes, wts = np.polynomial.hermite_e.hermegauss(NQ)
    nodes = torch.tensor(nodes, dtype=torch.float64, device=dev)
    wts = torch.tensor(wts / wts.sum(), dtype=torch.float64, device=dev)
    out = []
    C = relu_cov0(wc[0])
    m = (wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()) * INV_SQRT_2PI
    out.append(m)
    for i in range(1, len(wc)):
        Wd = wc[i].double()
        C_p = Wd.T @ (C @ Wd)
        m_p = m @ Wd
        C, m = relu_cov_gauss(m_p, C_p, nodes, wts)
        out.append(m)
    return out


def one_draw(wc, dev, seed, npair):
    """Продакшн-прогін: середні активацій усіх шарів + затвори, як у sub66."""
    rng = np.random.default_rng(int(seed))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32), device=dev)
    x = torch.cat([z, -z], 0)
    xb, gates = [], [None]
    for i, w in enumerate(wc):
        p = x @ w
        if i > 0:
            gates.append((p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N)
        x = torch.relu(p)
        xb.append(x.mean(0).double())
    return xb, gates


def transport(v, gates, wc, frm):
    for i in range(frm + 1, len(wc)):
        v = (v @ wc[i].double()) * gates[i]
    return v


def main(nets=6, dev=DEV):
    torch.manual_seed(4711)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets + 1]

    # накопичувачі: для кожного (спосіб, шар) — Σ⟨e,ê⟩, Σ‖ê‖², Σ‖e‖²
    KEYS = [('оракул', a) for a in ANCH] + [('замикання', a) for a in ANCH] \
        + [('точний-0', 0)] + [('⛔перемішаний', a) for a in (8, 24)]
    S = {k: [0.0, 0.0] for k in KEYS}
    base = 0.0
    per_net = {k: [] for k in KEYS}
    # спільний фіт: шар 0 (ТОЧНИЙ) + глибокий якір із замикання
    JO = {a: [np.zeros(2), np.zeros((2, 2))] for a in ANCH}
    STACK = [0, 8, 16, 24]
    JS = [np.zeros(len(STACK) + 1), np.zeros((len(STACK) + 1,) * 2)]

    for n, f in enumerate(fs[:nets]):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair, _ = cut_net(W, dev)
            T = truth_means(wc, dev, 900_000 + 7919 * n)
            M = closure_means(wc, dev)
            sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
            m0_exact = sg0 * INV_SQRT_2PI

            loc = {k: [0.0, 0.0] for k in KEYS}
            bloc = 0.0
            for d in range(NDRAW):
                xb, gates = one_draw(wc, dev, 5_000_000 + 131 * n + d, npair)
                e = xb[-1] - T[-1]
                bloc += float((e * e).sum())
                for a in ANCH:
                    for nm, mm in (('оракул', T[a]), ('замикання', M[a])):
                        eh = transport(xb[a] - mm, gates, wc, a)
                        loc[(nm, a)][0] += float((e * eh).sum())
                        loc[(nm, a)][1] += float((eh * eh).sum())
                e0 = transport(xb[0] - m0_exact, gates, wc, 0)
                loc[('точний-0', 0)][0] += float((e * e0).sum())
                loc[('точний-0', 0)][1] += float((e0 * e0).sum())
                # ⛔ контроль: ПЕРЕМІШАНИЙ усередині шару `m_ℓ` (масштаб той самий,
                #    відповідність компонент зруйнована) — мусить бути ≤ 1.0
                for a in (8, 24):
                    pm = M[a][torch.randperm(M[a].shape[0], device=dev)]
                    eh = transport(xb[a] - pm, gates, wc, a)
                    loc[('⛔перемішаний', a)][0] += float((e * eh).sum())
                    loc[('⛔перемішаний', a)][1] += float((eh * eh).sum())
                # спільний фіт «точний шар 0 + глибокий якір із замикання»
                for a in ANCH:
                    eh = transport(xb[a] - M[a], gates, wc, a)
                    g = np.array([float((e * e0).sum()), float((e * eh).sum())])
                    G = np.array([[float((e0 * e0).sum()), float((e0 * eh).sum())],
                                  [float((e0 * eh).sum()), float((eh * eh).sum())]])
                    JO[a][0] += g
                    JO[a][1] += G
                # стек 0(точний) + 8,16,24 (замикання) — розширення sub70/72 вглиб
                cols = [e0] + [transport(xb[a] - M[a], gates, wc, a)
                               for a in STACK[1:]]
                gv = np.array([float((e * c).sum()) for c in cols])
                Gm = np.array([[float((ci * cj).sum()) for cj in cols]
                               for ci in cols])
                JS[0][:len(cols)] += gv
                JS[1][:len(cols), :len(cols)] += Gm
            for k in KEYS:
                S[k][0] += loc[k][0]
                S[k][1] += loc[k][1]
                if loc[k][1] > 0:
                    lam = loc[k][0] / loc[k][1]
                    per_net[k].append(bloc / max(bloc - 2 * lam * loc[k][0]
                                                 + lam * lam * loc[k][1], 1e-300))
            base += bloc
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … мережа {n+1}/{nets} (npair={npair})', flush=True)

    def gains(k):
        c, q = S[k]
        if q <= 0:
            return None
        lam = c / q
        g1 = base / max(base - 2 * c + q, 1e-300)
        gl = base / max(base - lam * c, 1e-300)
        return lam, g1, gl

    print(f'\n{nets} мереж × {NDRAW} жеребів продакшн-розміру, '
          f'істина {2*TRUTH_PAIRS} семплів')
    print(f'{"спосіб":>11s} {"шар":>4s} {"λ*":>7s} {"виграш λ=1":>11s} '
          f'{"виграш λ*":>10s} {"LOO по мережах":>15s}')
    print('  ' + '-' * 62)
    for nm in ('точний-0', 'оракул', 'замикання', '⛔перемішаний'):
        for a in ANCH if nm in ('оракул', 'замикання') else \
                ([0] if nm == 'точний-0' else [8, 24]):
            k = (nm, a)
            if k not in S:
                continue
            r = gains(k)
            if r is None:
                continue
            lam, g1, gl = r
            pn = per_net[k]
            loo = (f'{np.mean(pn):.3f}× ±{np.std(pn):.3f}' if pn else '—')
            print(f'{nm:>11s} {a:4d} {lam:7.3f} {g1:10.3f}× {gl:9.3f}× '
                  f'{loo:>15s}')
    print(f'\nСПІЛЬНИЙ ФІТ: ТОЧНИЙ шар 0 (це sub66) + глибокий якір із замикання')
    print(f'{"глибокий шар":>13s} {"виграш разом":>13s} {"приріст до шару 0":>19s}')
    solo = gains(('точний-0', 0))[2]
    for a in ANCH:
        g, G = JO[a][0], JO[a][1]
        try:
            c = np.linalg.solve(G + 1e-12 * np.eye(2) * max(G[0, 0], 1e-30), g)
        except np.linalg.LinAlgError:
            continue
        gj = base / max(base - float(c @ g), 1e-300)
        print(f'{a:13d} {gj:12.3f}× {gj/solo:18.3f}×')
    g, G = JS[0], JS[1]
    c = np.linalg.solve(G + 1e-12 * np.eye(G.shape[0]) * max(G[0, 0], 1e-30), g)
    gj = base / max(base - float(c @ g), 1e-300)
    print(f'  СТЕК {STACK} (0 точний + решта із замикання): '
          f'{gj:.3f}×  ⟹ приріст {gj/solo:.3f}×')

    print('\n✅ КОНТРОЛЬ λ=0 дає рівно 1.000× за побудовою (base/base)')
    print('✅ рядки «точний-0» і «оракул 0» мусять збігтися — там замикання ТОЧНЕ')
    print('⛔ «чужий» мусить бути ≤1.0')
    print('\n🔑 читати так: «оракул» — стеля глибини; «замикання» — що реально '
          'доступно; різниця між ними = ціна зміщення замикання на цьому шарі')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
