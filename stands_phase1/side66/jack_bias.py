"""СКЛАДАНИЙ НІЖ / РІЧАРДСОН ПО РОЗМІРУ ВИБІРКИ — ступінь свободи, де НІХТО не стоїть.

ЗВІДКИ (ідея юзера, Квенуй-Тьюкі). «Якщо зміщення розкладається як `c/n + O(1/n²)`,
проста комбінація повної оцінки й оцінок із викинутими підвибірками знищує
провідний член БЕЗ знання `c`. Це статистичний двійник Річардсона і працює там,
де показник відомий, а константа — ні».

🔑 ЧОМУ ЦЕ НЕ ТЕ, ЩО ВЖЕ ЗАКРИТО. С68 закрила Річардсона **по ШИРИНІ** — там
блокер один: нема другої ширини для тієї самої цілі. Складаний ніж по РОЗМІРУ
ВИБІРКИ цього блокера не має: підвибірки в нас уже є.
🔑 І головне — у цій ступені свободи ПОРОЖНЬО. Антитетика вбиває НЕПАРНІ моменти,
вибілювання прибиває ДРУГИЙ, σ-поправка чинить ЗАМИКАННЯ. Скінченновибіркове
ЗМІЩЕННЯ САМОГО ВИБІЛЮВАННЯ не чіпає ніхто: `wh = cov^{−1/2}` — НЕЛІНІЙНА функція
вибіркової коваріації, тож `E[pred] ≠ істина` навіть при точному замиканні, і
похибка йде як `O(1/npair)`.

ЩО РАХУЄМО
  1. **Розклад зміщення**: `bias(npair) = E_жереби[pred] − істина` для
     `npair ∈ {n/8, n/4, n/2, n}`. Підганяємо `bias = B∞ + c/npair`.
     * `B∞` — те, що НЕ зникає з вибіркою = похибка ЗАМИКАННЯ (спрайсено
       окремо в `price31.py`: ідеальне замикання = 2.41×);
     * `c/npair` — те, що складаний ніж МОЖЕ зняти.
  2. **Прямий тест кінець-у-кінець**: `rich = pred(n) + θ·(pred(n) − pred(n/2))`,
     де половина береться з ТИХ САМИХ семплів (перша половина `z`). `θ=1` —
     чистий Річардсон; `θ` фітується LOO ПО МЕРЕЖАХ, бо Річардсон роздуває
     дисперсію, і оптимум може бути меншим за 1.
  3. **Скільки взагалі є що брати**: частка `‖bias‖²/MSE`. Якщо зміщення — 5%
     від MSE, стеля всієї ідеї 1.05×, і далі можна не читати.

КОНТРОЛІ
  * ⛔ НЕГ: різниця `d` з жеребу `k`, помилка з жеребу `k+1` — рве зв'язок
    одного жеребу, θ мусить дати ≈1.0;
  * ✅ вимкнене вибілювання: якщо джерело зміщення справді вибілювання, то без
    нього член `c/npair` має ВПАСТИ;
  * ✅ вимкнена σ-поправка: `B∞` має ЗМІНИТИСЬ (це інший канал зміщення);
  * 📊 ТРИ ЗРІЗИ: сума · МЕДІАНА · сума без топ-3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, KSIG, W_)
from tilt_wide import truth_last                                  # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
FRAC = [8, 4, 2, 1]                 # npair // f
RIDGE = 1e-10


def three_slices(base, var):
    base, var = np.asarray(base), np.asarray(var)
    s = base.sum() / max(var.sum(), 1e-300)
    md = float(np.median(base / np.maximum(var, 1e-300)))
    keep = np.argsort(-base)[3:]
    w3 = base[keep].sum() / max(var[keep].sum(), 1e-300)
    return s, md, w3


def loo_theta(G, b, c, ridge=RIDGE):
    """LOO ПО МЕРЕЖАХ для СКАЛЯРНОГО θ: MSE(θ) = c + 2θb + θ²G."""
    nn = len(G)
    Gt, bt = float(sum(G)), float(sum(b))
    out = []
    for n in range(nn):
        A = Gt - G[n]
        th = -(bt - b[n]) / max(A + ridge * max(A, 1e-300), 1e-300)
        out.append(float(c[n] + 2.0 * th * b[n] + th * th * G[n]))
    return np.array(out)


def main(nets=24, dev=DEV):
    torch.manual_seed(1618)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    VAR = {'база': dict(), '⛔без вибіл.': dict(white=False),
           'σ-попр. ВИМК': dict(lam_s=None)}
    bias2 = {v: {f: [] for f in FRAC} for v in VAR}       # ‖E[pred]−істина‖²
    mse = {v: {f: [] for f in FRAC} for v in VAR}         # E‖pred−істина‖²
    base_mse, G, B, C = [], [], [], []
    Gn, Bn, Cn = [], [], []
    rich1 = []                                            # θ=1, чистий Річардсон

    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KSIG, dev)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            nl = wc[-1].shape[1]
            acc = {v: {fr: torch.zeros(nl, dtype=torch.float64, device=dev)
                       for fr in FRAC} for v in VAR}
            sq = {v: {fr: 0.0 for fr in FRAC} for v in VAR}
            g = bb = cc = 0.0
            r1 = 0.0
            gN = bN = cN = 0.0
            prevd = None

            for d in range(NDRAW):
                sd = 66_000_000 + 733 * n + d
                gg = torch.Generator(device=dev).manual_seed(int(sd))
                z = torch.randn(npair, W_, device=dev, generator=gg)
                for vn, kw in VAR.items():
                    for fr in FRAC:
                        m = npair // fr
                        pr, _ = run_side1(wc, m, s_cl, m_cl, dev, sd,
                                          z_ext=z[:m], **kw)
                        acc[vn][fr] += pr
                        sq[vn][fr] += float(((pr - T) ** 2).sum())
                        if vn == 'база' and fr == 1:
                            a = pr
                        if vn == 'база' and fr == 2:
                            h = pr
                # Річардсон: rich(θ) = a + θ(a − h); MSE = ‖e‖²+2θ⟨e,dd⟩+θ²‖dd‖²
                e, dd = a - T, a - h
                g += float((dd * dd).sum())
                bb += float((e * dd).sum())
                cc += float((e * e).sum())
                r1 += float(((e + dd) ** 2).sum())
                if prevd is not None:                    # ⛔НЕГ: dd з жеребу d−1
                    gN += float((prevd * prevd).sum())
                    bN += float((e * prevd).sum())
                    cN += float((e * e).sum())
                prevd = dd
        del W, wc
        torch.cuda.empty_cache()
        for vn in VAR:
            for fr in FRAC:
                bs = acc[vn][fr] / NDRAW - T
                raw = float((bs * bs).sum())
                ms = sq[vn][fr] / NDRAW
                # 🚨 ‖mean−T‖² САМЕ ЗМІЩЕНЕ вгору на Var/NDRAW: знімаємо підлогу
                bias2[vn][fr].append(max((raw - ms / NDRAW)
                                         / (1.0 - 1.0 / NDRAW), 0.0))
                mse[vn][fr].append(ms)
        base_mse.append(cc / NDRAW)
        G.append(g / NDRAW); B.append(bb / NDRAW); C.append(cc / NDRAW)
        nd = max(NDRAW - 1, 1)
        Gn.append(gN / nd); Bn.append(bN / nd); Cn.append(cN / nd)
        rich1.append(r1 / NDRAW)
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    print(f'\n{nets} мереж × {NDRAW} жеребів\n')
    print('РОЗКЛАД ЗМІЩЕННЯ: ‖E[pred]−істина‖² як ЧАСТКА MSE '
          '(усе, що НЕ зміщення, — дисперсія)')
    print(f'{"варіант":>16s} ' + ' '.join(f'{"n/"+str(f):>9s}' for f in FRAC))
    for vn in VAR:
        row = []
        for fr in FRAC:
            b2 = float(np.sum(bias2[vn][fr]))
            m2 = float(np.sum(mse[vn][fr]))
            row.append(f'{b2 / max(m2, 1e-300):9.4f}')
        print(f'{vn:>16s} ' + ' '.join(row))
    print('\nСАМЕ ЗМІЩЕННЯ ‖bias‖² (абсолютне) — чи спадає як 1/npair?')
    print(f'{"варіант":>16s} ' + ' '.join(f'{"n/"+str(f):>10s}' for f in FRAC)
          + '   відн. n/2→n')
    for vn in VAR:
        vals = [float(np.sum(bias2[vn][fr])) for fr in FRAC]
        rat = vals[-2] / max(vals[-1], 1e-300)
        print(f'{vn:>16s} ' + ' '.join(f'{v:10.3e}' for v in vals)
              + f'   {rat:6.2f}× (чистий 1/n дав би 4.00)')

    base = np.array(base_mse)
    print('\nПРЯМИЙ ТЕСТ: rich = pred(n) + θ·(pred(n) − pred(n/2))')
    print(f'{"варіант":>18s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
    s, md, w3 = three_slices(base, np.array(rich1))
    print(f'{"θ=1 (Річардсон)":>18s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    v = loo_theta(G, B, C)
    s, md, w3 = three_slices(base, v)
    print(f'{"θ LOO по мережах":>18s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×')
    vN = loo_theta(Gn, Bn, Cn)
    s, md, w3 = three_slices(base, vN)
    print(f'{"⛔НЕГ (зсув жереб.)":>18s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×'
          f'   ⛔ мусить бути ≈1.0')
    tg = -float(sum(B)) / max(float(sum(G)), 1e-300)
    print(f'\nсукупне θ* = {tg:+.4f}  (чистий Річардсон = +1)')
    print('\n🔑 читати так: якщо частка зміщення в MSE мала — стеля ідеї = 1+частка, '
          'і складаний ніж закрито арифметикою. Якщо ‖bias‖² НЕ спадає як 1/npair — '
          'зміщення не вибіркове, а замикальне, і ніж його не бере в принципі.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
