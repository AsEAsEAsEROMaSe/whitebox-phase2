"""ПОЕЛЕМЕНТНА `β` ЗАМІСТЬ ОДНІЄЇ `λ` НА ШАР — пункти 1 і 2 ревізії юзера.

ЗВІДКИ. Поелементні `r` на шарах 1..8 = 0.09 / 0.56 / 1.09 ⟹ `β=1/(1+r²)` =
0.99 / 0.76 / 0.46, тобто **одна `λ` на шар мусить бути компромісом між нейронами
з `r=0.1` і `r=3`, і неминуче псує обидві групи**. Заміна не потребує ані нової
статистики, ані нового прогону, ані нових ФІТОВАНИХ параметрів: `λ_ℓ` як була
одна на шар, так і лишається; змінюється лише ФОРМА по нейронах.
І пункт 2: на шарі 30 частка нейронів із `r<1` = 18% — тобто ~46 нейронів із
добре знаним середнім там, де `ρ²=0.83`. Агрегована `β≈0.11` вбивала всіх;
поелементна дасть цим сорока шести майже повну вагу.

📐 МОДЕЛЬ ЮЗЕРА (відтворює наші числа на двох глибинах): виграш КЗ =
`1/(1 − ρ²/(1+r²))`. На шарі 30 `ρ²=0.828` уже велике — усе з'їдає `r²=8`.
⟹ мішень — крива `r(ℓ)`, і вона ВИМІРНА.

ВАРІАНТИ ВАГ (усі несуться ОДНИМ проходом, бо перенос лінійний за якорем)
  * `рівно` — `β≡1`: ТЕ, ЩО В САБМІТІ (одна `λ` на шар);
  * ⚠️`оракул` — `β=1/(1+r²)` з ІСТИННИМ зміщенням: ВЕРХНЯ МЕЖА;
  * ⚠️`маска` — `β=1[r<1]`: жорсткий відбір «добре знаних» нейронів;
  * `виведена c=…` — ДОСТАВНА: зміщення НЕ знаємо, але знаємо його ФОРМУ по
    нейронах із механізму С66 (`δE[relu]=Φ(t)δm+φ(t)δs` ⟹ форма `φ(t)·s`), а шум
    рахується із замикання (`√Var(a)/√(2·npair)`, `Var(a)=d`). Тоді
    `r̂ = c·φ(t)s/шум` і фітується ЛИШЕ `c` — один скаляр на ВСІ шари, тобто
    параметрів МЕНШЕ, ніж зараз.

КОНТРОЛІ
  * ✅`рівно` мусить відтворити нинішню поведінку (це і є база);
  * ✅`виведена c=0` ≡ `рівно` ПОБІТОВО (пломба, що свіп не зсунув базу);
  * ⛔НЕГ: `β` оракульна, але ПЕРЕМІШАНА всередині шару (`roll`) — мусить бути
    не краще за `рівно`;
  * 📊 три зрізи: сума · МЕДІАНА · без топ-3; `λ_ℓ` — LOO ПО МЕРЕЖАХ.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, W_, _phi, _Phi)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
KS = 30
CGRID = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
BANDS = {'1..8 (як у сабміті)': list(range(0, 8)),
         '24..30 (глибокий)': list(range(23, 30)),
         '1..30 (усі)': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def build_weights(wc, s_cl, m_cl, TL, npair, dev):
    """Ваги якоря для всіх варіантів. Повертає (список по шарах, назви)."""
    # 🚨 форму зміщення НЕ вгадувати: беремо кілька обчислюваних кандидатів і
    # окремо друкуємо, який із них корелює з ІСТИННИМ `r`.
    shapes = ('φ(t)·s', 's', 'Φ(t)·s', 'стала', '|m|')
    names = ['рівно'] + [f'{sh} c={c}' for sh in shapes for c in CGRID[1:]] \
        + ['⚠️оракул', '⚠️маска r<1', '⛔НЕГ (перемішана)']
    diag = {sh: [] for sh in shapes}
    out = []
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    for i in range(KS + 1):                       # повторюємо рекурсію по `t`,`d`
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if i == 0:
            continue
        noise = (d.clamp_min(1e-30).sqrt()
                 / np.sqrt(2.0 * npair)).clamp_min(1e-30)
        cand = {'φ(t)·s': ph * s, 's': s, 'Φ(t)·s': Ph * s,
                'стала': torch.ones_like(s), '|m|': m.abs()}
        bias = (m_cl[i - 1] - TL['m'][i - 1]).abs()
        ro = (bias / noise).clamp_min(1e-30)      # ⚠️ІСТИННЕ r (оракул)
        lro = ro.log()
        for sh, cv in cand.items():
            x = (cv / noise).clamp_min(1e-30).log()
            xc, yc = x - x.mean(), lro - lro.mean()
            diag[sh].append(float((xc * yc).sum()
                                  / (xc.norm() * yc.norm()).clamp_min(1e-30)))
        rows = [torch.ones_like(noise)]
        for sh in cand:
            for c in CGRID[1:]:
                rows.append(1.0 / (1.0 + (c * cand[sh] / noise) ** 2))
        rows.append(1.0 / (1.0 + ro * ro))
        rows.append((ro < 1.0).double())
        rows.append(torch.roll(1.0 / (1.0 + ro * ro), 7))
        out.append(torch.stack(rows))
    return out, names, {k: float(np.mean(v)) for k, v in diag.items()}


def main(nets=24, dev=DEV):
    torch.manual_seed(9001)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    base_mse, G, B, C = [], None, None, []
    names, DIAG = None, []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            AW, names, dg = build_weights(wc, s_cl, m_cl, TL, npair, dev)
            DIAG.append(dg)
            nv = AW[0].shape[0]
            if G is None:
                G = [[] for _ in range(nv)]
                B = [[] for _ in range(nv)]
            gg = torch.zeros(nv, KS, KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, KS, dtype=torch.float64, device=dev)
            cc = 0.0
            for d in range(NDRAW):
                sd = 12_000_000 + 733 * n + d
                pr, A = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM31, collect_anchors=True,
                                  anchor_w=AW)
                e = pr - T
                cc += float((e * e).sum())
                Am = A.view(nv, KS, -1)
                gg += Am @ Am.transpose(1, 2)
                bb += Am @ e
            for v in range(nv):
                G[v].append(gg[v])
                B[v].append(bb[v])
            C.append(cc)
            base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}  (npair={npair})', flush=True)

    base = np.array(base_mse)
    print('\n🔍 ЯКА ОБЧИСЛЮВАНА ВЕЛИЧИНА ПЕРЕДБАЧАЄ ІСТИННЕ `r`?')
    print('   (кореляція логарифмів, середнє по шарах і мережах)')
    for k in DIAG[0]:
        print(f'   {k:>8s}/шум : {np.mean([d[k] for d in DIAG]):+.3f}')
    print(f'\n{nets} мереж × {NDRAW} жеребів. `λ_ℓ` фітується LOO ПО МЕРЕЖАХ '
          f'у КОЖНОМУ варіанті\n')
    for bname, idx in BANDS.items():
        print(f'--- шари {bname} ---')
        print(f'{"варіант":>22s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            Gs = [g[np.ix_(idx, idx)] if isinstance(g, np.ndarray)
                  else g[idx][:, idx] for g in G[v]]
            Bs = [b[idx] for b in B[v]]
            val = loo(Gs, Bs, C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = ''
            if nm == 'рівно':
                tag = '  ← ЦЕ СТОЇТЬ У САБМІТІ'
            if nm.startswith('⛔'):
                tag = '  ⛔ не краще за «рівно»'
            print(f'{nm:>22s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    print('🔑 читати так: якщо «виведена» помітно б\'є «рівно» — ревізія '
          'підтверджена і це переписування кількох рядків у сабміті. Якщо лише '
          '«оракул» б\'є — форма зміщення вгадана невірно, але важіль існує. '
          'Якщо ніхто не б\'є — агрегована `λ` вже оптимальна, ревізія марна.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
