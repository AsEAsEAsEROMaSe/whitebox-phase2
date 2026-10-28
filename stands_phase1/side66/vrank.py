"""ПЕРЕРАНЖУВАННЯ ЗА ОЧІКУВАНИМ ВИГРАШЕМ `v_i²σ_i²/(1+r_i²)` — пункт 1 юзера.

ЗВІДКИ. `maskgain.py` показав, що передбачена маска за `r` бере 5.6% призу.
Юзер: ранжувати треба не за `r`, а за ВНЕСКОМ нейрона у зменшення дисперсії,
    `v_i² · σ_i² / (1 + r_i²)`,
де `v_i` — адьойнт-вага переносу до виходу (ланцюг `v` її вже рахує ТОЧНО), а
`σ_i` — шум із замикання. **Невідоме в цьому виразі лише `r_i`** ⟹ якщо
ранжування переважно керує `v_i²σ_i²`, то помилки в `r̂` перестають бути
фатальними. І окремо: оракульна маска за `r` виграє ВСУПЕРЕЧ тому, що ігнорує
`v` ⟹ врахування `v` може підняти й саму СТЕЛЮ 1.79×.

🔑 `v_i` РАХУЄТЬСЯ БЕЗ ЖОДНОГО ЖЕРЕБУ. Зворотна рекурсія по замикальних воротах:
    `M_L = I`,  `M_ℓ = (W_{ℓ+1} ⊙ Φ(t_{ℓ+1})) · M_{ℓ+1}`,  `v_i = ‖M_ℓ[i,:]‖`.
Ворота `Φ(t)` — це рівно очікування того `gate`, який рахує `run_side1`.

ВАРІАНТИ (усі з ОРАКУЛЬНИМ `r` — міряємо СТЕЛЮ, не метод)
  * `рівно` — те, що в сабміті;
  * `β=1/(1+r²)` і `маска r<1` — нинішні найкращі;
  * **`маска за v²σ²β`** на квантилях — НОВИЙ критерій;
  * `маска за v²σ²` (БЕЗ `r`) — контроль: чи не тягне все саме `v`;
  * `β на масці v²σ²β` — м'яка вага поверх нового відбору;
  * ⛔НЕГ: новий критерій, ПЕРЕМІШАНИЙ у шарі.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import (cut_and_budget, closure_parts, run_side1,  # noqa: E402
                       LAM_S, _phi, _Phi)
from tilt_wide import truth_last                                  # noqa: E402
from price31 import truth_layers31, loo, three_slices             # noqa: E402

L_ = 32
DEV = 'cuda:3'
NDRAW = 96
KS = 30
QGRID = [0.7, 0.5, 0.3, 0.15]
BANDS = {'1..8': list(range(0, 8)), '24..30': list(range(23, 30)),
         '1..30': list(range(0, 30))}
LAM31 = tuple(LAM_S) + (0.0,) * (KS - len(LAM_S))


def state_and_v(wc, dev, npair):
    """Замикальний стан + АДЬОЙНТ-ваги `v_i` зворотною рекурсією. Без жеребів."""
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    gates, noises = [], []
    for i in range(len(wc)):
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
        gates.append(Ph)
        noises.append((d.clamp_min(1e-30).sqrt() / np.sqrt(2.0 * npair)))
    # зворотний хід: `M_ℓ` — карта збурення активацій шару ℓ у вихід
    nl = wc[-1].shape[1]
    M = torch.eye(nl, dtype=torch.float64, device=dev)
    v = [None] * len(wc)
    v[len(wc) - 1] = M.norm(dim=1)
    for i in range(len(wc) - 1, 0, -1):
        M = (wc[i].double() * gates[i].unsqueeze(0)) @ M
        v[i - 1] = M.norm(dim=1)
    return v, noises


def main(nets=24, dev=DEV):
    torch.manual_seed(3141)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = (['рівно', '⚠️β=1/(1+r²)', '⚠️маска r<1']
             + [f'⚠️**маска v²σ²β** q={q}' for q in QGRID]
             + [f'⚠️маска v²σ² (без r) q={q}' for q in QGRID]
             + [f'⚠️β на масці v²σ²β q={q}' for q in QGRID]
             + ['⛔НЕГ (v²σ²β перемішана)'])
    base_mse, G, B, C = [], None, None, []
    corr = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, npair = cut_and_budget(W, dev)
            s_cl, m_cl = closure_parts(wc, KS, dev)
            TL = truth_layers31(wc, dev, 810_000 + 7919 * n)
            T = truth_last(wc, dev, 510_000 + 7919 * n)
            vs, nz = state_and_v(wc, dev, npair)
            AW = []
            for i in range(KS):
                # 🚨 зсув індексів: `m_cl[i]` це шар `i+1` у `wc`
                noise = nz[i + 1]
                r = ((m_cl[i] - TL['m'][i]).abs() / noise).clamp_min(1e-12)
                beta = 1.0 / (1.0 + r * r)
                crit = (vs[i + 1] * noise) ** 2 * beta          # очікуваний внесок
                cnr = (vs[i + 1] * noise) ** 2                  # той самий БЕЗ `r`
                lc = crit.clamp_min(1e-300).log()
                lb = beta.clamp_min(1e-300).log()
                corr.append(float(torch.corrcoef(
                    torch.stack([lc, lb]))[0, 1]))
                rows = [torch.ones_like(r), beta, (r < 1.0).double()]
                for q in QGRID:
                    rows.append((crit >= torch.quantile(crit, 1.0 - q)).double())
                for q in QGRID:
                    rows.append((cnr >= torch.quantile(cnr, 1.0 - q)).double())
                for q in QGRID:
                    rows.append(beta * (crit >= torch.quantile(crit, 1.0 - q)))
                rows.append(torch.roll(
                    (crit >= torch.quantile(crit, 0.5)).double(), 7))
                AW.append(torch.stack(rows))
            nv = AW[0].shape[0]
            if G is None:
                G = [[] for _ in range(nv)]
                B = [[] for _ in range(nv)]
            gg = torch.zeros(nv, KS, KS, dtype=torch.float64, device=dev)
            bb = torch.zeros(nv, KS, dtype=torch.float64, device=dev)
            cc = 0.0
            for d in range(NDRAW):
                sd = 33_000_000 + 733 * n + d
                pr, A = run_side1(wc, npair, s_cl, m_cl, dev, sd,
                                  lam_s=LAM31, collect_anchors=True,
                                  anchor_w=AW)
                e = pr - T
                cc += float((e * e).sum())
                Am = A.view(nv, KS, -1)
                gg += Am @ Am.transpose(1, 2)
                bb += Am @ e
            for v in range(nv):
                G[v].append(gg[v]); B[v].append(bb[v])
            C.append(cc); base_mse.append(cc)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    base = np.array(base_mse)
    print(f'\n{nets} мереж × {NDRAW} жеребів, `λ_ℓ` LOO ПО МЕРЕЖАХ у КОЖНОМУ '
          f'варіанті')
    print(f'🔍 кореляція log(v²σ²β) з log(β): {np.nanmean(corr):+.3f}  '
          f'(близько 1 ⟹ критерій керує `r`; близько 0 ⟹ керує `v²σ²`)\n')
    for bn, idx in BANDS.items():
        print(f'--- шари {bn} ---')
        print(f'{"варіант":>26s} {"сума":>9s} {"МЕДІАНА":>9s} {"без топ-3":>10s}')
        for v, nm in enumerate(names):
            val = loo([g[idx][:, idx] for g in G[v]],
                      [b[idx] for b in B[v]], C, len(idx))
            s, md, w3 = three_slices(base, val)
            tag = '  ← У САБМІТІ' if nm == 'рівно' else ''
            tag = '  ⛔' if nm.startswith('⛔') else tag
            print(f'{nm:>26s} {s:8.3f}× {md:8.3f}× {w3:9.3f}×{tag}')
        print()
    print('🔑 читати так: якщо «маска v²σ²β» помітно вища за «маска r<1» — '
          'стеля 1.79× була наслідком НЕПРАВИЛЬНОГО критерію відбору, і весь '
          'баланс зусиль зміщується. Якщо «маска v²σ² (без r)» уже близька — '
          'то `r` майже не потрібне, і блокер із передбаченням знімається.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 24)
