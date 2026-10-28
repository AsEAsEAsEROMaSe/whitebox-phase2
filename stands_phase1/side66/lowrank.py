"""ЦІНОВИЙ ГЕЙТ: ЧИ ДОСИТЬ МАЛОГО РАНГУ `C` У РЕКУРСІЇ ЗАМИКАННЯ.

ЗВІДКИ. Замикання = `2n³` на шар, `4n³×31 = 2.1e9 = 8.2%` бюджету. Еф.ранг
`C_ℓ` падає 165.6 → 4.8 з глибиною (`arena62/crank.py`) ⟹ якщо досить рангу
`r`, рекурсія коштує `r·n²` замість `2n³` ⟹ економія `2n/r`, множник
`0.1073 → 0.1018`, тобто **1.054× ДАРМА**.

🚨 ЦЕ ДІАГНОСТИКА, НЕ РЕАЛІЗАЦІЯ. Зрізаємо ЯВНО через `eigh` (дорого) — щоб
дізнатися, чи низький ранг узагалі ПРОХОДИТЬ. Дешева форма шукається потім, і
там є пастка: член `cp∘cp` (другий Мелер) від матриці рангу `r` має ранг до
`r²`, тож наївна факторизація НЕ пройде.

🔑 ПРАВИЛЬНИЙ КРИТЕРІЙ. Поріг «2e-4», який я записав у промпті, ЗАНАДТО
СУВОРИЙ: замикання вже має ВЛАСНЕ зміщення `b = m_замик − m_іст`. Якщо зсув
від зрізання ≪ `b`, він не міняє нічого. Тому міряємо ОБИДВІ величини і
дивимось на ВІДНОШЕННЯ.

ПЕЧАТКИ: `r = n` мусить дати зсув **0.0** (біт-у-біт) · ⛔ВИПАДКОВИЙ підпростір
того ж рангу мусить зламати сильніше.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, _phi, _Phi                  # noqa: E402
from price31 import truth_layers31                                # noqa: E402

L_ = 32
DEV = 'cuda:3'
KS = 31
RANKS = [8, 16, 32, 64, 128, 192, 256]


def closure_rank(wc, ksig, dev, rank=None, rnd=False, seed=0):
    """Рекурсія замикання зі зрізанням `C` до рангу `rank` після КОЖНОГО кроку.

    rnd=True — зріз на ВИПАДКОВИЙ підпростір того ж рангу (⛔контроль).
    """
    n0 = wc[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    m_o = []
    for i in range(ksig + 1):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s,
                                                                     ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(d - C.diagonal())
        if rank is not None and rank < C.shape[0]:
            C = 0.5 * (C + C.T)
            if rnd:
                # ⛔ ВИПАДКОВИЙ підпростір того ж рангу, з тією самою нормою
                Q = torch.linalg.qr(torch.randn(C.shape[0], rank, device=dev,
                                                dtype=torch.float64,
                                                generator=g))[0]
                C = Q @ (Q.T @ C @ Q) @ Q.T
            else:
                ev, U = torch.linalg.eigh(C)
                U = U[:, -rank:]
                C = U @ torch.diag(ev[-rank:]) @ U.T
            # діагональ ВІДОМА точно — відновлюємо (вона не з `C`, а з формули)
            C = C + torch.diag(d - C.diagonal())
        if i >= 1:
            m_o.append(m)
    return m_o


def main(nets=6, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    rows = {r: [] for r in RANKS}
    rnd_rows = {r: [] for r in (32, 128)}
    bias = []
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            full = closure_rank(wc, KS, dev, rank=None)
            TA = truth_layers31(wc, dev, 810_000 + 7919 * n)
            mf = full[-1]
            # ВЛАСНЕ зміщення замикання — масштаб, з яким порівнювати
            bias.append(float((mf - TA['m'][KS - 1]).norm() / mf.norm()))
            for r in RANKS:
                mr = closure_rank(wc, KS, dev, rank=r)[-1]
                rows[r].append(float((mr - mf).norm() / mf.norm()))
            for r in rnd_rows:
                mr = closure_rank(wc, KS, dev, rank=r, rnd=True,
                                  seed=77 + n)[-1]
                rnd_rows[r].append(float((mr - mf).norm() / mf.norm()))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    bm = float(np.median(bias))
    print(f'\n{nets} мереж. ВЛАСНЕ зміщення замикання `‖b₃₁‖/‖m₃₁‖` = '
          f'{bm:.5f} — ось масштаб, з яким порівнювати.\n')
    print(f'{"ранг r":>8s} {"зсув m₃₁":>12s} {"у частках b":>13s} '
          f'{"економія 2n/r":>14s}')
    print('-' * 52)
    for r in RANKS:
        v = float(np.median(rows[r]))
        print(f'{r:8d} {v:12.6f} {v/bm:13.4f} {512.0/r:13.2f}×')
    print()
    for r in sorted(rnd_rows):
        v = float(np.median(rnd_rows[r]))
        print(f'{f"⛔ВИПАДК r={r}":>8s} {v:12.6f} {v/bm:13.4f}')
    print('\n🎯 ЧИТАТИ: потрібен найменший `r`, де «у частках b» ≲0.1 (зріз не '
          'додає до вже наявного зміщення). ✅пломба r=256 мусить дати 0.0.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 6)
