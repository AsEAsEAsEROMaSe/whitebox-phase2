"""ПОПРАВКА ЕДЖВОРТА В РЕКУРСІЇ ЗАМИКАННЯ — чи падає похибка `m₃₀`? (С73)

ЗВІДКИ → [[finding-edgeworth-correction-is-the-live-lever]]. Формула
    `E[relu(p)] = mΦ + sφ − (γ₁/6)·s·t·φ + (γ₂/24)·s·(t²−1)·φ`
пломбована на штучному `p`: дає **5–11×** точності. Аналітично `γ` НЕ дістати
(скіс родиться з крос-кумулянтів, треба тензор 256³). Лишається взяти їх
З ВИБІРКИ — вона й так проходить крізь усі шари, `p` уже в руках.

🎯 ЦЕЙ СТЕНД: вставити поправку в РЕКУРСІЮ і зміряти похибку `m` на шарі 30.

ВАРІАНТИ
  A база (γ=0) — те, що в сабміті;
  B ⚠️ОРАКУЛ: `γ` з ВЕЛИКОЇ вибірки (стеля методу);
  C `γ` з РОБОЧОЇ вибірки (12400) — те, що реально можна собі дозволити;
  D лише `γ₁` з робочої;  E лише `γ₂` з робочої;
  ⛔F `γ` з ЧУЖОЇ мережі — мусить бути ГІРШЕ за базу.

🚨 ПЛОМБИ
  ✅ база мусить відтворити відому криву (≈0.27% шар 1 → ≈1.0% шар 30);
  ⛔ контроль з чужими `γ` — якщо він НЕ гірший, стенд не розсуджує;
  🔑 `γ` робочої вибірки НЕЗАЛЕЖНІ від істини (різні сіди).
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, W_, _phi, _Phi          # noqa: E402

L_ = 32
DEV = sys.argv[2] if len(sys.argv) > 2 else 'cuda:4'
SHOW = (1, 2, 4, 8, 16, 24, 30)


def sample_gammas(wc, dev, seed, npair):
    """`γ₁`,`γ₂` преактивацій кожного шару з вибірки заданого розміру."""
    g = torch.Generator(device=dev).manual_seed(seed)
    z = torch.randn(npair, W_, device=dev, generator=g)
    a = torch.cat([z, -z], 0).double()
    g1, g2 = [], []
    for w in wc:
        p = a @ w.double()
        m = p.mean(0)
        d = p - m
        s2 = (d * d).mean(0).clamp_min(1e-30)
        s = s2.sqrt()
        g1.append((d ** 3).mean(0) / s ** 3)
        g2.append((d ** 4).mean(0) / s2 ** 2 - 3.0)
        a = torch.relu(p)
    return g1, g2


def truth_m(wc, dev, seed, npair=131_072, chunk=16384):
    acc = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev) for w in wc]
    rng = np.random.default_rng(seed)
    done = 0
    while done < npair:
        k = min(chunk, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            a = torch.relu(a @ w)
            acc[i] += a.double().sum(0)
        done += k
    return [t / float(2 * npair) for t in acc]


def closure(wc, g1=None, g2=None):
    """Рекурсія із ПОПРАВКОЮ ЕДЖВОРТА в `m` (список `m` по шарах)."""
    dev = wc[0].device
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    out = []
    for i, wi in enumerate(wc):
        w = wi.double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if g1 is not None:
            m = m - (g1[i] / 6.0) * s * t * ph
        if g2 is not None:
            m = m + (g2[i] / 24.0) * s * (t * t - 1.0) * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        dg = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        C = C + torch.diag(dg - C.diagonal())
        out.append(m.clone())
    return out


def main(nets=4, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    names = ['A база (γ=0)', 'B ⚠️ОРАКУЛ γ (250k)', 'C γ з робочої (12.4k)',
             'D лише γ₁ (робоча)', 'E лише γ₂ (робоча)', '⛔F γ чужої мережі']
    acc = {k: {l: [] for l in SHOW} for k in names}
    prev = None
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            mt = truth_m(wc, dev, 990_000 + 7919 * n)
            go1, go2 = sample_gammas(wc, dev, 4000 + n, 125_000)   # оракул
            gw1, gw2 = sample_gammas(wc, dev, 8000 + n, 6_200)     # робоча
            runs = {names[0]: closure(wc),
                    names[1]: closure(wc, go1, go2),
                    names[2]: closure(wc, gw1, gw2),
                    names[3]: closure(wc, gw1, None),
                    names[4]: closure(wc, None, gw2)}
            if prev is not None and len(prev[0]) == len(gw1):
                try:
                    runs[names[5]] = closure(wc, prev[0], prev[1])
                except RuntimeError:
                    pass
            for k, r in runs.items():
                for l in SHOW:
                    if l - 1 < len(r):
                        e = 100.0 * float(((r[l - 1] - mt[l - 1]) ** 2).mean()
                                          .sqrt() / (mt[l - 1] ** 2).mean().sqrt())
                        acc[k][l].append(e)
            prev = (gw1, gw2)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж; похибка `m` у % (медіана)\n')
    hdr = f'{"шар":>4s}' + ''.join(f'{k:>22s}' for k in names)
    print(hdr)
    print('-' * len(hdr))
    for l in SHOW:
        row = f'{l:4d}'
        for k in names:
            v = acc[k][l]
            row += f'{np.median(v):21.4f}%' if v else f'{"—":>22s}'
        print(row)
    print('-' * len(hdr))
    b = np.median(acc[names[0]][30]) if acc[names[0]][30] else 0
    for k in names[1:]:
        v = acc[k][30]
        if v:
            print(f'  шар 30: {k:24s} {b/np.median(v):6.3f}× проти бази')
    print('\n🎯 ГЕЙТ: `C` мусить дати ≥2× на шарі 30, щоб мати сенс; '
          '⛔`F` мусить бути <1×.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 4)
