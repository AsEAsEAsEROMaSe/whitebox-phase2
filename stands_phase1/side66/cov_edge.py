"""ЧИ ВИСТАЧИТЬ МАРДЖИНАЛЬНИХ `γ`, ЩОБ ВИПРАВИТИ `C`? (С73)

ЗВІДКИ. `edgeworth.py`: поправка Еджворта до `m` дає 1.66× на шарі 2 і 1.47× на
16, але лише **1.04× на 30** ⟹ вузьке місце НЕ в `m`, а в `C` (її похибка
0.6–2.8%, на порядок більша). Тепер треба Еджворт для ДРУГОГО моменту.

🎯 ПИТАННЯ ЦЬОГО СТЕНДА — ЦІНОВЕ, І ВОНО ВИРІШУЄ ВСЕ:
поправка до `Cov(relu(p_i), relu(p_j))` у загальному випадку потребує
ЗМІШАНИХ кумулянтів `κ₃(p_i,p_i,p_j)` — а це тензор `n³`, мертвий за ціною.
Але МАРДЖИНАЛЬНІ `γ₁ᵢ`,`γ₂ᵢ` ми маємо ДЕШЕВО (одна редукція на шар).
⟹ Скільки похибки `C` пояснюється САМИМИ марджинальними?

ЯК МІРЯЄМО. Для реального шару беремо ІСТИННІ `(m_pre, cp)` і рахуємо:
  * `Cov_істина` = вибіркова коваріація `relu(p)` (велика вибірка);
  * `Cov_гаус`  = наша формула (K=2 Прайса, точна діагональ);
  * похибка `E = Cov_істина − Cov_гаус`;
  * РЕГРЕСІЯ `E` на дешеві марджинальні ознаки пари `(i,j)`.
`R²` цієї регресії — СТЕЛЯ будь-якої поправки на марджинальних `γ`.

🚨 ПЛОМБИ
  ✅ на шарі 1 `p` ІДЕАЛЬНО гаусів ⟹ `E` мусить бути на рівні шуму (перевірка,
     що ми міряємо негаусовість, а не помилку формули);
  ⛔ КОНТРОЛЬ: регресія на ВИПАДКОВІ ознаки тієї ж форми ⟹ `R²`≈0;
  🔑 фіт і вимір — на РІЗНИХ парах (розділені за індексом).
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
NP_, CH = 262_144, 16384
LAYERS = (1, 2, 4, 8, 16, 24, 29)


def moments(wc, dev, seed, upto):
    """Вибіркові `m`,`C` преактивацій І активацій шару `upto`, плюс `γ₁`,`γ₂`."""
    rng = np.random.default_rng(seed)
    n = wc[upto].shape[1]
    sp = torch.zeros(n, dtype=torch.float64, device=dev)
    gp = torch.zeros(n, n, dtype=torch.float64, device=dev)
    sa = torch.zeros(n, dtype=torch.float64, device=dev)
    ga = torch.zeros(n, n, dtype=torch.float64, device=dev)
    m3 = torch.zeros(n, dtype=torch.float64, device=dev)
    m4 = torch.zeros(n, dtype=torch.float64, device=dev)
    done = 0
    while done < NP_:
        k = min(CH, NP_ - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        a = torch.cat([z, -z], 0)
        for i in range(upto + 1):
            p = a @ wc[i]
            a = torch.relu(p)
        pd = p.double()
        ad = a.double()
        sp += pd.sum(0)
        gp += pd.T @ pd
        sa += ad.sum(0)
        ga += ad.T @ ad
        m3 += (pd ** 3).sum(0)
        m4 += (pd ** 4).sum(0)
        done += k
    N = float(2 * NP_)
    mp = sp / N
    cp = gp / N - torch.outer(mp, mp)
    ma = sa / N
    ca = ga / N - torch.outer(ma, ma)
    s = cp.diagonal().clamp_min(1e-30).sqrt()
    mu3 = m3 / N - 3 * mp * (cp.diagonal() + mp * mp) + 2 * mp ** 3
    mu4 = (m4 / N - 4 * mp * (m3 / N) + 6 * mp * mp * (gp.diagonal() / N)
           - 3 * mp ** 4)
    return mp, cp, ca, s, mu3 / s ** 3, mu4 / s ** 4 - 3.0


def gauss_cov(mp, cp, s):
    """Наша формула `Cov(relu,relu)`: K=2 Прайса + точна діагональ."""
    t = mp / s
    ph, Ph = _phi(t), _Phi(t)
    m = mp * Ph + s * ph
    C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
    dg = (mp * mp + s * s) * Ph + mp * s * ph - m * m
    return C + torch.diag(dg - C.diagonal()), t, ph, Ph


def main(nets=3, dev=DEV):
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    rows = {l: {'err': [], 'r2': [], 'r2n': [], 'g1': []} for l in LAYERS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            for l in LAYERS:
                if l >= len(wc) - 1:
                    continue
                mp, cp, ca, s, g1, g2 = moments(wc, dev, 770_000 + 7919 * n + l, l)
                cg, t, ph, Ph = gauss_cov(mp, cp, s)
                E = ca - cg
                off = ~torch.eye(len(s), dtype=torch.bool, device=dev)
                sc = torch.outer(s, s)
                y = (E / sc)[off]
                rows[l]['err'].append(float(y.abs().mean()
                                            / (ca / sc)[off].abs().mean()) * 100)
                rows[l]['g1'].append(float(g1.abs().mean()))
                # ── дешеві МАРДЖИНАЛЬНІ ознаки пари ──
                r = (cp / sc)[off]
                G1i, G1j = torch.outer(g1, torch.ones_like(g1))[off], \
                    torch.outer(torch.ones_like(g1), g1)[off]
                G2i, G2j = torch.outer(g2, torch.ones_like(g2))[off], \
                    torch.outer(torch.ones_like(g2), g2)[off]
                Ti, Tj = torch.outer(t, torch.ones_like(t))[off], \
                    torch.outer(torch.ones_like(t), t)[off]
                X = torch.stack([r, r * r, r * (G1i + G1j), r * (G2i + G2j),
                                 r * G1i * G1j, r * (Ti + Tj),
                                 r * r * (G1i + G1j), r * Ti * Tj,
                                 torch.ones_like(r)], 1)
                XR = torch.randn(X.shape, dtype=torch.float64, device=dev,
                                 generator=torch.Generator(device=dev)
                                 .manual_seed(3 + l))

                def r2(M):
                    h = M.shape[0] // 2
                    A, B = M[:h], M[h:]
                    ya, yb = y[:h], y[h:]
                    co = torch.linalg.lstsq(A, ya.unsqueeze(1)).solution
                    pr = (B @ co).squeeze(1)
                    return 1.0 - float(((yb - pr) ** 2).sum()
                                       / ((yb - yb.mean()) ** 2).sum())
                rows[l]['r2'].append(r2(X))
                rows[l]['r2n'].append(r2(XR))
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж, істина {2*NP_} семплів\n')
    print(f'{"шар":>4s} {"|скіс| p":>9s} {"похибка C, %":>13s} '
          f'{"R² марджинальних":>18s} {"⛔R² випадкових":>16s}')
    print('-' * 66)
    for l in LAYERS:
        if rows[l]['r2']:
            m = np.median
            print(f'{l:4d} {m(rows[l]["g1"]):9.4f} {m(rows[l]["err"]):12.3f}% '
                  f'{m(rows[l]["r2"]):18.4f} {m(rows[l]["r2n"]):16.4f}')
    print('-' * 66)
    print('🎯 ГЕЙТ: якщо `R²` марджинальних >0.7 — поправка `C` ДЕШЕВА і шлях '
          'відкритий.\n   Якщо <0.3 — потрібні ЗМІШАНІ кумулянти (тензор n³) '
          '⟹ мертво за ціною.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
