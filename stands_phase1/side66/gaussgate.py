"""ЧИ ПОХИБКА ЗАМИКАННЯ — ЦЕ НЕГАУСОВІСТЬ `p`? ВИРІШАЛЬНИЙ ТЕСТ. (С73)

════════ ЗВІДКИ ════════
`mehler3.py` показав: третій член Прайса змінює `C` на **0.01%** при похибці
**5.8–9.7%**, а похибку `m` — РІВНО НІ НА ЩО (1.00× на всіх кроках, і контроль
з перевернутим знаком теж). ⟹ ряд за коваріацією збігається ШВИДКО, але
збігається НЕ ДО ІСТИНИ. Ряд Прайса точний ЛИШЕ якщо `(p_i,p_j)` спільно
ГАУСОВІ; у нас `p = relu(...)·W` — сума relu'ів.

════════ ТЕСТ ════════
Беремо ІСТИННІ `(m_pre, cp)` шару і рахуємо `E[relu(p)]` та `Cov(relu(p))`
ДВІЧІ:
  A) на СПРАВЖНІХ `p` мережі (як є) — істина;
  B) на ШТУЧНИХ гаусових `p ~ N(m_pre, cp)` з ТИМИ САМИМИ моментами.
Формула замикання застосована до тих самих `(m_pre, cp)` дає передбачення `F`.

  • якщо `F ≈ B`, а `B ≠ A` ⟹ формула ТОЧНА для гауса, і вся похибка — це
    НЕГАУСОВІСТЬ ⟹ ⛔жодне уточнення ряду не допоможе, клас закрито;
  • якщо `F ≠ B` ⟹ похибка в самій формулі ⟹ ряд ще можна лагодити.

📣 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: `F` проти `B` зійдеться до **<0.02%** по `m`
(формула `mΦ+sφ` для гауса точна аналітично, лишиться тільки шум вибірки),
а `B` проти `A` дасть **≈0.15–0.25%** по `m` — тобто рівно ту локальну похибку,
що зміряв `deepsrc.py`.

🚨 ПЛОМБИ
  ✅ обидві вибірки — той самий розмір і той самий генератор ⟹ шум однаковий;
  ✅ `m_pre` і `cp` для гаусової вибірки беруться з ІСТИННИХ моментів мережі;
  ⛔ КОНТРОЛЬ: третій момент `p` (скіс) — якщо він ~0, гіпотеза негаусовості
     хибна і причину треба шукати в четвертому.
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
TR_PAIRS, CHUNK = 131_072, 16384
STEPS = (0, 1, 2, 4, 8, 16, 24, 28)


def rel(a, b):
    return 100.0 * float(((a - b) ** 2).mean().sqrt() / (b ** 2).mean().sqrt())


def main(nets=3, dev=DEV):
    torch.manual_seed(101)
    WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
    SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
    have = {int(v): j for j, v in enumerate(SD)}
    fs = [f for f in sorted(glob.glob(
        '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
        if int(np.load(f)['seed']) in have][:nets]

    acc = {k: {'fb_m': [], 'ba_m': [], 'fa_m': [], 'fb_c': [], 'ba_c': [],
               'skew': [], 'kurt': []} for k in STEPS}
    for n, f in enumerate(fs):
        sd0 = int(np.load(f)['seed'])
        W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=dev)
             for q in range(L_)]
        with torch.no_grad():
            wc, _ = cut_and_budget(W, dev)
            d = len(wc)
            # ── один прохід: збираємо СПРАВЖНІ преактивації кожного шару ──
            rng = np.random.default_rng(950_000 + 7919 * n)
            sum_p = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
                     for w in wc]
            gram_p = [torch.zeros(w.shape[1], w.shape[1], dtype=torch.float64,
                                  device=dev) for w in wc]
            sum_r = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
                     for w in wc]
            gram_r = [torch.zeros(w.shape[1], w.shape[1], dtype=torch.float64,
                                  device=dev) for w in wc]
            m3 = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
                  for w in wc]
            m4 = [torch.zeros(w.shape[1], dtype=torch.float64, device=dev)
                  for w in wc]
            done = 0
            while done < TR_PAIRS:
                k = min(CHUNK, TR_PAIRS - done)
                z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                                 device=dev)
                a = torch.cat([z, -z], 0)
                for i in range(d):
                    p = (a @ wc[i]).double()
                    sum_p[i] += p.sum(0)
                    gram_p[i] += p.T @ p
                    m3[i] += (p ** 3).sum(0)
                    m4[i] += (p ** 4).sum(0)
                    r = torch.relu(p)
                    sum_r[i] += r.sum(0)
                    gram_r[i] += r.T @ r
                    a = r.float()
                done += k
            nn = float(2 * TR_PAIRS)

            for k in STEPS:
                if k >= d:
                    continue
                mp = sum_p[k] / nn
                cp = gram_p[k] / nn - torch.outer(mp, mp)
                s = cp.diagonal().clamp_min(1e-20).sqrt()
                # A) істина мережі
                mA = sum_r[k] / nn
                cA = gram_r[k] / nn - torch.outer(mA, mA)
                # F) формула замикання на ТИХ САМИХ (mp, cp)
                t = mp / s
                ph, Ph = _phi(t), _Phi(t)
                mF = mp * Ph + s * ph
                cF = cp * torch.outer(Ph, Ph) \
                    + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
                dg = (mp * mp + s * s) * Ph + mp * s * ph - mF * mF
                cF = cF + torch.diag(dg - cF.diagonal())
                # B) ШТУЧНІ ГАУСОВІ p з тими самими (mp, cp)
                ev, U = torch.linalg.eigh(cp)
                A = U @ torch.diag(ev.clamp_min(0.0).sqrt())
                g = torch.Generator(device=dev).manual_seed(5_000 + k)
                sr = torch.zeros_like(mp)
                gr = torch.zeros_like(cp)
                dn = 0
                while dn < TR_PAIRS:
                    kk = min(CHUNK, TR_PAIRS - dn)
                    e = torch.randn(kk, len(mp), device=dev, generator=g,
                                    dtype=torch.float64)
                    q = torch.cat([e, -e], 0) @ A.T + mp
                    r = torch.relu(q)
                    sr += r.sum(0)
                    gr += r.T @ r
                    dn += kk
                mB = sr / nn
                cB = gr / nn - torch.outer(mB, mB)

                acc[k]['fb_m'].append(rel(mF, mB))
                acc[k]['ba_m'].append(rel(mB, mA))
                acc[k]['fa_m'].append(rel(mF, mA))
                acc[k]['fb_c'].append(rel(cF, cB))
                acc[k]['ba_c'].append(rel(cB, cA))
                mu3 = m3[k] / nn - 3 * mp * (cp.diagonal() + mp * mp) + 2 * mp ** 3
                mu4 = (m4[k] / nn - 4 * mp * m3[k] / nn
                       + 6 * mp * mp * (gram_p[k].diagonal() / nn)
                       - 3 * mp ** 4)
                acc[k]['skew'].append(float((mu3 / s ** 3).abs().mean()))
                acc[k]['kurt'].append(float((mu4 / s ** 4).mean()) - 3.0)
        del W, wc
        torch.cuda.empty_cache()
        print(f'  … {n+1}/{nets}', flush=True)

    print(f'\n{nets} мереж, по {2*TR_PAIRS} семплів у КОЖНІЙ з двох вибірок\n')
    print(f'{"шар":>4s} │ {"F проти B":>10s} {"(формула vs гаус)":>18s} │ '
          f'{"B проти A":>10s} {"(гаус vs мережа)":>17s} │ '
          f'{"F проти A":>10s} │ {"|скіс|":>7s} {"exc.курт":>9s}')
    print(f'{"":>4s} │ {"m, %":>10s} {"C, %":>18s} │ {"m, %":>10s} '
          f'{"C, %":>17s} │ {"m, %":>10s} │')
    print('-' * 108)
    for k in STEPS:
        a = acc[k]
        if not a['fb_m']:
            continue
        f = np.median
        print(f'{k:4d} │ {f(a["fb_m"]):10.4f} {f(a["fb_c"]):18.4f} │ '
              f'{f(a["ba_m"]):10.4f} {f(a["ba_c"]):17.4f} │ '
              f'{f(a["fa_m"]):10.4f} │ {f(a["skew"]):7.4f} {f(a["kurt"]):9.4f}')
    print('-' * 108)
    print('🔴 якщо «F проти B» ≪ «B проти A» ⟹ формула для ГАУСА точна, а вся '
          'похибка — НЕГАУСОВІСТЬ ⟹ ⛔уточнення ряду НЕ ДОПОМОЖЕ В ПРИНЦИПІ.')
    print('🟢 якщо «F проти B» співмірне з «B проти A» ⟹ формулу ще є куди '
          'лагодити.')


if __name__ == '__main__':
    main(nets=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
