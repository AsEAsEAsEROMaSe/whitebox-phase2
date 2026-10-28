"""ЗВІДКИ ВЗЯТИ `γ₁`: ДЕШЕВІ ДЖЕРЕЛА ТРЕТЬОГО МОМЕНТУ (С81)

## ЗВІДКИ ПИТАННЯ
`side66/qedge.py` щойно показав: Еджворт-3 з ІСТИННИМ `γ₁` дає канал воріт
**×1.1296** (бал 1.954e-07 → **1.7297e-07**). Уся гілка тепер тримається на
одному: `γ₁` у сабміті НЕМА, а аналітичний шлях `μ₃ ≈ ΣW³μ₃` уже зміряний і
промахується у **26×** ([[finding-edgeworth-correction-is-the-live-lever]]).

## 🔑 ЩО МІРЯЄМО (дешевий етап: лише ПОХИБКА `q`, без важкої регресії)
Похибка `q` — не повна характеристика (`qedge.py` довів, що ФОРМА залишку
важить не менше за розмір), але вона відсіює завідомо мертві джерела за
чотири хвилини замість сорока п'яти.

Еталон завжди будуємо реалістично — з `m, s` ІЗ ЗАМИКАННЯ (як у сабміті),
міняємо лише джерело `γ₁`:
  ⛔ 0 · `γ₁ = 0`                     — це рівно нинішній вільний `Φ`
  🎯 1 · `γ₁` ІСТИННИЙ                — стеля
  🟢 2 · `γ₁` з ПІЛОТА (256 семплів)  — БЕЗКОШТОВНО, семпли вже є й викидаються
  🟢 3 · `γ₁` з 4096 семплів          — скільки дає вдесятеро більша вибірка
  🟢 4 · `γ₁` = СЕРЕДНЄ ПО ШАРУ (істинне, один скаляр на шар) — чи потрібна
        взагалі понейронна деталізація, чи вистачить одного числа
  🟢 5 · `γ₁` істинний, УСАДЖЕНИЙ до середнього по шару (0.5/0.5)

⛔ ПЛОМБИ:
 * варіант 0 мусить дати **0.887%** (`qsplit.py`, ті самі мережі).
   🚨 ПЕРША ВЕРСІЯ ЦЬОГО СТЕНДА ДАЛА 33.67% — я взяв `t = m_cl/s_cl`, а `m_cl`
   у замиканні це середнє ПІСЛЯ ReLU, не `mp = m@w`. Пломба спіймала одразу;
 * варіант 1 мусить дати ≈**0.42%** — трохи гірше за 0.385% із `qsplit.py`,
   бо там були ІСТИННІ `m,s`, а тут замикання;
 * ✅внутрішня: `γ₁` з 4096 семплів мусить бути точнішим за `γ₁` з 256.

📣 ПЕРЕДБАЧЕНО: пілот (256 семплів) дасть шум `γ₁` близько `√(6/256) ≈ 15%`
   ⟹ похибка `q` осяде близько 0.5…0.6%, тобто ПІВДОРОГИ. Найцікавіший
   варіант — 4 (одне число на шар): якщо асиметрія в шарі однорідна, він дасть
   майже стільки ж, скільки істинний понейронний, і тоді гілка стає ДЕШЕВОЮ.
   Якщо ж варіант 4 дасть ≈0.9% (як `γ₁=0`), асиметрія понейронна, і без
   точного третього моменту гілка мертва.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qgam.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/arena62')
sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from base_real import cut_and_budget, closure_parts            # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, W_, KS = 32, 256, 31
NPAIR, CH = 524_288, 32768
I2PI = 0.3989422804014327
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _phi(x):
    return torch.exp(-0.5 * x * x) * I2PI


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def moments(wc, dev, seed, npair):
    """`q`, `m`, `s`, `γ₁` перед-активацій шарів 1..KS за `2·npair` семплів."""
    ns = [w.shape[1] for w in wc]
    acc = [[torch.zeros(n, dtype=torch.float64, device=dev) for n in ns]
           for _ in range(4)]
    rng = np.random.default_rng(int(seed))
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.tensor(rng.standard_normal((k, W_)).astype(np.float32),
                         device=dev)
        x = torch.cat([z, -z], 0)
        for i, w in enumerate(wc):
            p = (x @ w).double()
            acc[0][i] += (p > 0).double().sum(0)
            p2 = p * p
            acc[1][i] += p.sum(0)
            acc[2][i] += p2.sum(0)
            acc[3][i] += (p2 * p).sum(0)
            x = torch.relu(p).float()
        done += k
    n = 2.0 * done
    out = []
    for i in range(1, KS + 1):
        m1, m2, m3 = acc[1][i] / n, acc[2][i] / n, acc[3][i] / n
        s = (m2 - m1 * m1).clamp_min(1e-300).sqrt()
        c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
        out.append((acc[0][i] / n, m1, s, c3 / s ** 3))
    return out


def rel(pred, tru):
    num = sum(float(((pred[i] - tru[i]) ** 2).sum()) for i in range(KS))
    den = sum(float((tru[i] ** 2).sum()) for i in range(KS))
    return np.sqrt(num / max(den, 1e-300))


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]

NM = ['⛔ 0 · γ₁ = 0 (нинішній вільний Φ)',
      '🎯 1 · γ₁ ІСТИННИЙ',
      '🟢 2 · γ₁ з ПІЛОТА (256 семплів)',
      '🟢 3 · γ₁ з 4096 семплів',
      '🟢 4 · γ₁ = СЕРЕДНЄ ПО ШАРУ',
      '🟢 5 · γ₁ усаджений 0.5 до середнього']
E = [[] for _ in NM]
GERR = [[] for _ in range(4)]           # похибка самого `γ₁` у варіантах 2..5
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, _ = cut_and_budget(W, DEV)
        s_cl, m_cl, f_cl, _ = closure_parts(wc, KS, DEV, extras=True)
        MT = moments(wc, DEV, 830_000 + 7919 * n, NPAIR)
        q_t = [MT[i][0] for i in range(KS)]
        g_t = [MT[i][3] for i in range(KS)]
        # 🔑 РЕАЛІСТИЧНИЙ аргумент. 🚨 `m_cl` — це середнє ПІСЛЯ ReLU, а не
        #    `mp = m@w`, тож `m_cl/s_cl` НЕ є `t` (перша спроба дала 33.67%
        #    замість 0.887% — пломба спіймала). Правильно: `f_cl = Φ(t)` уже
        #    Є в замиканні, тому `t` дістаємо ТОЧНИМ оберненням.
        t = [1.4142135623730951 * torch.erfinv(
            2.0 * f_cl[i].clamp(1e-12, 1.0 - 1e-12) - 1.0) for i in range(KS)]
        MP = moments(wc, DEV, 44_000_000 + 7919 * n, 128)      # пілот
        M4 = moments(wc, DEV, 55_000_000 + 7919 * n, 2048)     # 4096 семплів
        g_p = [MP[i][3] for i in range(KS)]
        g_4 = [M4[i][3] for i in range(KS)]
        g_m = [torch.full_like(g_t[i], float(g_t[i].mean()))
               for i in range(KS)]
        g_h = [0.5 * g_t[i] + 0.5 * g_m[i] for i in range(KS)]
        for v, g in enumerate((g_p, g_4, g_m, g_h)):
            GERR[v].append(rel(g, g_t))
        for v, g in enumerate((None, g_t, g_p, g_4, g_m, g_h)):
            if g is None:
                P = [_Phi(t[i]) for i in range(KS)]
            else:
                P = [_Phi(t[i]) + _phi(t[i]) * g[i] * (t[i] ** 2 - 1.0) / 6.0
                     for i in range(KS)]
            E[v].append(rel(P, q_t))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж · істина {2*NPAIR} семплів · аргумент `t` ІЗ ЗАМИКАННЯ')
print(f'{"варіант":38s} {"похибка γ₁":>11s} {"похибка q":>11s} {"проти γ₁=0":>11s}')
print('-' * 76)
a0 = float(np.mean(E[0]))
for v, nm in enumerate(NM):
    m = float(np.mean(E[v]))
    ge = (f'{100*float(np.mean(GERR[v-2])):10.2f}%' if v >= 2 else ' ' * 11)
    print(f'{nm:38s} {ge} {100*m:10.4f}% {a0/m:10.2f}×')
print('\n🎯 ПОРОГИ: 0.887% → ×0.9896 · 0.385% → **×1.1296** · 0.222% → ×1.1114')
print('   ⛔рівень контролю ×1.0165 ⟹ нижче нього джерело не годиться')
print('📊 БАЛ 1.954e-07 · зі стелею гілки 1.7297e-07')
