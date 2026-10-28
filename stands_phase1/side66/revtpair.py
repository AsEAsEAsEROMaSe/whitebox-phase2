"""АНТИКОРЕЛЯЦІЯ СПРЯЖЕНОЇ МЕРЕЖІ: ЩО ЦЕ І ЧИ МОЖНА ВЗЯТИ. (С85)

## ЗВІДКИ
`side66/revclose.py` дав несподіване: похибка замикання СПРЯЖЕНОЇ мережі
(`W_31ᵀ…W_0ᵀ`) корелює з похибкою прямої на **−0.4559** (24 мережі), тоді як
просто перевернута (REV) дає +0.13, а ⛔перемішана — +0.06. Тобто це не шум і
не «будь-яка перестановка»: транспонування робить щось конкретне.
🚨 Прямо взяти НЕ МОЖНА: похибка REVT потребує істини ПЕРЕВЕРНУТОЇ мережі.

## ЩО ПЕРЕВІРЯЄМО — ТРИ ПИТАННЯ
1️⃣ **ЗАКОН ЗБЕРЕЖЕННЯ?** Якщо `err_FWD + err_REVT ≈ const`, то розкид суми
   мусить бути ПОМІТНО менший за розкид доданків. Рахується з готового
   `revclose.npz` — безкоштовно.
2️⃣ **ЧИ АНТИКОРЕЛЯЦІЯ ВИЖИВАЄ ПРИ БІЛЬШІЙ ВИБІРЦІ** — на 24 мережах |r|=0.46
   має p≈0.02, тобто на межі. Довіряти не можна, поки не перевірено ширше.
3️⃣ 🎯 **ЧИ Є СПОСТЕРЕЖНИЙ ПРОКСІ.** Замикання спряженої мережі рахується БЕЗ
   істини — це просто ще одне замикання. Питання: чи скаляри з ПАРИ
   (`m_FWD`, `m_REVT`) корелюють з оракульним масштабом `c*`?
   Ознаки (всі — з ваг, БЕЗ жеребів, тобто без шуму, що вбив sub95):
     `r1 = ‖m_REVT‖₁/‖m_FWD‖₁` · `r2 = Σlog s̄` різниця FWD−REVT
     `r3 = ⟨Φ(t)⟩` різниця · `r4 = cos(m_FWD, m_REVT)`
   ⛔КОНТРОЛІ: ті самі ознаки з REV і з ⛔SHUF замість REVT.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: закону збереження НЕМАЄ (розкид суми впаде менш ніж
   на 15%), а антикореляція — наслідок спільної залежності обох похибок від
   спектра ваг із протилежним знаком. Спостережні проксі дадуть |корел| < 0.3.
   Шанс на важіль ~12%. Головна цінність — закрити питання числом.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/revtpair.py [cuda:N] [NETS]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

# ── 1️⃣ ЗАКОН ЗБЕРЕЖЕННЯ — з ГОТОВОГО revclose.npz, без прогону ───────────
R = np.load('/mnt/win_d/llm/White-Box/arena62/logs/revclose.npz')
fw, rv, rt, sh = R['FWD'], R['REV'], R['REVT'], R['⛔SHUF']
print('1️⃣ ЗАКОН ЗБЕРЕЖЕННЯ (з готового revclose.npz, 24 мережі)')
for nm, x in (('FWD', fw), ('REVT', rt), ('FWD+REVT', fw + rt),
              ('FWD+REV ⛔', fw + rv), ('FWD+SHUF ⛔', fw + sh)):
    print(f'   {nm:>12s}  сер {x.mean():.5f}  розкид {x.std():.5f}'
          f'  ({x.std()/x.mean()*100:5.2f}%)')
ind = np.sqrt(fw.std()**2 + rt.std()**2)
print(f'   ⟹ розкид суми {(fw+rt).std():.5f} проти {ind:.5f}, якби доданки були'
      f' НЕЗАЛЕЖНІ ⟹ стиснення ×{ind/(fw+rt).std():.3f}')
print(f'   🎯 закон збереження ⟺ стиснення ≳1.3; корел FWD↔REVT = '
      f'{np.corrcoef(fw, rt)[0,1]:+.4f}\n', flush=True)

# ── 2️⃣+3️⃣ ШИРША ВИБІРКА І СПОСТЕРЕЖНІ ПРОКСІ ─────────────────────────────
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж, {DEV}\n', flush=True)


def closure_stats(W, dev):
    """m_deep + скалярні характеристики замикання. БЕЗ істини."""
    m = torch.zeros(W[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(W[0].shape[0], dtype=torch.float64, device=dev)
    lgs, opn = 0.0, 0.0
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        lgs += float(s.log().mean())
        opn += float(Ph.mean())
        m = mp * Ph + s * ph
        if i < len(W) - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            d = (mp*mp + s*s) * Ph + mp*s*ph - m*m
            C = C + torch.diag(d - C.diagonal())
    return m, lgs, opn / len(W)


N = len(FS)
KEYS = ['REVT', 'REV', '⛔SHUF']
F = {f'{k}|{q}': np.zeros(N) for k in KEYS for q in ('r1', 'r2', 'r3', 'r4')}
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        gg = torch.Generator(); gg.manual_seed(777 + n)
        perm = torch.randperm(L_, generator=gg).tolist()
        m0, l0, o0 = closure_stats(W, DEV)
        ALT = {'REVT': [w.T.contiguous() for w in W[::-1]],
               'REV': W[::-1], '⛔SHUF': [W[j] for j in perm]}
        for k in KEYS:
            mk, lk, ok = closure_stats(ALT[k], DEV)
            F[f'{k}|r1'][n] = float(mk.abs().sum() / m0.abs().sum())
            F[f'{k}|r2'][n] = l0 - lk
            F[f'{k}|r3'][n] = o0 - ok
            F[f'{k}|r4'][n] = float((m0 * mk).sum() / (m0.norm() * mk.norm()))
    del W, ALT
    torch.cuda.empty_cache()
    if (n + 1) % 8 == 0:
        print(f'  … {n+1}/{N}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/revtpair.npz', **F)
LA = np.load('/mnt/win_d/llm/White-Box/arena62/logs/layalign.npz')
cstar = (LA['PT'] / (2.0 * LA['PP']))[:N]
nfw = min(N, len(fw))
print(f'\n3️⃣ СПОСТЕРЕЖНІ ПРОКСІ ({N} мереж; корел з похибкою FWD — по перших {nfw})')
print(f'{"ознака":>16s} {"розкид,%":>9s} {"корел з c*":>11s} '
      f'{"корел з err_FWD":>16s}')
print('─' * 58)
for k in KEYS:
    for q, nm in (('r1', '‖m‖ відношення'), ('r2', 'Σlog s̄ різниця'),
                  ('r3', '⟨Φ⟩ різниця'), ('r4', 'cos(m,m_alt)')):
        v = F[f'{k}|{q}']
        rel = v.std() / abs(v.mean()) * 100 if v.mean() != 0 else float('nan')
        print(f'{k+" "+nm:>16s} {rel:9.2f} '
              f'{np.corrcoef(v, cstar)[0,1]:+11.4f} '
              f'{np.corrcoef(v[:nfw], fw[:nfw])[0,1]:+16.4f}')
    print('─' * 58)
print('\n🎯 ГЕЙТ: |корел з c*| ≥ 0.3 у REVT-ознаки І помітно вище за ⛔SHUF ⟹')
print('   спостережний проксі є, гілка жива. Інакше — антикореляція')
print('   REVT↔FWD лишається цікавою, але НЕ конвертованою.')
