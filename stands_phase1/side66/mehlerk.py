"""ВЗАЄМОДІЯ ЙМОВІРНОСТЕЙ = ВИЩІ ЧЛЕНИ РЯДУ МЕЛЕРА В `C`. (С85, ідея ЮЗЕРА)

## ЗВІДКИ — ФОРМУЛЮВАННЯ ЮЗЕРА
«Нам треба певно подоба хвильової кореляції ймовірностей або їх взаємодії.»
🎯 Це називає РЕАЛЬНУ дірку. У нашому `C` перший член — **добуток
НЕЗАЛЕЖНИХ ймовірностей відкриття** `Φ(t_i)·Φ(t_j)`, тоді як нейрони
відкриваються КОРЕЛЬОВАНО (через `cp_ij`). Спільна ймовірність ≠ добуток, і
поправка на це — саме вищі члени ряду Мелера.

## 📐 РЯД МЕЛЕРА ДЛЯ `E[a_i a_j]`, ВИВЕДЕНИЙ ТУТ ЖЕ
`h_i = mp_i + s_i·X`, `a_i = s_i·ReLU(X + t_i)`, `t_i = mp_i/s_i`, `ρ = cp/(s_i s_j)`.
`E[f(X)g(Y)] = Σ_k (ρ^k/k!)·A_k^i·A_k^j`, `A_k = E[ReLU(X+t)·He_k(X)]`:
  `A_0 = tΦ+φ` · `A_1 = Φ` · `A_2 = φ` · **`A_3 = −tφ`** · **`A_4 = (t²−1)φ`**
⟹ `E[a_i a_j] = m_i m_j + cp·ΦΦ + (cp²/2)·φφ/(s s)`
              **+ (cp³/6)·(tφ)(tφ)/(s² s²) + (cp⁴/24)·((t²−1)φ)((t²−1)φ)/(s³ s³)**
✅ перші три члени ТОЧНО збігаються з тим, що в сабміті ⟹ виведення звірено.
🔑 `k=3,4` — це **два `outer` по 256²** = 131k флопів на шар проти 16.7M у
матмулі ⟹ **фактично безкоштовно у флопах** (+~6 викликів/шар у residual).
🚨 ВАЖЛИВО: ряд Мелера ТОЧНИЙ для двовимірного ГАУСА. Тобто `k=3,4` усувають
НЕ негаусовість, а **чисту похибку обрізання** — це інший клас, ніж Еджворт.

## ⛔ ЩО ПРО ЦЕ ВЖЕ ЗАПИСАНО (і що мусимо відтворити ПРИЛАДОМ)
[[finding-local-closure-formula-ceiling-is-1.111x]]: «⛔Прайс K=3 → **1.003×**»,
і стеля ВСЬОГО класу «краща локальна формула» = **1.111×** (взято 1.013×), бо
**2/3 похибки робить ПЕРЕНОС, а не крок**. Теорема Прайса `∂E/∂ρ = E[f'g']`
дає рівно розклад по `ρ` ⟹ «Прайс K=3» ≡ «Мелер K=3».
🚨 АЛЕ те міряли на `m₃₀` і на 6 мережах, і **K=4 не міряли зовсім**. Правило
«число з пам'яті, на яке спираєш дію, відтворити приладом у цій сесії» ⟹
міряємо самі, і одразу в БАЛІ, а не лише в похибці `m`.

## ЩО МІРЯЄМО (`m_deep` проти БАНКОВОЇ істини, 24 мережі)
K=2 (як у сабміті) · K=3 · K=4 · ⛔K=4 зі ЗЛАМАНИМ знаком `k=3` (контроль:
чи виграш від ФОРМУЛИ, чи від будь-якого збурення тієї ж норми)
⛔ПЛОМБА: K=2 мусить дати рівно ту похибку, що `revclose.py` (0.0098).

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: K=3 дасть ~1.00–1.01× (як записано), K=4 —
   не більше. Причина: `ρ = cp/(s_i s_j)` між нейронами ОДНОГО шару мала
   (ваги незалежні ⟹ `|ρ| ~ 1/√n`), а члени йдуть як `ρ^k/k!` ⟹ `k=3` менший
   за `k=2` у ~`ρ/3 ≈ 0.02` разів. Шанс на важіль ~10%. Але вимір дешевий,
   і він ЗАКРИВАЄ клас числом цієї сесії, а не посиланням.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/mehlerk.py [cuda:N] [NETS]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import _phi, _Phi                                  # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
L_, W_ = 32, 256
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

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


def closure_k(W, dev, K=2, sign3=1.0):
    """Замикання з рядом Мелера до порядку K (K=2 — як у сабміті)."""
    m = torch.zeros(W[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(W[0].shape[0], dtype=torch.float64, device=dev)
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if i < len(W) - 1:
            cp2 = cp * cp
            u2 = ph / s
            Cn = cp * torch.outer(Ph, Ph) + 0.5 * cp2 * torch.outer(u2, u2)
            if K >= 3:                       # 🎯 A_3 = −tφ ⟹ добуток ДОДАТНИЙ
                u3 = (t * ph) / (s * s)
                Cn = Cn + (sign3 / 6.0) * (cp2 * cp) * torch.outer(u3, u3)
            if K >= 4:                       # 🎯 A_4 = (t²−1)φ
                u4 = ((t * t - 1.0) * ph) / (s * s * s)
                Cn = Cn + (1.0 / 24.0) * (cp2 * cp2) * torch.outer(u4, u4)
            d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
            C = Cn + torch.diag(d - Cn.diagonal())
    return m


VAR = [('K=2 (сабміт)', 2, 1.0), ('K=3', 3, 1.0), ('K=4', 4, 1.0),
       ('⛔K=4 зі зламаним знаком k=3', 4, -1.0)]
ERR = {v[0]: np.zeros(len(FS)) for v in VAR}

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for nm, K, sg in VAR:
            md = closure_k(W, DEV, K, sg)
            # ✅КРОС-ЦІЛЬ: шум істини має нульове середнє в добутку
            ERR[nm][n] = float(((md - T1) * (md - T2)).mean())
        print(f'  … {n+1}/{len(FS)} seed {sd0} · ' +
              ' '.join(f'{nm.split()[0]}={ERR[nm][n]:.3e}' for nm, _, _ in VAR),
              flush=True)
    del W
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/mehlerk.npz', **ERR)
base = ERR['K=2 (сабміт)'].mean()
print(f'\n⛔ПЛОМБА: K=2 — це рівно замикання сабміта')
print(f'{"варіант":>32s} {"MSE(крос)":>13s} {"×":>9s} {"кращих":>9s}')
print('─' * 68)
for nm, _, _ in VAR:
    g = base / ERR[nm].mean()
    better = int((ERR[nm] < ERR['K=2 (сабміт)']).sum())
    print(f'{nm:>32s} {ERR[nm].mean():13.6e} {g:9.5f} {better:6d}/{len(FS)}')
print('─' * 68)
print('\n🎯 ГЕЙТ: K=4 має дати ≥1.01× І бути кращим на ≥18/24 мережах,')
print('   ⛔зламаний знак — помітно ГІРШЕ за K=4, інакше виграш не від формули.')
