"""ЧИ МОЖНА ВЕСТИ `C` НИЗЬКИМ РАНГОМ — ВОРОТА ДО ШИРИНИ 1024 (С80).

## 🎯 НАВІЩО (ціль сесії: тести будуть і на 1024)
`side66/cliff.py`: на 1024 найкраще, що метод може — **РІВНО НІЧИЯ з МК**
(1.0001×), бо замикання коштує **48.95%** бюджету. Бутилкове горло знайдено
точно: `cp = wᵀ(C·w)` — ДВА матмулі `ширина³` на шар (`_closure_parts`).

🔑🔑 І ТУТ ПЕРЕВЕРТАЄТЬСЯ ДАВНІЙ ВИРОК. На 256 ми ПІД підлогою множника 0.1,
тому «економія флопів = НУЛЬ» ([[MEMORY]] §ціна78). На 1024 ми ВИЩЕ підлоги
(C/B = 0.83) ⟹ кожен зекономлений флоп іде ПРЯМО В БАЛ, лінійно.
💰 Арифметика: прибрати 0.49 з 0.83 ⟹ бал кращий у **2.1×** ⟹ на 1024
вийшли б із нічиєї 1.00× у ~**2.1×** проти МК.

## 🔑 ЧОМУ НИЗЬКИЙ РАНГ ВЗАГАЛІ МОЖЛИВИЙ (сьогоднішні виміри)
* різноманітність активацій колапсує `PR 165.6 → 2.61` (С79);
* промах МК живе у **2.9** ефективних напрямках із 256 (`hitmiss2.py`);
* **40.9%** енергії промаху — вздовж САМОЇ істини, тобто ОДИН напрямок;
* атрактор — провідний власний вектор усього переносу (`fixed.py`).
⟹ якщо `C` реально низькорангова, її можна вести за `O(k·ширина²)` замість
`O(ширина³)`, тобто дешевше у `ширина/k` разів.

## 📐 ЧОМУ СХЕМА ЗАМИКАЄТЬСЯ (важливо: клас НЕ руйнується рекурсією)
`C = D + UΛUᵀ`. Тоді:
* `cp = wᵀCw = wᵀDw + (wᵀU)Λ(wᵀU)ᵀ` — діагональ обох доданків рахується за
  `O(ширина²)` і `O(k·ширина²)`, ПОВНА матриця не потрібна;
* `cp ⊙ outer(a,a) = diag(a)·cp·diag(a)` — зберігає вигляд «діаг + ранг-k»;
* `cp ⊙ cp`: діагональ×будь-що = діагональ, а `L⊙L` для `L=UΛUᵀ` має ранг
  ≤ `k(k+1)/2` З ЯВНИМ БАЗИСОМ (Хатрі-Рао стовпців `U`) ⟹ ранг росте, але
  контрольовано, і зрізається назад до `k`.

## ЩО МІРЯЄМО (ворота, БЕЗ ефективної реалізації)
Ведемо ПОВНУ `C`, але після кожного шару ЗРІЗАЄМО її до «точна діагональ +
ранг `k` позадіагоналі». Міряємо, що це робить із КІНЦЕВИМ `m` (саме він іде
у прогноз) проти ПОВНОГО замикання. Порівнюємо з власною похибкою замикання
проти істини (~0.5%): якщо втрата від зрізу МЕНША — зріз безкоштовний.

⛔ ПЛОМБИ: `k = ширина` ⟹ похибка РІВНО 0 · ⛔`k=0` (сама діагональ) —
   відома межа, мусить бути помітно гірше.
📣 ПЕРЕДБАЧЕНО: `k ≈ 8…32` дасть похибку нижче 0.1%, тобто вп'ятеро менше за
   власну похибку замикання ⟹ ворота відчиняються. Якщо ж і при `k=64`
   похибка ≳0.5% — низький ранг не працює, і горло треба різати інакше.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/lowrank_c.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
D = 32
WIDTHS = [256, 1024]
RANKS = [0, 1, 2, 4, 8, 16, 32, 64, 128]
I2PI = 0.3989422804014327
print(__doc__.split('Запуск')[0].split('## ЩО МІРЯЄМО')[0], flush=True)


def _phi(x):
    return torch.exp(-0.5 * x * x) * I2PI


def _Phi(x):
    return 0.5 * (1.0 + torch.erf(x * 0.7071067811865476))


def closure(W, width, k=None):
    """Замикання; `k` — ранг, до якого зрізаємо позадіагональ (None = повне)."""
    m = torch.zeros(width, dtype=torch.float64, device=DEV)
    C = torch.eye(width, dtype=torch.float64, device=DEV)
    for i in range(D):
        w = W[i]
        cp = w.T @ (C @ w)
        s = torch.diagonal(cp).clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        if i == D - 1:
            break
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s,
                                                                    ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + torch.diag(d - torch.diagonal(C))
        if k is not None and k < width:          # 🔪 зріз до «діаг + ранг k»
            dg = torch.diagonal(C).clone()
            R = C - torch.diag(dg)
            if k == 0:
                C = torch.diag(dg)
            else:
                ev, V = torch.linalg.eigh(R)
                idx = ev.abs().argsort(descending=True)[:k]
                Rk = (V[:, idx] * ev[idx]) @ V[:, idx].T
                C = Rk + torch.diag(dg - torch.diagonal(Rk))
    return m


ERR = {k: [] for k in RANKS}
SEAL = []
for width in WIDTHS:
    e = {k: [] for k in RANKS}
    sl = []
    for n in range(NETS):
        rng = np.random.default_rng(70_000 + 991 * n + 7 * width)
        sc = np.sqrt(2.0 / width)
        W = [torch.tensor((rng.standard_normal((width, width)) * sc
                           ).astype(np.float32), device=DEV).double()
             for _ in range(D)]
        with torch.no_grad():
            m_full = closure(W, width)
            nf = m_full.norm().clamp_min(1e-300)
            for k in RANKS:
                e[k].append(float((closure(W, width, k) - m_full).norm() / nf))
            sl.append(float((closure(W, width, width) - m_full).norm() / nf))
        del W
        torch.cuda.empty_cache()
    print(f'\nШИРИНА {width}   ({NETS} мереж)')
    print(f'⛔ПЛОМБА k=ширина: {max(sl):.2e}  (0 = ок)')
    print(f'{"ранг k":>8s} {"відн. похибка `m`":>19s} {"ціна проти повної":>19s}')
    print('-' * 50)
    for k in RANKS:
        if k >= width:
            continue
        rel = float(np.median(e[k]))
        print(f'{k:8d} {100*rel:18.4f}% {max(k,1)/width:18.4f}×')
    print(f'📐 власна похибка замикання проти ІСТИНИ ≈ 0.5% '
          f'⟹ зріз безкоштовний, поки колонка 2 ≪ 0.5%')
print('\n📊 БАЛ 1.954e-07 — ворота; виграш на 1024 дасть повний прогін')
