"""ТОЧНА ФОРМУЛА РОЗГОРТКИ І ПРОФІЛЬ ВНЕСКУ ПО ГЛИБИНІ. (С85, вивід з нуля)

## 📐 ТОЧНА ТОТОЖНІСТЬ (виводиться за два рядки, без наближень)
`ReLU(x) = (x + |x|)/2` ⟹ `E[a_ℓ] = ½·W_ℓᵀE[a_{ℓ−1}] + ½·E[|h_ℓ|]`.
Перший доданок — ТОЧНО лінійний. Розгортаємо рекурсію до кінця:
    **`E[a_L] = Σ_{ℓ=1..L} 2^{−(L−ℓ+1)} · (W_LᵀW_{L−1}ᵀ…W_{ℓ+1}ᵀ) · E[|h_ℓ|]`**
(член з `E[a₀]=E[z]=0` зникає).
🔑 **ВСЯ ЗАДАЧА — ЦЕ ЗВАЖЕНА СУМА `E[|h_ℓ|]` ПО ШАРАХ.** Уся негаусовість, уся
похибка замикання сидить рівно в цих `L` векторах, і більше ніде.

## 🎯 ЩО КАЖЕ ВАГА
He-init тримає `‖W‖ ≈ √2` (саме щоб норма зберігалась через ReLU) ⟹ множник
на шар `≈ √2/2 = 0.707` ⟹ **прямий внесок шару спадає як `0.707^(L−ℓ)`**:
90% значення має приходити з ОСТАННІХ ~7 шарів, перші 20 — під 0.1%.
🚨 А наші λ-канали стоять на шарах **1..8**, тобто на ПЕРШИХ. Якщо профіль
підтвердиться, це прямо каже, куди їх переносити.
📌 узгоджується з тим, що sub89 («канал `q31` + замикання на 31») дав MSE
**−3.0%** — тобто канал на ОСТАННЬОМУ шарі працює, і помер лише ціною.

## ЩО МІРЯЄМО (48 мереж, лише ваги + замикання — БЕЗ вибірки, дешево)
1. `E[|h_ℓ|]` із замикання (гаусове наближення: `σ[2φ(t)+t(2Φ(t)−1)]`);
2. **прямий внесок** шару `ℓ` у `E[a_L]`: норма `2^{−(L−ℓ+1)}·(ΠWᵀ)·E[|h_ℓ|]`,
   рахується зворотною рекурсією `u_ℓ = ½·W_{ℓ+1}ᵀ u_{ℓ+1}`, `u_L = ½·I`;
3. ⛔ПЛОМБА: **сума внесків мусить дорівнювати `m_deep`** із замикання —
   якщо тотожність виведена правильно, розбіжність має бути ~1e-12.

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: профіль спадає геометрично з показником ≈0.7 на
   шар; половина значення — з останніх 2–3 шарів, 90% — з останніх 7.
   Якщо так, гіпотеза «переносити канали на хвіст» стає головною на С86.
   Ризик: `E[|h_ℓ|]` не незалежні між шарами (масштаб `σ_ℓ` несе всю історію),
   тож малий ПРЯМИЙ внесок ≠ малий вплив. Профіль каже про ваги, не про причини.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/depthprof.py [cuda:N] [NETS]
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
L_ = 32
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж, {DEV}\n', flush=True)


def parts(W, dev):
    """`E[|h_ℓ|]` для кожного шару + `m_deep` — усе з замикання."""
    n0 = W[0].shape[0]
    m = torch.zeros(n0, dtype=torch.float64, device=dev)
    C = torch.eye(n0, dtype=torch.float64, device=dev)
    AB = []
    for i in range(len(W)):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        # E[|h|] = σ[2φ(t) + t(2Φ(t)−1)]
        AB.append(s * (2.0 * ph + t * (2.0 * Ph - 1.0)))
        m = mp * Ph + s * ph
        if i < len(W) - 1:
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            d = (mp*mp + s*s) * Ph + mp*s*ph - m*m
            C = C + torch.diag(d - C.diagonal())
    return AB, m


CONTR = np.zeros((len(FS), L_))
SEAL = np.zeros(len(FS))
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        AB, mdeep = parts(W, DEV)
        # зворотна рекурсія: внесок шару ℓ є ½·(W_L…W_{ℓ+1})ᵀ · E[|h_ℓ|]
        acc = torch.zeros_like(mdeep)
        U = torch.eye(W[-1].shape[1], dtype=torch.float64, device=DEV)
        for i in range(L_ - 1, -1, -1):
            c = 0.5 * (AB[i] @ U)
            CONTR[n, i] = float(c.abs().sum())
            acc = acc + c
            if i > 0:
                U = 0.5 * (W[i].double() @ U)
        SEAL[n] = float((acc - mdeep).abs().max() / mdeep.abs().max())
    del W
    torch.cuda.empty_cache()
    if (n + 1) % 12 == 0:
        print(f'  … {n+1}/{len(FS)}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/depthprof.npz',
         CONTR=CONTR, SEAL=SEAL)
print(f'\n⛔ПЛОМБА тотожності: max |Σвнесків − m_deep| / |m_deep| = '
      f'{SEAL.max():.3e}  (мусить бути ~1e-12)')
P = CONTR.mean(0)
tot = P.sum()
print(f'\n{"шар":>5s} {"внесок":>12s} {"частка":>9s} {"хвіст ≥":>9s} '
      f'{"відн. до попер.":>16s}')
print('─' * 58)
cum = 0.0
for i in range(L_):
    cum += P[i]
    if i >= L_ - 12 or i % 6 == 0:
        r = P[i] / P[i-1] if i > 0 and P[i-1] > 0 else float('nan')
        print(f'{i+1:5d} {P[i]:12.5e} {P[i]/tot*100:8.3f}% '
              f'{(tot-cum+P[i])/tot*100:8.3f}% {r:16.4f}')
print('─' * 58)
for k in (1, 2, 3, 5, 7, 10, 15):
    print(f'   останні {k:2d} шарів дають {P[-k:].sum()/tot*100:6.2f}% значення')
rr = np.exp(np.diff(np.log(P[8:])).mean())
print(f'\n📐 середній множник на шар у хвості: {rr:.4f} '
      f'(теорія √2/2 = 0.7071)')
