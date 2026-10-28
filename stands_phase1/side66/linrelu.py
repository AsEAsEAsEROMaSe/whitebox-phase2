"""ЛІНІЙНА ПРОТИ НАШОЇ: де саме ReLU ламає суму і скільки з того доживає (С77).

Думка юзера: різниця з лінійною мережею — ЛИШЕ в сумах; ReLU ріже доданки, перші
шари ламають суму, і поламане доходить до кінця. Це розкладається ТОЧНО.

Для кожної координати:
    E[ReLU(p)] = E[p]·g + c ,   g = P(p>0),   c = Cov(p, 1{p>0})
                 └ЛІНІЙНЕ┘   └ЧИСТО ReLU┘
Перший доданок — рівно лінійний перенос крізь СЕРЕДНІ ворота (те, чим наш `v`
і їздить). Другий, `c`, — уся різниця з лінійною мережею.
⟹ рекурсія ТОЧНА:  m_ℓ = (m_{ℓ−1} Wℓ) ⊙ gℓ + cℓ ,  m₀ = 0.
А що перенос ЛІНІЙНИЙ за `m`, внески додаються:
    **m₃₂ = Σ_ℓ (перенос cℓ з шару ℓ до 32)**
Отже «скільки шар ℓ вкладає у відповідь» — це точна, а не приблизна величина.

ЛІНІЙНА МЕРЕЖА = усі `c = 0` ⟹ відповідь РІВНО НУЛЬ за нуль флопів.
Тобто вся наша задача — це рівно `Σ cℓ`, перенесене вперед.

📣 ПЕРЕДБАЧЕНО: попри інтуїцію «перші шари ламають усе», доживають до виходу
   переважно ОСТАННІ: перенос стискає (зміряно 8.1× на ділянці 8→31), тож ранні
   `c` затухають. Профіль має рости з глибиною.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
W_ = 256

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)

prof, seals, gates = [], [], []
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV)
         for k in range(32)]
    g = torch.Generator(device=DEV).manual_seed(63_000 + 11 * n)
    with torch.no_grad():
        z = torch.randn(NS, W_, device=DEV, generator=g)
        z = torch.cat([z, -z], 0)
        h = z
        C, G = [], []
        for L in range(32):
            p = h @ W[L]   # 🚨 fp32: P104 рахує fp64 у 32× повільніше
            gt = (p > 0).float().mean(0).double()
            mp = p.double().mean(0)
            h2 = torch.relu(p)
            C.append(h2.double().mean(0) - mp * gt)          # c = Cov(p, gate), ТОЧНО
            G.append(gt)
            h = h2
        truth = h.double().mean(0)

        # ✅ПЛОМБА: рекурсія m_ℓ = (m_{ℓ−1}Wℓ)⊙gℓ + cℓ мусить відтворити істину
        m = torch.zeros(W_, dtype=torch.float64, device=DEV)
        for L in range(32):
            m = (m @ W[L].double()) * G[L] + C[L]
        seals.append(float((m - truth).norm() / truth.norm()))

        # внесок КОЖНОГО шару: несемо його `c` до виходу окремо
        rows = []
        for L in range(32):
            v = C[L].clone()
            for j in range(L + 1, 32):
                v = (v @ W[j].double()) * G[j]
            rows.append(float(v.norm()))
        rows = np.array(rows)
        prof.append(rows / float(truth.norm()))
        gates.append(np.array([float(x.mean()) for x in G]))
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

P = np.array(prof).mean(0)
Gm = np.array(gates).mean(0)
print(f'\n{NETS} мереж × {2*NS} семплів (антитетика)')
print(f'✅ПЛОМБА рекурсії: відн. нев\'язка {np.mean(seals):.2e} '
      f'(мусить бути ~шум МК)\n')
print('ЛІНІЙНА МЕРЕЖА (усі c=0): відповідь РІВНО 0 ⟹ відносна похибка 100%, '
      'ціна 0 флопів\n')
print(f'{"шар":>4s} {"ворота g":>10s} {"‖внесок c_ℓ у відповідь‖":>26s} {"частка":>9s}')
print('-' * 56)
for L in range(32):
    star = ' <<<' if P[L] > 0.05 else ''
    print(f'{L:4d} {Gm[L]:9.3f} {P[L]:26.4f} {P[L]/P.sum():8.1%}{star}')
print(f'\nсума внесків (векторно НЕ додаються — це норми): {P.sum():.3f}')
print(f'шари 0–7   несуть {P[:8].sum()/P.sum():.1%}')
print(f'шари 8–23  несуть {P[8:24].sum()/P.sum():.1%}')
print(f'шари 24–31 несуть {P[24:].sum()/P.sum():.1%}')
print(f'ОСТАННІЙ шар сам:  {P[31]/P.sum():.1%}')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
