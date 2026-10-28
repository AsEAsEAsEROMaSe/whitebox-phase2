"""ЧИ ОКУПАЄТЬСЯ ІНТЕГРАЦІЯ ВЗДОВЖ ЛІНІЇ: те саме число обчислень (С78).

`lines.py` дав ціну: 8328 зламків на півколо ⟹ ~130 семплів-еквівалентів за одну
пряму. Питання: чи варта пряма 130 незалежних точок.

Порівнюємо при ОДНАКОВІЙ кількості проходів мережі:
  A  K незалежних антитетичних пар (як у сабміті, з ZCA)
  B  M великих кіл по K/M точок кожне (точки на 1-вимірному зрізі сфери)

Обидві оцінюють E[o]. Дивимось дисперсію по жеребах ⟹ хто ефективніший НА ТОЧКУ.
🔑 Велике коло гасить усі НЕПАРНІ сферичні гармоніки повністю — рівно як
   антитетика. Тому виграш можливий тільки на ПАРНИХ, і питання, чи його
   вистачить, щоб перебити кореляцію точок усередині кола.
"""
import sys
import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NPT = int(sys.argv[3]) if len(sys.argv) > 3 else 4096   # точок на оцінку
DRAWS = int(sys.argv[4]) if len(sys.argv) > 4 else 24
W_, D = 256, 32
MS = (1, 4, 16, 64)                                     # скільки кіл
from math import lgamma
# E‖z‖ = √2·Γ(128.5)/Γ(128) — радіус, відокремлений позитивною однорідністю
ENZ = float(np.sqrt(2.0) * np.exp(lgamma(128.5) - lgamma(128.0)))
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('🔑')[1], flush=True)

def _fwd(W, x):
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x.double().mean(0)

res = np.zeros(1 + len(MS)); ref = 0.0
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    g = torch.Generator(device=DEV).manual_seed(8100 + 7 * n)
    with torch.no_grad():
        big = torch.randn(1 << 16, W_, device=DEV, generator=g)
        truth = _fwd(W, torch.cat([big, -big], 0))
        ref += float((truth * truth).sum())
        for d in range(DRAWS):
            z = torch.randn(NPT // 2, W_, device=DEV, generator=g)
            e = _fwd(W, torch.cat([z, -z], 0)) - truth
            res[0] += float((e * e).sum())
            for mi, M in enumerate(MS):
                K = NPT // (2 * M)                 # точок на півколо
                acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
                for _ in range(M):
                    a = torch.randn(W_, device=DEV, generator=g)
                    b = torch.randn(W_, device=DEV, generator=g)
                    a = a / a.norm(); b = b - (b @ a) * a; b = b / b.norm()
                    th = torch.linspace(0, float(np.pi), K, device=DEV)
                    # 🔑 радіус ОКРЕМО: E[o] = E‖z‖·E[o(ẑ)] (позитивна однорідність)
                    u = torch.outer(th.cos(), a) + torch.outer(th.sin(), b)
                    acc += _fwd(W, torch.cat([u, -u], 0))
                acc = acc / M * ENZ
                e = acc - truth
                res[1 + mi] += float((e * e).sum())
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж × {DRAWS} жеребів по {NPT} точок (однакова ціна)\n')
print(f'{"оцінка":>22s} {"MSE":>13s} {"проти iid":>11s}')
print('-' * 50)
print(f'{"A iid антитетика":>22s} {res[0]:13.4e} {1.0:10.3f}×')
for mi, M in enumerate(MS):
    print(f'{("B " + str(M) + " кіл × " + str(NPT//(2*M))):>22s} '
          f'{res[1+mi]:13.4e} {res[0]/max(res[1+mi],1e-30):10.3f}×')
best = res[1:].min()
print(f'\n🔑 НАЙКРАЩЕ КОЛО: {res[0]/max(best,1e-30):.3f}×  '
      f'{"🟢 ВИГРАШ" if best < res[0] else "⛔ ПРОГРАШ — і це БЕЗ ціни відстеження зламків"}')
print('   🚨 ціна точної інтеграції ще +130 семплів на пряму (`lines.py`)')
print('📊 БАЛ: 1.954e-07 — вимір прайсує клас')
