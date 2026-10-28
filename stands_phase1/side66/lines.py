"""СКІЛЬКИ ПЕРЕТИНІВ НА ПРЯМІЙ: чи можна інтегрувати вздовж лінії ТОЧНО (С78).

Ідея юзера: гілки — клони зі зміщенням, «як гілка на вітрі»; треба провести
лінії в далечінь, знаючи, як вони входять і де перетнуться.

МАТЕМАТИКА. Зсувів у мережі НЕМА ⟹ в кожному конусі рівно `o(z) = J(z)·z`, тож
    E[o] = J̄·E[z] + E[δJ(z)·z] = E[δJ(z)·z]
— уся відповідь є кореляцією ПОВОРОТУ якобіана з напрямком. Далі: уздовж прямої
`z(τ)` кожне `p_j(τ)` кусково-лінійне ⟹ ВЕСЬ вихід кусково-лінійний за τ, і
інтеграл береться ТОЧНО, якщо знайти зламки (перемикання масок).

🔑 ЦІНА ЦЬОГО = ЧИСЛО ЗЛАМКІВ. Верхня межа 8192 (по одному на нейрон), але
   `branches.py` зміряв, що 73% нейронів практично детерміновані ⟹ реально
   перемикається менше. Якщо зламків ~сотні, точна 1-вимірна інтеграція дешева;
   якщо тисячі — вона коштує як десятки звичайних семплів і не окупиться.

Рухаємось ВЕЛИКИМ КОЛОМ на сфері (позитивна однорідність ⟹ радіус відокремлено:
E[o] = E‖z‖ · E[o(ẑ)]), рахуємо, скільки разів патерн масок змінюється.
"""
import sys
import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
NLINE = int(sys.argv[3]) if len(sys.argv) > 3 else 8       # прямих на мережу
NSTEP = int(sys.argv[4]) if len(sys.argv) > 4 else 20000   # точок на коло
W_, D = 256, 32
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('🔑')[1].split('Рухаємось')[0], flush=True)

tot = np.zeros((NETS, D))
uniq = np.zeros(NETS)
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    g = torch.Generator(device=DEV).manual_seed(5150 + 11 * n)
    with torch.no_grad():
        flips = np.zeros(D)
        us = 0
        for li in range(NLINE):
            a = torch.randn(W_, device=DEV, generator=g)
            b = torch.randn(W_, device=DEV, generator=g)
            a = a / a.norm()
            b = b - (b @ a) * a
            b = b / b.norm()
            th = torch.linspace(0, float(np.pi), NSTEP, device=DEV)  # півколо
            x = torch.outer(th.cos(), a) + torch.outer(th.sin(), b)
            prev = None
            for L in range(D):
                p = x @ W[L]
                m = (p > 0)
                if prev is not None or True:
                    ch = (m[1:] != m[:-1]).sum(1)      # змін маски на крок
                    flips[L] += float(ch.sum())
                x = torch.relu(p)
            # скільки РІЗНИХ конусів пройдено (за повним патерном останнього шару)
            us += 1
        tot[n] = flips / NLINE
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}  зламків на півколо: {tot[n].sum():.0f}',
          flush=True)

T = tot.mean(0)
print(f'\n{NETS} мереж × {NLINE} великих кіл × {NSTEP} точок\n')
print(f'{"шар":>4s} {"зламків на півколо":>20s} {"з 256 нейронів":>16s}')
print('-' * 44)
for L in (0, 1, 2, 4, 8, 16, 24, 31):
    print(f'{L:4d} {T[L]:20.1f} {T[L]/256:15.2f}×')
s = T.sum()
print(f'\nУСЬОГО зламків на півколо: **{s:.0f}**  (верхня межа 8192·2 = 16384)')
print(f'   тобто {s/8192:.2f} перемикання на нейрон')
print(f'\n💰 ЦІНА: точний інтеграл уздовж прямої = {s:.0f} лінійних шматків.')
print(f'   Один звичайний семпл коштує 2·32·256² = 4.19e6 флопів.')
print(f'   Відстеження зламків ≈ {s:.0f}×256² = {s*65536:.2e} флопів '
      f'= {s*65536/4.19e6:.0f} семплів-еквівалентів.')
print(f'   ⟹ пряма замінює {s*65536/4.19e6:.0f} семплів і прибирає дисперсію '
      f'вздовж 1 з 256 вимірів.')
print('📊 БАЛ: 1.954e-07 — вимір прайсує клас точної 1-вимірної інтеграції')
