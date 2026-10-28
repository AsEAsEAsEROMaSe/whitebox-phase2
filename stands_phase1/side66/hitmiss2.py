"""ЧОМУ КОНТРОЛЬ У `hitmiss.py` ДАВ −0.103 ЗАМІСТЬ 0 (С80).

`hitmiss.py` намірив `cos` двох НЕЗАЛЕЖНИХ промахів = −0.1029 при очікуваних 0
(≈6 сигм). Поки це не з'ясовано, решту чисел того стенда читати не можна.

🔑 ГІПОТЕЗА: розподіл `cos` не вузький навколо нуля, а ДВОГОРБИЙ біля ±1 —
   бо вектор виходу майже одновимірний (С79: усі семпли дивляться в один бік,
   `cos(o₁,E[o])`=0.9885). Тоді промах — це майже СКАЛЯР (помилка масштабу
   вздовж однієї осі), знак випадковий, і МЕДІАНА по 24 мережах гуляє.
   Якщо так — «промах у випадковому 256-вимірному напрямку» НЕВІРНО, і це
   змінює вирок: скалярну ручку можна виправляти.

## ЩО МІРЯЄМО
 1. сам розподіл `cos(e₁,e₂)` — усі 24 значення, а не медіана;
 2. **частку енергії промаху вздовж головного напрямку** істини `T/‖T‖`
    і вздовж власного головного напрямку промахів;
 3. ✅ КОНТРОЛЬ РОЗМІРНОСТІ: те саме для СИНТЕТИЧНОГО гаусового шуму тієї ж
    форми — там мусить вийти рівно `1/√256`;
 4. 🔑 **скільки НЕЗАЛЕЖНИХ напрямків** у промаху: `(Σλ)²/Σλ²` по 64 повторах.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/hitmiss2.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
REP = 64
K = 16384
TRUTH = 1 << 21
W_, D, CH = 256, 32, 8192
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')


def _fwd(W, z):
    for L in range(D):
        z = torch.relu(z @ W[L])
    return z


def _mean(W, gen, npair, anti):
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    done = 0
    while done < npair:
        k = min(CH, npair - done)
        z = torch.randn(k, W_, device=DEV, generator=gen)
        acc += _fwd(W, torch.cat([z, -z], 0) if anti else z).double().sum(0)
        done += k
    return acc / ((2.0 if anti else 1.0) * done)


cc, rad, eff, own = [], [], [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV)
         for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(4200 + 31 * net)
        T = _mean(W, g, TRUTH, True)
        u = T / T.norm()
        E = torch.stack([_mean(W, g, K, False) - T for _ in range(REP)])
        En = E / E.norm(dim=1, keepdim=True)
        M = En @ En.T
        iu = torch.triu_indices(REP, REP, 1)
        cc.append(float(M[iu[0], iu[1]].mean()))
        rad.append(float(((En @ u) ** 2).mean()))          # частка вздовж істини
        lam = torch.linalg.eigvalsh(E.T @ E / REP).clamp_min(0)
        eff.append(float(lam.sum() ** 2 / (lam ** 2).sum()))
        own.append(float(lam.max() / lam.sum()))           # частка в 1-й осі
    del W
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

G = torch.randn(REP, W_, dtype=torch.float64, device=DEV)   # ✅контроль
Gn = G / G.norm(dim=1, keepdim=True)
iu = torch.triu_indices(REP, REP, 1)
lg = torch.linalg.eigvalsh(G.T @ G / REP)
cc, rad, eff, own = map(np.array, (cc, rad, eff, own))

print(f'\n{NETS} мереж × {REP} незалежних промахів по {K} семплів\n')
print(f'{"величина":44s} {"наше":>10s} {"✅ізотропний шум":>17s}')
print('-' * 74)
print(f'{"середній cos(e_i,e_j)":44s} {cc.mean():10.4f} '
      f'{float(Gn[iu[0]] .mul(Gn[iu[1]]).sum(1).mean()):17.4f}')
print(f'{"частка енергії вздовж істини  (1/256=0.0039)":44s} {rad.mean():10.4f} '
      f'{float(((Gn @ (G[0]/G[0].norm())) ** 2).mean()):17.4f}')
print(f'{"частка енергії в 1-й головній осі":44s} {own.mean():10.4f} '
      f'{float(lg.max()/lg.sum()):17.4f}')
print(f'{"ефективна кількість напрямків (з 256)":44s} {eff.mean():10.1f} '
      f'{float(lg.sum()**2/(lg**2).sum()):17.1f}')
print(f'\nрозкид cos по мережах: min {cc.min():+.4f}  медіана {np.median(cc):+.4f}'
      f'  max {cc.max():+.4f}')
print(f'🔑 якби промах був СКАЛЯРОМ — ефективних напрямків було б 1, '
      f'частка вздовж істини 1.0')
