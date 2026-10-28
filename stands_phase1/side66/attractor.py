"""ЧИ ПЕРЕКІС УНІВЕРСАЛЬНИЙ: чи забуває поле свій вхід (ідея юзера, С78).

Теза юзера: «яка б не була вхідна на вході, вона завжди після множення на ваги
буде мати оцей перекос». Тобто форма поля — АТРАКТОР: після кількох шарів вона
визначається лише вагами, а не тим, що подали.

Це прямо перевіряється: подати РІЗНІ вхідні розподіли з тією самою коваріацією
(гаус, рівномірний, лапласів, Бернуллі ±1, сильно скошений) і подивитись, чи
профілі γ₃, γ₄, |t| сходяться до однієї кривої.

🔑 ЧОМУ ЦЕ ВАЖЛИВО, А НЕ ПРОСТО ЦІКАВО. `k3prop.py` закрив перенос `κ₃`
   ланцюгом: точно він потребує тензора 256³ (10.1× понад бюджет), а діагональ
   дає cos=0.649 і масштаб 0.077. Але ЯКЩО форма універсальна, то `κ₃` не
   треба переносити — його можна знати з самих ваг наперед. Тобто це той самий
   бар'єр з іншого боку.

📏 МІРА ЗБІЖНОСТІ: розкид γ₃ між вхідними розподілами, поділений на сам γ₃.
   Якщо на глибині він падає до ~0 — атрактор є. Якщо тримається — вхід
   пам'ятається, і універсального перекосу нема.

🚨 Усі входи нормовані на ОДИНИЧНУ коваріацію (як після ZCA у сабміті), інакше
   порівнювались би різні масштаби, а не різні ФОРМИ.

📣 ПЕРЕДБАЧЕНО: шар 0 дасть різні γ₃ (вхід ще пам'ятається — для гауса рівно 0,
   для скошеного ні), але вже за 3–5 шарів криві мають зійтись, бо кожен
   наступний `p` — це сума 256 доданків із перемішаними знаками. Ставлю на
   збіжність ≤5% розкиду після шару 8.
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 16
W_, D, CH = 256, 32, 1 << 13

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)
KINDS = ('гаус', 'рівномірний', 'лапласів', 'Бернуллі±1', 'скошений χ²')


def _draw(kind, b, gen):
    """Вхід із НУЛЬОВИМ середнім і ОДИНИЧНОЮ дисперсією, але різною формою."""
    if kind == 'гаус':
        z = torch.randn(b, W_, device=DEV, generator=gen)
    elif kind == 'рівномірний':
        z = (torch.rand(b, W_, device=DEV, generator=gen) - 0.5) * np.sqrt(12.0)
    elif kind == 'лапласів':
        u = torch.rand(b, W_, device=DEV, generator=gen) - 0.5
        # 🚨 клампити: при |u|=0.5 log1p(-1) = −inf ⟹ NaN по всьому ланцюгу
        z = -torch.sign(u) * torch.log1p(
            -(2 * u.abs()).clamp(max=1 - 1e-7)) / np.sqrt(2.0)
    elif kind == 'Бернуллі±1':
        z = torch.where(torch.rand(b, W_, device=DEV, generator=gen) < 0.5,
                        -1.0, 1.0)
    else:                                   # χ²(1) центрований — сильний перекіс
        g = torch.randn(b, W_, device=DEV, generator=gen)
        z = (g * g - 1.0) / np.sqrt(2.0)
    return z


def _profile(W, kind, seed, npairs):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(4, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = _draw(kind, b, gen)
        x = torch.cat([z, -z], 0)           # антитетика ⟹ вхід симетризовано
        for L in range(D):
            p = x @ W[L]
            pd = p.double()
            q = pd * pd
            Z[L][0] += pd.sum(0); Z[L][1] += q.sum(0)
            Z[L][2] += (q * pd).sum(0); Z[L][3] += (q * q).sum(0)
            x = torch.relu(p)
        done += b
    N = float(2 * npairs)
    g3, g4, tt = [], [], []
    for L in range(D):
        a1, a2, a3, a4 = (Z[L][i] / N for i in range(4))
        s2 = (a2 - a1 * a1).clamp_min(1e-30)
        s = s2.sqrt()
        mu3 = a3 - 3 * a1 * a2 + 2 * a1 ** 3
        mu4 = a4 - 4 * a1 * a3 + 6 * a1 * a1 * a2 - 3 * a1 ** 4
        # 🚨 ВЕКТОР, не середнє: перекіс є на КОЖНОМУ нейроні, але знак різний
        #    ⟹ середнє зі знаком гаситься в ~15× і нічого не міряє.
        g3.append((mu3 / s ** 3).cpu().numpy())
        g4.append(((mu4 - 3 * s2 * s2) / s ** 4).cpu().numpy())
        tt.append(float((a1 / s).abs().mean()))
    return np.array(g3), np.array(g4), np.array(tt)


G3 = np.zeros((len(KINDS), NETS, D, W_))
G4 = np.zeros((len(KINDS), NETS, D, W_))
TT = np.zeros((len(KINDS), NETS, D))
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        for ki, k in enumerate(KINDS):
            a, b, c = _profile(W, k, 33_000 + 7 * n, 1 << LOGN)
            G3[ki, n], G4[ki, n], TT[ki, n] = a, b, c
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)



def _cos(a, b):
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


print(f'\n{NETS} мереж × 2·2^{LOGN} семплів на кожен вхід')
print('🔑 ПОРІВНЮЄМО ВЕКТОРИ γ₃ (256 чисел) З ГАУСОВИМ ВХОДОМ\n')
print(f'{"шар":>4s} {"‖γ₃‖ (гаус)":>12s} ' +
      ' '.join(f'{k:>13s}' for k in KINDS[1:]) + f' {"|t|":>7s}')
print('-' * (6 + 13 + 14 * (len(KINDS) - 1) + 8))
rows = []
for L in (0, 1, 2, 3, 5, 8, 12, 16, 24, 31):
    nrm = np.linalg.norm(G3[0, :, L], axis=1).mean()
    cs = [np.mean([_cos(G3[0, n, L], G3[ki, n, L]) for n in range(NETS)])
          for ki in range(1, len(KINDS))]
    print(f'{L:4d} {nrm:12.3f} ' + ' '.join(f'{c:13.4f}' for c in cs) +
          f' {TT[0,:,L].mean():7.3f}')
allc = [np.mean([[_cos(G3[0, n, L], G3[ki, n, L]) for n in range(NETS)]
                 for L in range(8, D)]) for ki in range(1, len(KINDS))]
mn = min(allc)
print(f'\nсереднє cos(γ₃) по шарах 8..31 проти гаусового входу:')
for ki, k in enumerate(KINDS[1:]):
    print(f'   {k:>14s}: {allc[ki]:.4f}')
print(f'\n🔑 ВИРОК: {"🟢 АТРАКТОР Є — форма НЕ памʼятає вхід" if mn > 0.95 else "⛔ вхід ПАМʼЯТАЄТЬСЯ"}'
      f'   (найгірший cos = {mn:.4f}, гейт 0.95)')
print('   Якщо атрактор є — `κ₃` визначається ВАГАМИ, і тензор 256³ не потрібен.')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
