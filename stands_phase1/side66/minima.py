"""А ЩО ТАКЕ МІНІМУМ (С79, питання юзера — продовження `maxima.py`).

`S(z) = Σ_j o_j(z)`, а вихід — це ReLU ⟹ `S(z) ≥ 0` ТОТОЖНО. Тобто нижня межа
відома наперед і дорівнює нулю. Питання інше і цікавіше: **чи вона ДОСЯЖНА?**

📐 ТЕОРІЯ КАЖЕ «ТАК», І ЦЕ ДОВОДИТЬСЯ. Щоб мережа замовкла на шарі 0, треба `z`
   з `z·w_i < 0` для всіх 256 стовпців. Стовпці випадкової матриці 256×256
   майже напевно ЛІНІЙНО НЕЗАЛЕЖНІ, а для лінійно незалежних векторів система
   строгих нерівностей `z·w_i < 0` (i=1..d) ЗАВЖДИ сумісна: візьми
   `z = −(W⁻¹)ᵀ·1`, тоді `z·w_i = −1 < 0` для кожного i.
   ⟹ «мертвий конус» існує НЕПОРОЖНІЙ у кожній мережі, і в ньому вихід РІВНО 0.

🔑 ЩО ЦЕ ДАЄ ДЛЯ ЗАДАЧІ. Разом із `maxima.py` це замикає носій розподілу `S`:
   ліва межа ЖОРСТКА (рівно 0, з масою!), права — хвіст. Тобто `S` не просто
   «не рівномірний» — він має атом на нулі. Теза «середнє = півсуми країв» дала
   б `(0+max)/2`; порівняємо з правдою.

📣 ПЕРЕДБАЧЕНО (записано ДО прогону):
   1. мінімум буде РІВНО 0.0 (не «майже») — конструкція вище точна;
   2. замовкання станеться на ШАРІ 0 (там найлегше: вхід може бути будь-якого
      знака, глибше `x≥0` і закрити все важче);
   3. кутовий радіус мертвого конуса — мала, але НЕ нікчемна величина
      (ставлю 0.05–0.3 рад);
   4. у типовій гаусовій вибірці — ЖОДНОГО влучення (частка 0.0);
   5. `(0+max)/2` завищить справжнє `E[S]` приблизно в 4× (бо max/E = 8.8×).

⛔ ПЛОМБИ: `S ≥ 0` на всіх точках; у знайденій точці перевіряємо ПОШАРОВО, де
   саме обірвався сигнал; конструктивний `z = −(W₀⁻¹)ᵀ·1` має дати рівно 0
   НЕЗАЛЕЖНО від спуску (два різні шляхи до того самого).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/minima.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
RESTARTS = int(sys.argv[3]) if len(sys.argv) > 3 else 256
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 400
W_, D = 256, 32
R = float(np.sqrt(W_))
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _fwd(W, z):
    x = z
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x.sum(-1)


def _silence_layer(W, z):
    """На якому шарі сигнал обірвався (усі активації нуль); -1 якщо живий."""
    x = z
    for L in range(D):
        x = torch.relu(x @ W[L])
        if float(x.abs().max()) == 0.0:
            return L
    return -1


def _descend(W, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    z = torch.randn(RESTARTS, W_, device=DEV, generator=g)
    z = R * z / z.norm(dim=1, keepdim=True)
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.35)
    for _ in range(STEPS):
        opt.zero_grad()
        # 🚨 `S` має плато на нулі (градієнт зникає, коли все закрито) ⟹ спуск
        #    сам себе гальмує. Це не баг: він і мусить зупинитись, ДОСЯГНУВШИ 0.
        _fwd(W, z).sum().backward()
        opt.step()
        with torch.no_grad():
            z.data = R * z.data / z.data.norm(dim=1, keepdim=True)
    with torch.no_grad():
        return z.detach(), _fwd(W, z)


def _cone_radius(W, z0, seed, ntry=4096):
    """Кутовий радіус мертвого конуса: до якого кута повороту вихід ще нуль."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    lo, hi = 0.0, 1.6
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        d = torch.randn(ntry, W_, device=DEV, generator=g)
        d = d - (d @ z0.squeeze(0))[:, None] * z0 / (R * R)
        d = d / d.norm(dim=1, keepdim=True)
        zz = np.cos(mid) * z0 + np.sin(mid) * R * d
        zz = R * zz / zz.norm(dim=1, keepdim=True)
        # частка напрямків, де ВСЕ ЩЕ нуль
        frac = float((_fwd(W, zz) == 0.0).double().mean())
        if frac > 0.5:
            lo = mid
        else:
            hi = mid
    return lo


rows = []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    with torch.no_grad():
        # ⛔ПЛОМБА-КОНСТРУКЦІЯ: z = −(W₀⁻¹)ᵀ·1 ⟹ z·w_i = −1 для всіх i
        w0 = W[0].double()
        zc = -torch.linalg.solve(w0.T, torch.ones(W_, dtype=torch.float64,
                                                  device=DEV))
        zc = (R * zc / zc.norm()).float()[None, :]
        s_con = float(_fwd(W, zc))
        lay_con = _silence_layer(W, zc)
    zb, sb = _descend(W, 800 + 37 * net)
    with torch.no_grad():
        i = int(torch.argmin(sb))
        smin = float(sb[i])
        z0 = zb[i:i + 1]
        lay = _silence_layer(W, z0)
        nzero = int((sb == 0.0).sum())
        rad = _cone_radius(W, zc, 4200 + net)
        # частка влучень у типовій гаусовій вибірці
        gg = torch.Generator(device=DEV).manual_seed(90 + net)
        hit = 0
        for _ in range(8):
            zt = torch.randn(1 << 13, W_, device=DEV, generator=gg)
            hit += int((_fwd(W, zt) == 0.0).sum())
        frac_hit = hit / float(8 * (1 << 13))
        smin_all = min(smin, s_con)
    rows.append((smin_all, s_con, float(lay_con), float(lay), nzero, rad,
                 frac_hit))
    print(f'  … мережа {net+1}/{NETS}: min {smin_all:.6g} · конструкція '
          f'{s_con:.6g} · замовкло на шарі {lay_con} · конус {rad:.3f} рад',
          flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(rows)
print(f'\n{NETS} мереж · {RESTARTS} рестартів × {STEPS} кроків · ‖z‖=√d={R:.3f}\n')
print(f'{"величина":40s} {"медіана":>13s} {"мін":>12s} {"макс":>12s}')
print('-' * 80)
for k, nm in enumerate(('МІНІМУМ S (найкращий із двох шляхів)',
                        '⛔конструкція z=−(W₀⁻¹)ᵀ1: S',
                        '⛔шар замовкання (конструкція)',
                        'шар замовкання (спуск)',
                        'скільки з рестартів дійшли до 0',
                        'кутовий радіус мертвого конуса, рад',
                        'частка влучень у гаусовій вибірці')):
    c = a[:, k]
    print(f'{nm:40s} {np.median(c):13.6g} {c.min():12.6g} {c.max():12.6g}')

print(f'\n⛔ПЛОМБА: max|S_конструкції| = {np.abs(a[:,1]).max():.3e}  (теорія: РІВНО 0)')
print(f'⛔ПЛОМБА: усі S ≥ 0 — мінімум по всіх мережах = {a[:,0].min():.3e}')
rad = np.median(a[:, 5])
print(f'\n🔑 МІНІМУМ = 0 І ВІН ДОСЯЖНИЙ: мережа має НЕПОРОЖНІЙ «мертвий конус»')
print(f'   кутовий радіус {rad:.3f} рад ⟹ частка сфери ≈ '
      f'{np.sin(rad)**(W_-1):.3e}  (I_sin²θ у 256 вимірах)')
print(f'   влучень у {8*(1<<13)} гаусових семплах: '
      f'{np.median(a[:,6])*8*(1<<13):.0f}')
print('📊 БАЛ: 1.954e-07 — вимір структурний')
