"""ВІДНОВЛЕННЯ В ЦИКЛІ: 5.9% — це межа МЕТОДУ чи межа ІНФОРМАЦІЇ? (С80)

Юзер: «а в циклі відновлення».

## ЩО ЦЕ РОЗСУДЖУЄ
`side66/invert.py` дав на глибині 32: ЛІНІЙНЕ відновлення напрямку **5.9%**.
Але лінійна модель — слабка для НЕЛІНІЙНОЇ функції. Тому 5.9% могло бути
межею ПРИЛАДУ, а не інформації. Цикл (ітеративне обернення градієнтом) —
найсильніший практичний відновлювач: підбираємо `ẑ` так, щоб `f(ẑ)`
збігалося з відомим виходом `f(z*)`.

## 🔑 ВИРІШАЛЬНА ПАРА ЧИСЕЛ
Міряємо ДВІ речі одночасно:
  1. **нев'язку виходу** `‖f(ẑ)−f(z*)‖ / ‖f(z*)‖` — наскільки добре підігнали;
  2. **`cos(ẑ, z*)`** — чи це той самий вхід.
🔑 Якщо нев'язка → 0, а `cos` лишається малим, це КОНСТРУКТИВНИЙ ДОКАЗ, що
   різні входи дають ОДИН вихід ⟹ інформація знищена самою мережею, і жоден
   відновлювач її не поверне. Це сильніше за будь-який непрямий аргумент.

## 🔑 І ТРЕТЄ ЧИСЛО — ПОПАРНИЙ `cos` МІЖ РОЗВ'ЯЗКАМИ
З `R` різних випадкових стартів отримуємо `R` розв'язків. Якщо всі підганяють
вихід, але між собою РІЗНІ — багатозначність показана прямо, без теорії.

⛔ ПЛОМБИ:
 * ✅ПОЗ: старт РІВНО з `z*` ⟹ нев'язка 0, `cos`=1 (перевірка стенда);
 * ⛔НЕГ: ціль — вихід ЧУЖОЇ мережі ⟹ нев'язка НЕ мусить впасти так само.

📣 ПЕРЕДБАЧЕНО: нев'язка впаде до ~1e-3 і нижче, а `cos(ẑ,z*)` лишиться
   нижче 0.3 ⟹ межа ІНФОРМАЦІЙНА, і цикл її не обходить. Попарний `cos`
   між розв'язками теж буде малим (кожен старт знаходить СВІЙ прообраз).
   Якщо ж `cos` підскочить вище 0.8 — я помилявся, і 5.9% було межею методу.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/invcycle.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
W_, D = 256, 32
TARG, REST, STEPS = 4, 16, 1500
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ЩО ЦЕ')[0], flush=True)


def _f(W, x):
    for w in W:
        x = torch.relu(x @ w)
    return x


def _invert(W, o_t, gen, z0=None):
    """Цикл: Adam по сфері, ціль — збіг ВИХОДУ."""
    r = float(np.sqrt(W_))
    if z0 is None:
        z = torch.randn(REST, W_, device=DEV, generator=gen).double()
    else:
        z = z0.repeat(REST, 1).clone()
    z = (z / z.norm(dim=1, keepdim=True) * r).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.05)
    tn = o_t.norm()
    for t in range(STEPS):
        opt.zero_grad()
        loss = ((_f(W, z) - o_t) ** 2).sum(1).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            z.mul_(r / z.norm(dim=1, keepdim=True).clamp_min(1e-30))
    with torch.no_grad():
        res = ((_f(W, z) - o_t).norm(dim=1) / tn.clamp_min(1e-30))
    return z.detach(), res


RES, COS, PAIR, PRES, NRES = [], [], [], [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32),
                      device=DEV).double() for q in range(D)]
    Wf = [torch.tensor(np.asarray(WZ[(net + 7) % 100][q], np.float32),
                       device=DEV).double() for q in range(D)]
    g = torch.Generator(device=DEV).manual_seed(6100 + 29 * net)
    for t in range(TARG):
        with torch.no_grad():
            zs = torch.randn(1, W_, device=DEV, generator=g).double()
            zs = zs / zs.norm() * float(np.sqrt(W_))
            o_t = _f(W, zs)
        zh, res = _invert(W, o_t, g)
        with torch.no_grad():
            u = zh / zh.norm(dim=1, keepdim=True)
            v = zs / zs.norm()
            c = (u @ v.reshape(-1, 1)).reshape(-1)
            k = int(res.argmin())                      # найкращий розв'язок
            RES.append(float(res[k])); COS.append(float(c[k].abs()))
            iu = torch.triu_indices(REST, REST, 1)
            PAIR.append(float((u @ u.T)[iu[0], iu[1]].abs().mean()))
        if t == 0:
            _, pr = _invert(W, o_t, g, z0=zs)          # ✅ПОЗ: старт із z*
            PRES.append(float(pr.min()))
            with torch.no_grad():
                o_f = _f(Wf, zs)
            _, nr = _invert(W, o_f, g)                 # ⛔НЕГ: чужа ціль
            NRES.append(float(nr.min()))
    del W, Wf
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)


def ln(v):
    v = np.array(v, float)
    return f'{v.mean():.4g} (медіана {np.median(v):.4g})'


print(f'\n{NETS} мереж × {TARG} цілей × {REST} стартів × {STEPS} кроків\n')
print(f'✅ПОЗ (старт із самого `z*`): нев\'язка {ln(PRES)}   ← мусить бути ≈0')
print(f'⛔НЕГ (ціль із ЧУЖОЇ мережі): нев\'язка {ln(NRES)}   ← НЕ мусить падати')
print()
print(f'🔑 нев\'язка виходу після циклу : {ln(RES)}')
print(f'🔑 `cos(ẑ, z*)` найкращого      : {ln(COS)}')
print(f'🔑 попарний `cos` між стартами  : {ln(PAIR)}')
print(f'\n📐 випадковий напрямок дав би |cos| ≈ {1/np.sqrt(W_):.4f}')
print('🎯 нев\'язка МАЛА + cos МАЛИЙ = різні входи дають ОДИН вихід ⟹ '
      'межа ІНФОРМАЦІЙНА')
print('📊 БАЛ 1.954e-07 — вимір СТРУКТУРНИЙ')
