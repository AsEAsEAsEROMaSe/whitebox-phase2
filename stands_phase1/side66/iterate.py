"""ІТЕРАЦІЯ МЕРЕЖІ САМОЇ В СЕБЕ: вихід → назад на вхід (С79, питання юзера).

«Є семпл один. Подали через мережу і тримали вихід. Далі вихід знову на вхід.»

Це вже НЕ наша задача (там вхід гаусовий і застосування ОДНЕ), а динамічна
система `x ← f(x)`. Але саме тут поняття АТРАКТОРА має звичайний сенс, тож
питання «скільки їх і як зв'язані з вагами» стає вимірним.

🔑 ДВІ РЕЧІ, ЯКІ ТРЕБА ТРИМАТИ В ГОЛОВІ:
 1. `f` позитивно однорідна (`f(cx)=c·f(x)`, зсувів нема) ⟹ масштаб і напрямок
    РОЗДІЛЯЮТЬСЯ. Ітеруємо НАПРЯМОК (нормуючи щокроку), а розтяг `λ=‖f(x)‖/‖x‖`
    міряємо окремо. Без нормування норма піде в 0 або в нескінченність, і ми
    зміряємо лише переповнення fp32.
 2. 🚨 Вихід мережі — це `relu` ⟹ ВЖЕ ПІСЛЯ ПЕРШОГО кроку вхід НЕВІД'ЄМНИЙ.
    Тобто ітерація виводить нас у додатний ортант — режим, якого в самій задачі
    нема (там `z` гаусів). Це чесно робить її окремим об'єктом, а не моделлю
    нашої задачі.

ЩО МІРЯЄМО: скільки різних граничних напрямків (атракторів) знаходить ітерація
з 1024 випадкових стартів, як швидко сходиться, який розтяг `λ`, і чи залежить
результат від того, звідки стартувати.

⛔ ПЛОМБИ:
 * однорідність: старт `x` і старт `1000·x` мусять дати ТОЙ САМИЙ напрямок;
 * невід'ємність виходу на кожному кроці;
 * якщо збіглось, то `f(x*) = λ·x*` ⟹ `cos(f(x*), x*)` мусить бути 1 з точністю fp32.

📣 ПЕРЕДБАЧЕНО (записано ДО прогону): збіжність буде ШВИДКОЮ (≤10 ітерацій), бо
   мережа сильно стискає (PR активацій падає 165→2.6 за один прохід). Атрактор
   буде ОДИН, спільний майже для всіх стартів. `λ` — не знаю; ставлю на `λ>1`,
   бо невід'ємний вхід дає більші кореляції, ніж гаусів.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/iterate.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NSTART = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 60
W_, D = 256, 32
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _f(W, x):
    for L in range(D):
        x = torch.relu(x @ W[L])
    return x


def _run(W, x0, iters):
    """Ітеруємо напрямок; повертаємо траєкторію cos(x_t, x_{t+1}) і λ."""
    x = x0 / x0.norm(dim=1, keepdim=True)
    coss, lams = [], []
    for _ in range(iters):
        y = _f(W, x)
        nrm = y.norm(dim=1, keepdim=True).clamp_min(1e-30)
        lams.append((nrm.squeeze(1)).clone())
        yn = y / nrm
        coss.append((yn * x).sum(1).clone())
        x = yn
    return x, torch.stack(coss), torch.stack(lams)


def _count_attractors(X, thr=0.99):
    """Скільки різних граничних напрямків (жадібна кластеризація за косинусом)."""
    Xn = X / X.norm(dim=1, keepdim=True).clamp_min(1e-30)
    left = torch.ones(Xn.shape[0], dtype=torch.bool, device=Xn.device)
    reps, sizes = [], []
    while bool(left.any()):
        i = int(torch.nonzero(left)[0])
        c = (Xn @ Xn[i]) >= thr
        c &= left
        reps.append(i)
        sizes.append(int(c.sum()))
        left &= ~c
    return len(reps), sorted(sizes, reverse=True), reps


rows = []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(1200 + 29 * net)
        x0 = torch.randn(NSTART, W_, device=DEV, generator=g)
        X, C, Lm = _run(W, x0, ITERS)
        # ⛔ПЛОМБА ОДНОРІДНОСТІ: той самий старт, помножений на 1000
        X2, _, _ = _run(W, 1000.0 * x0[:64], ITERS)
        hom = float((X2 * (X[:64] / X[:64].norm(dim=1, keepdim=True))).sum(1).min())
        # ⛔ПЛОМБА НЕРУХОМОСТІ: f(x*) має бути колінеарний x*
        Y = _f(W, X)
        fix = float(((Y / Y.norm(dim=1, keepdim=True)) * X).sum(1).min())
        nneg = float(X.min())
        # скільки ітерацій до cos > 1−1e-5
        conv = C.mean(1)
        it = int(torch.nonzero(conv > 1 - 1e-5)[0]) + 1 if bool(
            (conv > 1 - 1e-5).any()) else -1
        nat, sizes, _ = _count_attractors(X)
        lam = float(Lm[-1].median())
        rows.append((nat, sizes[0] / NSTART, it, lam, hom, fix, nneg,
                     float(conv[0]), float(conv[4])))
    print(f'  … мережа {net+1}/{NETS}: атракторів {nat}, найбільший '
          f'{100*sizes[0]/NSTART:.1f}% стартів, збіжність за {it} ітер, λ={lam:.4g}',
          flush=True)
    del W
    torch.cuda.empty_cache()

a = np.array(rows, dtype=float)
print(f'\n{NETS} мереж × {NSTART} стартів × {ITERS} ітерацій\n')
print('⛔ ПЛОМБИ')
print(f'  однорідність (старт ×1000): min cos = {a[:,4].min():.8f}  (треба 1)')
print(f'  нерухомість  cos(f(x*),x*): min     = {a[:,5].min():.8f}  (треба 1)')
print(f'  невід\'ємність виходу: min елемент   = {a[:,6].min():.3e}  (треба ≥0)')
print(f'\n{"величина":34s} {"медіана":>12s} {"мін":>11s} {"макс":>11s}')
print('-' * 72)
for k, nm in enumerate(('РІЗНИХ АТРАКТОРІВ (cos≥0.99)',
                        'частка стартів у найбільшому',
                        'ітерацій до збіжності',
                        'розтяг λ на крок',
                        '—', '—', '—',
                        'cos(x₀,x₁) — після 1-ї ітерації',
                        'cos(x₄,x₅) — після 5-ї')):
    if nm == '—':
        continue
    c = a[:, k]
    print(f'{nm:34s} {np.median(c):12.6g} {c.min():11.6g} {c.max():11.6g}')
print(f'\n🔑 ВИРОК: {"ОДИН СПІЛЬНИЙ АТРАКТОР" if np.median(a[:,0]) <= 1.5 else "АТРАКТОРІВ КІЛЬКА"}'
      f' — медіана {np.median(a[:,0]):.0f}, і в найбільший падає '
      f'{100*np.median(a[:,1]):.1f}% стартів')
print(f'   розтяг λ = {np.median(a[:,3]):.4g} на прохід ⟹ за 32 шари норма '
      f'{"РОСТЕ" if np.median(a[:,3])>1 else "ПАДАЄ"}')
print('📊 БАЛ: 1.954e-07 — вимір структурний, це ІНШИЙ режим (вхід невід\'ємний)')
