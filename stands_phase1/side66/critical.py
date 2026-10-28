"""ЧИ СХОДЯТЬСЯ КРИТИЧНІ СЕМПЛИ І ЯК (С80, питання юзера).

Юзер: «перевір, чи критичні семпли сходяться і як. наприклад також поламані або
максимізовані. загалом різні критичні».

ЗВІДКИ ВОНИ ВІДОМІ (С79): носій `S=Σo_j` замкнено з обох боків —
    `max ≥ 986` (47 сигм від середнього, `maxima.py`, Adam по сфері)
    `min = 0` РІВНО, конструктивно: `z = −(W₀⁻¹)ᵀ·1` ⟹ всі `z·w_i = −1`
    `E[S] = 124.2`
С79 гнала ітерацію лише з ВИПАДКОВИХ стартів (1024 шт, 100% в один атрактор).
Чи потрапляють туди КРАЇ носія — не питали.

## ШІСТЬ ТИПІВ СТАРТУ
  1 типовий гаусів                       — контроль, має збігтися (С79)
  2 МАКСИМІЗОВАНИЙ (Adam по сфері)       — верхній край носія
  3 МІНІМАЛЬНИЙ `z = −(W₀⁻¹)ᵀ1`          — нижній край, `S=0` ТОЧНО
  4 НЕВІД'ЄМНИЙ `|z|`                    — режим, у який ітерація сама заганяє
  5 ОДНА КООРДИНАТА (орт `e₀`)           — гранично вироджений
  6 ⛔НЕГ: атрактор ЧУЖОЇ мережі          — має НЕ бути нерухомим тут

## ЩО МІРЯЄМО ДЛЯ КОЖНОГО
  * чи прийшов у ТОЙ САМИЙ атрактор (`cos` із еталонним `x*`);
  * за скільки ітерацій (до `cos > 1−1e-6`);
  * `S` на старті (де на носії) і `λ` на старті проти сталого;
  * ⛔чи не вироджується (норма в нуль).

🔑 ГІПОТЕЗА, ЯКУ ПЕРЕВІРЯЄМО: мінімум — ОСОБЛИВИЙ. Там `f(z)=0` РІВНО, отже
   нормувати нема чого: `0/0`. Тобто це не «повільна збіжність», а ДРУГА
   нерухома точка — поглинальна. Якщо так, твердження С79/С80 «атрактор ОДИН»
   треба уточнити: один СЕРЕД НЕНУЛЬОВИХ.

📣 ПЕРЕДБАЧЕНО: 1,2,4,5 зійдуться в той самий `x*` (басейн майже глобальний);
   3 (мінімум) дасть нуль і НЕ зійдеться; ⛔НЕГ не буде нерухомим.
   Максимум зійдеться ШВИДШЕ за типовий — він уже лежить у бік найсильнішого
   розтягу, а це і є напрямок атрактора.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/critical.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
ITERS = 400
W_, D = 256, 32
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)


def _f(W, X):
    for L in range(D):
        X = torch.relu(X @ W[L])
    return X


def _S(W, x):
    return float(_f(W, x.reshape(1, -1)).sum())


def _attr(W, gen, it=1500):
    x = torch.randn(1, W_, device=DEV, generator=gen)
    x = x / x.norm()
    for _ in range(it):
        y = _f(W, x)
        x = y / y.norm().clamp_min(1e-30)
    return x.reshape(-1)


def _maximize(W, gen, steps=400):
    """Верхній край носія: Adam по сфері ‖z‖=√d (як `maxima.py` у С79)."""
    r = float(np.sqrt(W_))
    z = torch.randn(1, W_, device=DEV, generator=gen)
    z = (z / z.norm() * r).requires_grad_(True)
    opt = torch.optim.Adam([z], lr=0.3)
    for _ in range(steps):
        opt.zero_grad()
        (-_f(W, z).sum()).backward()
        opt.step()
        with torch.no_grad():
            z.mul_(r / z.norm().clamp_min(1e-30))
    return z.detach().reshape(-1)


def _minimize(W):
    """⛔КОНСТРУКЦІЯ (С79): `z = −(W₀⁻¹)ᵀ·1` ⟹ всі `z·w_i = −1` ⟹ шар 0 мертвий."""
    w0 = W[0].double()
    one = torch.ones(W_, dtype=torch.float64, device=DEV)
    return (-torch.linalg.solve(w0.transpose(0, 1), one)).float()


rows = []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(606 + 19 * net)
        xs = _attr(W, g)
    Wf = [torch.tensor(np.asarray(WZ[(net + 5) % 100][k], np.float32),
                       device=DEV) for k in range(D)]
    with torch.no_grad():
        xf = _attr(Wf, g)
    del Wf
    zmax = _maximize(W, g)
    with torch.no_grad():
        starts = {
            'типовий гаусів': torch.randn(W_, device=DEV, generator=g),
            'МАКСИМІЗОВАНИЙ': zmax,
            'МІНІМАЛЬНИЙ (S=0)': _minimize(W),
            'невід\'ємний |z|': torch.randn(W_, device=DEV,
                                            generator=g).abs(),
            'одна координата': torch.eye(W_, device=DEV)[0],
            '⛔НЕГ чужий x*': xf,
        }
        Emean = float(_f(W, torch.randn(4096, W_, device=DEV, generator=g)
                         ).sum(1).mean())
        r = {}
        for nm, z0 in starts.items():
            x = (z0 / z0.norm().clamp_min(1e-30)).reshape(1, -1)
            s0 = _S(W, x.reshape(-1))
            lam0 = float(_f(W, x).norm())
            it, dead = -1, False
            for k in range(ITERS):
                y = _f(W, x)
                ny = float(y.norm())
                if ny < 1e-25:
                    dead = True
                    break
                x = y / ny
                if it < 0 and float((x.reshape(-1) * xs).sum()) > 1 - 1e-6:
                    it = k + 1
            c = -1.0 if dead else float((x.reshape(-1) * xs).sum())
            r[nm] = (c, it, s0, lam0, float(dead))
        rows.append(r)
    print(f'  … мережа {net+1}/{NETS}  (E[S]={Emean:.1f})', flush=True)
    del W
    torch.cuda.empty_cache()

print(f'\n{NETS} мереж, до {ITERS} ітерацій\n')
print(f'{"старт":22s} {"cos із x*":>11s} {"ітерацій":>9s} '
      f'{"S старту / E[S]":>16s} {"λ старту":>9s} {"вироджень":>10s}')
print('-' * 86)
for nm in rows[0]:
    c = np.array([r[nm][0] for r in rows])
    it = np.array([r[nm][1] for r in rows], float)
    s0 = np.array([r[nm][2] for r in rows])
    lm = np.array([r[nm][3] for r in rows])
    dd = np.array([r[nm][4] for r in rows])
    itv = it[it > 0]
    print(f'{nm:22s} {np.median(c):11.6f} '
          f'{(np.median(itv) if len(itv) else float("nan")):9.1f} '
          f'{np.median(s0):16.4g} {np.median(lm):9.4f} '
          f'{int(dd.sum()):6d}/{NETS}')

hit = {nm: float(np.mean([r[nm][0] > 1 - 1e-4 for r in rows])) for nm in rows[0]}
print(f'\n🔑 ЧАСТКА, ЩО ПРИЙШЛА В ТОЙ САМИЙ АТРАКТОР:')
for nm, v in hit.items():
    print(f'   {nm:22s} {100*v:5.0f}%')
print('📊 БАЛ 1.954e-07 — вимір СТРУКТУРНИЙ')
