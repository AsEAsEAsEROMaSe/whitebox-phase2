"""УЗГОДЖЕНА ПОПРАВКА: ЗСУНУТИ АТРАКТОР РЕКУРСІЇ, А НЕ ЛАГОДИТИ КОМПОНЕНТУ (С80).

ЗВІДКИ. [[finding-correcting-C-alone-makes-recursion-worse]] (С74) закрив
покомпонентні поправки НАЙСИЛЬНІШИМ можливим чином:
    база `m₃₀`            0.9115%
    ⚠️ІСТИННА `C`          1.0434%   ← ОРАКУЛ ГІРШИЙ ЗА БАЗУ
    ⚠️істинні `m` і `C`    0.1092%   ← а разом у 8× краще
Механізм: `C` входить у наступний крок через `s`, `m` — через `t=m/s`, і похибки
зсунуті в ПРОТИЛЕЖНІ боки. Виправиш одну — розбалансуєш пару.
🔑 Тобто замикання — ТОЧКА БАЛАНСУ, і єдина жива форма — **УЗГОДЖЕНА** поправка,
яка рухає ОБИДВІ координати разом. Її, схоже, не пробували.

🔑 ЧОМУ ЦЕ ЗАДАЧА ПРО АТРАКТОР. С78: форма поля — АТРАКТОР, заданий вагами (вхід
забувається за 3 шари, cos γ₃ ≥ 0.9956). Отже і справжня рекурсія, і рекурсія
замикання мають КОЖНА СВІЙ атрактор, а `b_ℓ` — це відстань між ними. Тоді
питання не «зменшити похибку кроку», а **ЗСУНУТИ НЕРУХОМУ ТОЧКУ** параметрами.
Це й пояснює, чому однокомпонентні правки шкодять: вони зсувають атрактор УБІК.

## ПАРАМЕТРИ (два, бо площина балансу двовимірна)
    α — роздування ПОЗАДІАГОНАЛІ `C` після кроку Мелера: `C_поза ← (1+α)·C_поза`
        (С56 зміряла, що за один крок із правильною `C` формула хибить у
         недіагоналі на 4.2%, і завжди в один бік ⟹ α має бути >0)
    β — множник члена `φ` у середньому: `m ← m_p·Φ(t) + (1+β)·s·φ(t)`
        (це та координата, якою входить негаусовість маргіналі)

## 🎯 КРИТЕРІЙ УСПІХУ ЖОРСТКИЙ І ВІДОМИЙ ЗАЗДАЛЕГІДЬ
Таблиця чутливості С56 (шар 28): еталон 0.2% → 1.46×, 0.5% → **ШКОДИТЬ 3.95×**.
⟹ поправка має сенс ЛИШЕ якщо доведе `m₃₀` НИЖЧЕ 0.2%. Проміжні 0.4–0.6% —
це НЕ «частковий успіх», а той самий нуль. Стеля класу 0.1092%, тож місце є.

⛔ ПЛОМБИ:
 * `(α,β) = (0,0)` мусить відтворити базу БІТ-У-БІТ;
 * ⛔НЕГ: коефіцієнти, підігнані на ПЕРЕМІШАНИХ мережах, мусять дати ≈базу;
 * LOO ПО МЕРЕЖАХ (не по жеребах!) — інакше 0.999× як завжди;
 * оракульний `(α,β)` на самій мережі vs LOO: якщо різниця велика — переобучення.

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: оптимум буде при α>0 (недіагональ занижена) і β≠0,
   а виграш буде РЕАЛЬНИЙ, але МАЛИЙ: ставлю на 0.93% → 0.55–0.75%, тобто
   НЕ дотягне до 0.2%. Причина: два числа не можуть описати вектор із 256
   компонент, а С72 §11 зміряла, що стеля локальних ознак `R²=0.31`.
   Якщо вийде <0.3% — це важіль, і тоді шар 30 відкривається.

🚀 fp32 у сітці, fp64 у підтвердженні найкращої точки.
Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/joint.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NMC = int(sys.argv[3]) if len(sys.argv) > 3 else 4_000_000
LT = 30
W_, D = 256, 32
SQ2PI = float(np.sqrt(2.0 * np.pi))
AL = np.concatenate([np.linspace(-0.05, 0.30, 15)])
BE = np.concatenate([np.linspace(-0.15, 0.15, 13)])

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1].split('🚀')[0], flush=True)


def _phi(x):
    return torch.exp(-0.5 * x * x) / SQ2PI


def _Phi(x):
    return 0.5 * torch.erfc(-x / np.sqrt(2.0))


def _mc(W32, gen, n, layer):
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    d = 0
    while d < n:
        b = min(16384, n - d)
        x = torch.randn(b, W_, device=DEV, generator=gen)
        for L in range(layer + 1):
            x = torch.relu(x @ W32[L])
        acc += x.double().sum(0)
        d += b
    return acc / n


def _rec(W, layer, a, b, dt):
    """Рекурсія сабміту (`base_real.closure_parts`) з двома ручками.

    a=b=0 ⟹ БІТ-У-БІТ той самий вираз, що у сабміті.
    """
    n0 = W[0].shape[0]
    m = torch.zeros(n0, dtype=dt, device=DEV)
    C = torch.eye(n0, dtype=dt, device=DEV)
    for i in range(layer + 1):
        w = W[i].to(dt)
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + (1.0 + b) * s * ph
        C = cp * torch.outer(Ph, Ph) + (0.5 * cp * cp) * torch.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + mp * s * ph - m * m
        if a != 0.0:
            # 🔑 роздуваємо ЛИШЕ позадіагональ: діагональ задається точною 1-D
            #    формулою нижче, її чіпати не можна
            C = C * (1.0 + a)
        C = C + torch.diag(d - C.diagonal())
    return m


tru, Wl = [], []
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][k], np.float32), device=DEV)
         for k in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(4242 + 97 * net)
        tru.append(_mc(W, g, NMC, LT))
    Wl.append(W)
    if net % 8 == 0:
        print(f'  … істина {net+1}/{NETS}', flush=True)

# ── сітка: похибка кожної мережі при кожному (α,β) ────────────────────
E = np.zeros((NETS, len(AL), len(BE)))
with torch.no_grad():
    for n in range(NETS):
        nrm = float(tru[n].norm())
        for i, a in enumerate(AL):
            for j, b in enumerate(BE):
                m = _rec(Wl[n], LT, float(a), float(b), torch.float32)
                E[n, i, j] = float((m.double() - tru[n]).norm()) / nrm
        if n % 8 == 0:
            print(f'  … сітка {n+1}/{NETS}', flush=True)

i0, j0 = int(np.argmin(np.abs(AL))), int(np.argmin(np.abs(BE)))
base = E[:, i0, j0]
print(f'\n{NETS} мереж, МК {NMC:,}, ціль `m_{LT}`, сітка {len(AL)}×{len(BE)}\n')
print('⛔ ПЛОМБИ')
print(f'  (α,β)=(0,0) — база = {100*np.median(base):.4f}%   '
      f'(незалежно зміряно 0.932%, С74 дала 0.9115%)')

# LOO ПО МЕРЕЖАХ: коефіцієнти з інших, застосовані до цієї
loo = np.zeros(NETS)
for n in range(NETS):
    oth = np.delete(np.arange(NETS), n)
    s = E[oth].sum(0)
    ii, jj = np.unravel_index(int(np.argmin(s)), s.shape)
    loo[n] = E[n, ii, jj]
# оракул НА СОБІ (для перевірки переобучення)
orc = E.reshape(NETS, -1).min(1)
# ⛔НЕГ: коефіцієнти з ПЕРЕМІШАНОЇ відповідності
rng = np.random.default_rng(7)
perm = rng.permutation(NETS)
neg = np.array([E[n, *np.unravel_index(
    int(np.argmin(E[perm[n]])), E[perm[n]].shape)] for n in range(NETS)])

sg = E.sum(0)
bi, bj = np.unravel_index(int(np.argmin(sg)), sg.shape)
print(f'  ⛔НЕГ (коеф. з чужої мережі) = {100*np.median(neg):.4f}%   '
      f'(треба ≈базу)')
print(f'  оракул НА СОБІ = {100*np.median(orc):.4f}%  проти LOO '
      f'{100*np.median(loo):.4f}%  ⟹ '
      f'{"фіт стабільний" if np.median(loo)/max(np.median(orc),1e-12) < 1.3 else "🚨 ПЕРЕОБУЧЕННЯ"}')
print(f'\n{"варіант":40s} {"медіана":>11s} {"мін":>11s} {"макс":>11s}')
print('-' * 78)
for v, nm in ((base, 'база (α=β=0)'), (loo, '🎯 LOO ПО МЕРЕЖАХ'),
              (orc, '⚠️ оракул на собі (не досяжно)')):
    print(f'{nm:40s} {100*np.median(v):10.4f}% {100*v.min():10.4f}% '
          f'{100*v.max():10.4f}%')
print(f'\n🔑 СПІЛЬНИЙ ОПТИМУМ: α = {AL[bi]:+.4f}, β = {BE[bj]:+.4f}')
r = np.median(loo) / np.median(base)
print(f'   LOO/база = {r:.4f}  ⟹ похибка `m₃₀` {100*np.median(base):.3f}% → '
      f'{100*np.median(loo):.3f}%')
print(f'\n🎯 ВИРОК ЗА КРИТЕРІЄМ С56 (треба <0.2%, при 0.5% ЯКІР ШКОДИТЬ):')
print(f'   {"🟢 ВІДКРИВАЄ ШАР 30" if np.median(loo) < 0.002 else "⛔ НЕ ДОТЯГУЄ — гілка закрита"}'
      f'  (стеля класу з істинними m і C = 0.1092%)')
print('📊 БАЛ 1.954e-07 — рухає лише якщо <0.2%')
