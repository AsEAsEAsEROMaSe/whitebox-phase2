"""ПОЗИТИВНА ОДНОРІДНІСТЬ: РАДІАЛЬНА ЧАСТИНА ВІДОМА ТОЧНО. (С85)

## ЗВІДКИ — ВИВІД З НУЛЯ, ТІЛЬКИ З ВВІДНИХ КОНКУРСУ
Мережа: `h_ℓ = W_ℓᵀ a_{ℓ−1}`, `a_ℓ = ReLU(h_ℓ)`, **БЕЗ ЗСУВІВ**.
ReLU додатно однорідний: `ReLU(cx) = c·ReLU(x)` для `c>0`; матмул лінійний ⟹
🔑 **УСЯ МЕРЕЖА ДОДАТНО ОДНОРІДНА СТЕПЕНЯ 1: `a_L(cz) = c·a_L(z)`.**
Пишемо `z = r·u`, `r = ‖z‖`, `u = z/‖z‖`. Для `z ~ N(0,I_n)` **`r` і `u`
НЕЗАЛЕЖНІ**, `u` рівномірний на сфері ⟹
    **`E[a_L(z)] = E[r] · E_u[a_L(u)]`**
і `E[r] = √2·Γ((n+1)/2)/Γ(n/2)` — ТОЧНЕ ЧИСЛО (для n=256 це 15.98…).
⟹ 🎯 радіальну компоненту дисперсії можна прибрати **ТОЧНО**, взявши всі
семпли однакової норми `√n` і помноживши результат на `E[r]/√n`.
Це НЕ те саме, що whitening: whitening фіксує `Σ‖z‖²/N = n` (другий момент),
а `a_L ∝ ‖z‖` — потрібен ПЕРШИЙ.

## ЧОМУ ЦЕ МОЖЕ БУТИ ВЕЛИКИМ
`Var(a_L) = E[r²]·Var_u(a_L(u)) + Var(r)·E_u[a_L(u)]²`.
У 256 вимірах `a_L(u)` КОНЦЕНТРУЄТЬСЯ по напрямках (Var_u мала) ⟹ частка
радіального доданка може бути СУТТЄВОЮ, хоча сам `Var(r)/E[r]² ≈ 1/(2n) = 0.2%`.
Гейт: якщо варіант B дасть ≥1.02× — гілка жива.

## ВАРІАНТИ (усі з антитетикою `±z`, однаковий `npair`)
  A чистий МК                    — база без будь-яких трюків
  B МК + ТОЧНА НОРМА             — `z_k = √n·u_k`, результат × `E[r]/√n`
  C МК + whitening               — як у сабміті
  D whitening + точна норма ДО   — чи додається щось понад whitening
⛔ПЛОМБА: C мусить бути помітно кращим за A (whitening — відомий виграш).

📣 ПЕРЕДБАЧЕНО ДО ЗАПУСКУ: B побʼє A (радіальна частина реальна), але D ≈ C,
   бо whitening уже фіксує `Σ‖z‖²/N = n` точно, а звідси `Σ‖z‖/N` відхиляється
   лише у ДРУГОМУ порядку `~Var/(8n²)`. Тобто ідея правильна, але вже майже
   реалізована іншим способом. Шанс на приріст понад whitening: ~20%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/homog.py [cuda:N] [NETS] [NDRAW]
🚫 cuda:0 — ТАБУ, cuda:4 — монітор ЮЗЕРА. Робочі: cuda:1, cuda:2, cuda:3.
"""
import glob
import math
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import cut_budget91, W_                            # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NDRAW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
L_ = 32
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

# E[r] = √2·Γ((n+1)/2)/Γ(n/2) — через lgamma, щоб не переповнити
E_R = math.sqrt(2.0) * math.exp(math.lgamma((W_ + 1) / 2) - math.lgamma(W_ / 2))
print(f'E[‖z‖] для n={W_}: {E_R:.6f}  (√n = {math.sqrt(W_):.6f}, '
      f'відношення {E_R/math.sqrt(W_):.8f})\n', flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів, {DEV}\n', flush=True)


def run(wc, npair, dev, seed, mode):
    """Прямий МК-прохід. mode: 'A' | 'B' | 'C' | 'D'."""
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((128, W_))                       # пропуск пілота
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    scale = 1.0
    if mode in ('B', 'D'):        # 🎯 ТОЧНА НОРМА: z_k = √n·u_k
        z = z * (math.sqrt(W_) / z.norm(dim=1, keepdim=True))
        scale = E_R / math.sqrt(W_)
    if mode in ('C', 'D'):        # whitening, як у сабміті
        cov = (z.T @ z) / float(npair)
        ev, V = torch.linalg.eigh(cov.double())
        wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
        z = z @ wh
    a = torch.cat([z, -z], 0)                            # антитетика
    for i in range(len(wc)):
        a = torch.relu(a @ wc[i])
    return (a.double().sum(0) / float(2 * npair)) * scale


MODES = [('A чистий МК', 'A'), ('B + ТОЧНА НОРМА', 'B'),
         ('C + whitening (сабміт)', 'C'), ('D whitening + норма', 'D')]
MSE = {nm: np.zeros(len(FS)) for nm, _ in MODES}

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for nm, md in MODES:
            acc = 0.0
            for d in range(NDRAW):
                p = run(wc, npair, DEV, 51_000_000 + 733*n + d, md)
                acc += float(((p - T1) * (p - T2)).mean()) / NDRAW
            MSE[nm][n] = acc
        print(f'  … {n+1}/{len(FS)} seed {sd0} npair={npair} · ' +
              ' '.join(f'{md}={MSE[nm][n]:.3e}' for nm, md in MODES), flush=True)
    del W, wc
    torch.cuda.empty_cache()

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/homog.npz', **MSE)
base = MSE['A чистий МК'].mean()
print(f'\n{"варіант":>26s} {"MSE(крос)":>13s} {"× до A":>9s} {"× до C":>9s}')
print('─' * 62)
cc = MSE['C + whitening (сабміт)'].mean()
for nm, _ in MODES:
    print(f'{nm:>26s} {MSE[nm].mean():13.6e} {base/MSE[nm].mean():9.4f} '
          f'{cc/MSE[nm].mean():9.4f}')
print('─' * 62)
print('\n🎯 ГЕЙТ: B/A ≥ 1.02 ⟹ радіальна частина реальна;')
print('   D/C ≥ 1.02 ⟹ вона НЕ покрита whitening і гілка ЖИВА.')
print('   D/C ≈ 1.00 ⟹ whitening уже забирає її ⟹ вивід правильний, важеля нема.')
