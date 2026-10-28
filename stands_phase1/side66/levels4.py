"""СТАТИСТИКА РІВНІВ — ЗВІРКА QR ПРОТИ СПРАВЖНІХ СИНГУЛЯРНИХ ЧИСЕЛ (С80).

## ⛔ ТРИ ПОПЕРЕДНІ СПРОБИ І ЩО ВОНИ НАСПРАВДІ СКАЗАЛИ
`levels.py`/`levels2.py`/`levels3.py` тричі дали «глибина 1 = 0.385 замість
0.536». Я тричі шукав ваду в приладі (логарифм · краї · ворота). Помилковим
було ПЕРЕДБАЧЕННЯ.

🔑 ДІАГНОЗ. QR-розгортка на ГЛИБИНІ 1 дає НЕ сингулярні числа, а діагональ `R`
розкладу однієї гаусової матриці. За розкладом Бартлетта ці діагоналі —
НЕЗАЛЕЖНІ випадкові величини. Незалежні ⟹ Пуассон ⟹ рівно **0.386**.
Тобто контроль ПРОЙШОВ — просто проти правильного еталона, а не мого.
QR сходиться до справжнього спектра лише З ГЛИБИНОЮ (теорема Оселедця).

## ЩО ЦЕЙ СТЕНД РОБИТЬ
Ставить поруч на КОЖНІЙ глибині:
  * ✅**СПРАВЖНІ сингулярні числа** добутку (`svdvals`, поки fp64 витримує) —
    на глибині 1 мусять дати ≈GOE 0.536 (відштовхування ансамблю Вішарта);
  * 🟢**QR-накопичення** — те, чим я міряв.
Якщо на глибині 1 вони РОЗІЙДУТЬСЯ (0.536 проти 0.386), а з глибиною ЗІЙДУТЬСЯ
— діагноз підтверджено, і число на глибині 32 читати МОЖНА.
Якщо не зійдуться — QR не годиться, і весь вимір знімається.

📊 Плюс: динамічний розмах `log σ_max − log σ_min` по глибині — він і каже,
   доки взагалі можна формувати добуток у fp64.

📣 ПЕРЕДБАЧЕНО: на глибині 1 svd дасть ≈0.536, QR ≈0.386; до глибини 8 вони
   зійдуться в межах похибки; і обидва лишаться біля 0.386–0.45, тобто
   ⛔МОЯ ГІПОТЕЗА «глибина кристалізує спектр» ХИБНА, а правильна відповідь —
   показники Ляпунова асимптотично НЕЗАЛЕЖНІ.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/levels4.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W_, D = 256, 32
KS = [1, 2, 4, 8, 16, 32]
BULK = 0.70
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ЩО ЦЕЙ')[0], flush=True)


def _r(y):
    s = np.diff(np.sort(np.asarray(y, np.float64)))
    s = s[s > 0]
    if len(s) < 4:
        return np.nan
    a, b = s[:-1], s[1:]
    return float(np.mean(np.minimum(a, b) / np.maximum(a, b)))


def _bulk(x):
    x = np.sort(np.asarray(x, np.float64))
    x = x[np.isfinite(x)]
    n = len(x)
    c = int(n * (1.0 - BULK) / 2)
    return x[c:n - c] if n >= 8 else x


RS, RQ, SPAN = ({k: [] for k in KS} for _ in range(3))
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV).double()
         for q in range(D)]
    with torch.no_grad():
        P = torch.eye(W_, dtype=torch.float64, device=DEV)   # справжній добуток
        Q = torch.eye(W_, dtype=torch.float64, device=DEV)   # QR-розгортка
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        for k in range(1, D + 1):
            M = W[k - 1]
            P = P @ M
            Q, R = torch.linalg.qr(M.transpose(0, 1) @ Q)
            acc = acc + R.diagonal().abs().clamp_min(1e-300).log()
            if k in KS:
                sv = torch.linalg.svdvals(P).clamp_min(1e-300).log()
                sv = sv.cpu().numpy()
                RS[k].append(_r(_bulk(sv)))
                RQ[k].append(_r(_bulk(acc.cpu().numpy())))
                SPAN[k].append(float(sv.max() - sv.min()))
    del W, P, Q
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

rng = np.random.default_rng(7)
goe = [_r(_bulk(np.linalg.eigvalsh(
    (lambda A: (A + A.T) / 2)(rng.standard_normal((W_, W_)))))) for _ in range(64)]
poi = np.mean([_r(rng.random(180)) for _ in range(200)])


def ln(v):
    v = np.array(v, float)
    v = v[~np.isnan(v)]
    return f'{v.mean():.4f}±{v.std(ddof=1)/np.sqrt(len(v)):.4f}'


print(f'\n{NETS} мереж. ⟨r⟩ на об\'ємі {int(100*BULK)}%\n')
print('✅ЕТАЛОНИ (теорія): Пуассон 0.3863 · GOE 0.5359 · GUE 0.6027')
print(f'✅ЗМІРЯНО: Пуассон {poi:.4f} · GOE {ln(goe)}\n')
print(f'{"глибина":>8s} {"✅СПРАВЖНІ svd":>19s} {"🟢QR-накопичення":>19s} '
      f'{"розмах log σ":>14s}')
print('-' * 66)
for k in KS:
    print(f'{k:8d} {ln(RS[k]):>19s} {ln(RQ[k]):>19s} '
          f'{np.median(SPAN[k]):13.1f}')
print(f'\n🔑 ВОРОТА: глибина 1 svd = {ln(RS[1])}  ← мусить бути ≈0.536 (Вішарт)')
print(f'🔑 глибина 1 QR = {ln(RQ[1])}  ← мусить бути ≈0.386 (Бартлетт)')
print('📊 БАЛ 1.954e-07 — вимір КЛАСИФІКАЦІЙНИЙ, важеля не дає')
