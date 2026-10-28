"""СТАТИСТИКА РІВНІВ — ПРИЛАД ПОЛАГОДЖЕНО (С80, друга спроба).

⛔ ЧОМУ ПЕРША СПРОБА (`levels.py`) НЕ РАХУЄТЬСЯ.
Позитивний контроль ПРОВАЛИВСЯ: глибина 1 — це ОДНА гаусова матриця, ⟨r⟩ мав
вийти ≈GOE 0.536, вийшов **0.3710** (≈пуассонівські 0.386). Заразом ⛔НЕГ дав
**0.19–0.23**, тобто НИЖЧЕ за Пуассона, чого бути не може, якщо проміжки
розподілені експоненційно.

🔑 ДІАГНОЗ. Я міряв ⟨r⟩ на ЛОГАРИФМАХ сингулярних чисел. Відношення сусідніх
проміжків нечутливе до густини рівнів лише коли та густина МІНЯЄТЬСЯ ПОВІЛЬНО
на масштабі проміжку. Логарифм біля нуля розтягує спектр як завгодно сильно
(жорсткий край ансамблю Марченка-Пастура), густина летить, і прилад міряє
розтяг, а не відштовхування. ⛔НЕГ нижче за Пуассона — прямий симптом: наш
розподіл проміжків ШИРШИЙ за експоненційний.

🔧 ЩО ЗМІНЕНО
 1. **тільки ОБ'ЄМ** — центральні `BULK` рівнів, краї ансамблю відкинуто;
 2. **РОЗГОРТАННЯ** — поліном степеня `DEG` підганяємо до сходинки
    «рівень → номер», рівні відображаємо в нього ⟹ середній проміжок = 1;
 3. обидва варіанти друкуються поруч, щоб було видно ціну кожного кроку;
 4. ✅ДОДАНО контроль на СИРИХ сингулярних числах глибини 1 (без логарифма).

✅ ВОРОТА ПРИЛАДУ: поки глибина 1 не дасть ≈0.536, решта рядків НЕ ЧИТАЄТЬСЯ.

📣 ПЕРЕДБАЧЕНО (те саме, що й до першої спроби): ⟨r⟩ росте з глибиною і на 32
   перевищує GUE 0.603. Якщо після лагодження глибина 1 дасть 0.536, а глибина
   32 лишиться там же — глибина спектр НЕ жорсткішає, і моя картина хибна.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/levels2.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W_, D = 256, 32
KS = [1, 2, 4, 8, 16, 32]
BULK, DEG = 0.70, 8
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('🔧')[0], flush=True)


def _r(y):
    s = np.diff(np.sort(y))
    s = s[s > 0]
    if len(s) < 3:
        return np.nan
    a, b = s[:-1], s[1:]
    return float(np.mean(np.minimum(a, b) / np.maximum(a, b)))


def _bulk(x):
    x = np.sort(np.asarray(x, np.float64))
    n = len(x)
    c = int(n * (1.0 - BULK) / 2)
    return x[c:n - c]


def _unfold(x):
    """Розгортання: поліном підганяємо до сходинки «рівень → номер»."""
    x = _bulk(x)
    i = np.arange(len(x), dtype=np.float64)
    try:
        p = np.polyfit(x, i, DEG)
    except Exception:
        return np.nan
    return _r(np.polyval(p, x))


def sweep(mats):
    Q = torch.eye(W_, dtype=torch.float64, device=DEV)
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    out = {}
    for k, M in enumerate(mats, 1):
        Q, R = torch.linalg.qr(M.transpose(0, 1).double() @ Q)
        acc = acc + R.diagonal().abs().clamp_min(1e-300).log()
        if k in KS:
            out[k] = acc.cpu().numpy().copy()
    return out


def gated(W, z):
    mats, a = [], z
    for L in range(D):
        p = a @ W[L]
        mats.append(W[L] * (p > 0).to(p.dtype).reshape(-1).unsqueeze(0))
        a = torch.relu(p)
    return mats


def attractor(W, gen, it=800):
    x = torch.randn(1, W_, device=DEV, generator=gen)
    x = x / x.norm()
    for _ in range(it):
        a = x
        for L in range(D):
            a = torch.relu(a @ W[L])
        x = a / a.norm().clamp_min(1e-30)
    return x


RB = {k: [] for k in KS}          # об'єм, БЕЗ розгортання
RU = {k: [] for k in KS}          # об'єм + РОЗГОРТАННЯ
RN = {k: [] for k in KS}          # ⛔НЕГ: ті самі проміжки, перетасовані
RAW1 = []                         # ✅сирі сингулярні числа глибини 1
rng = np.random.default_rng(2026)
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV)
         for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(770 + 13 * net)
        sp = sweep(gated(W, attractor(W, g)))
        RAW1.append(_r(_bulk(np.exp(sp[1]))))
        for k in KS:
            RB[k].append(_r(_bulk(sp[k])))
            RU[k].append(_unfold(sp[k]))
            s = np.diff(_bulk(sp[k]))
            RN[k].append(_r(np.concatenate([[0.0],
                                            np.cumsum(rng.permutation(s))])))
    del W
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

poi = np.mean([_r(rng.random(int(W_ * BULK))) for _ in range(200)])
goe, goeu = [], []
for _ in range(64):
    A = rng.standard_normal((W_, W_))
    e = np.linalg.eigvalsh((A + A.T) / 2)
    goe.append(_r(_bulk(e)))
    goeu.append(_unfold(e))


def ln(nm, v):
    v = np.array(v, float)
    v = v[~np.isnan(v)]
    return f'{nm} {v.mean():.4f}±{v.std(ddof=1)/np.sqrt(len(v)):.4f}'


print(f'\n{NETS} мереж. ⟨r⟩ — об\'єм {int(100*BULK)}%, розгортання поліномом '
      f'степеня {DEG}\n')
print('✅ЕТАЛОНИ (теорія): Пуассон 0.3863 · GOE 0.5359 · GUE 0.6027 · '
      'частокіл 1.0000')
print(f'✅ЗМІРЯНО: Пуассон {poi:.4f} · {ln("GOE об%s єм" % chr(39), goe)} · '
      f'{ln("GOE розгорн.", goeu)}')
print(f'🔑 ВОРОТА: {ln("глибина 1 на СИРИХ сингулярних числах", RAW1)}  '
      f'← мусить бути ≈0.536\n')
print(f'{"глибина":>8s} {"об\'єм":>18s} {"РОЗГОРНУТО":>20s} '
      f'{"⛔НЕГ перетасовані":>22s}')
print('-' * 74)
for k in KS:
    print(f'{k:8d} {ln("", RB[k]):>18s} {ln("", RU[k]):>20s} '
          f'{ln("", RN[k]):>22s}')
print('\n📊 БАЛ 1.954e-07 — вимір КЛАСИФІКАЦІЙНИЙ, важеля не дає')
