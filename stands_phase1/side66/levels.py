"""СТАТИСТИКА РІВНІВ: наш спектр — випадкова матриця чи жорсткіший? (С80)

Юзер: «це схоже як шукати нулі в дзета-функції» → «постав експеримент на це».

## ЗВІДКИ ПИТАННЯ
Гіпотеза Монтґомері–Дайсона: відстані між нулями дзети розподілені так само, як
відстані між власними значеннями ВИПАДКОВИХ МАТРИЦЬ. А ми буквально є добутком
32 гаусових матриць. С80 уже зміряла ОДИН бік цього: дійсних власних значень
168/256 проти едельманівських 12.8 для ОДНІЄЇ матриці
([[finding-attractor-is-a-left-eigenvector-and-deep-spectrum-is-real]]).
Тут — ДРУГИЙ бік: чи є ВІДШТОВХУВАННЯ РІВНІВ, і як воно міняється з глибиною.

## ПРИЛАД: ВІДНОШЕННЯ СУСІДНІХ ПРОМІЖКІВ `r`
`s_n = x_{n+1} − x_n`,  `r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})`.
🔑 Обрано САМЕ його, бо він НЕ ПОТРЕБУЄ «розгортання» (нормування на локальну
густину) — а розгортання є головним джерелом артефактів у цій темі.

| ансамбль | ⟨r⟩ |
|---|---|
| Пуассон (рівні НЕЗАЛЕЖНІ, відштовхування нема) | 0.3863 |
| GOE (дійсні симетричні випадкові) | 0.5359 |
| GUE (комплексні ермітові випадкові) | 0.6027 |
| ідеальна ґратка («частокіл») | 1.0000 |

## ЩО САМЕ МІРЯЄМО
`x` = логарифми сингулярних чисел добутку `W₀G₀·…·W_{k-1}G_{k-1}`, здобуті
QR-розгорткою (стабільно; пряме множення 32 матриць виходить за межі fp64 —
на цьому вже падав `capture.py`). Свіп по глибині `k=1,2,4,8,16,32` показує,
чи жорсткість НАРОСТАЄ з глибиною.

## КОНТРОЛІ
 * ✅ПОЗ-1 глибина 1 = ОДНА гаусова матриця ⟹ мусить дати ≈GOE 0.536;
 * ✅ПОЗ-2 синтетичний Пуассон (256 незалежних чисел) ⟹ мусить дати ≈0.386;
 * ✅ПОЗ-3 частокіл (рівні проміжки) ⟹ мусить дати 1.000;
 * ⛔НЕГ: наші ж рівні, ПЕРЕТАСОВАНІ як незалежні з ТІЄЇ САМОЇ густини
   (беремо наші проміжки й переставляємо) ⟹ мусить впасти до Пуассона;
 * ✅ворота з ВИПАДКОВОГО семпла проти воріт АТРАКТОРА — не має бути різниці
   ([[finding-attractor-is-just-a-typical-sample]]).

📣 ПЕРЕДБАЧЕНО ДО ПРОГОНУ: ⟨r⟩ РОСТЕ з глибиною і на 32 буде ВИЩЕ за GUE
   (0.60), десь 0.7–0.9 — бо логарифми сингулярних чисел довгого добутку
   збігаються до показників Ляпунова, а ті САМОУСЕРЕДНЮЮТЬСЯ, тобто спектр
   «кристалізується». Якщо ⟨r⟩ застрягне на 0.536 — глибина спектр НЕ міняє,
   і моя картина «глибина робить спектр жорстким» хибна.
🎓 ЗАЗДАЛЕГІДЬ: це КЛАСИФІКАЦІЯ, не важіль. Виграшу в балі вона не дає й дати
   не може — вимір відповідає на «що це за об'єкт», а не «як його здешевити».

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/levels.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W_, D = 256, 32
KS = [1, 2, 4, 8, 16, 32]
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ПРИЛАД')[0], flush=True)


def rstat(x):
    """⟨r⟩ по відсортованій послідовності рівнів."""
    x = np.sort(np.asarray(x, np.float64))
    s = np.diff(x)
    s = s[s > 0]
    if len(s) < 3:
        return np.nan
    a, b = s[:-1], s[1:]
    return float(np.mean(np.minimum(a, b) / np.maximum(a, b)))


def sweep(mats):
    """log-сингулярні числа добутку, накопичені QR-розгорткою по шарах."""
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
    """Матриці `W_ℓ·G_ℓ` уздовж траєкторії семпла `z`."""
    mats, a = [], z
    for L in range(D):
        p = a @ W[L]
        g = (p > 0).to(p.dtype).reshape(-1)
        mats.append(W[L] * g.unsqueeze(0))
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


RA = {k: [] for k in KS}          # ворота атрактора
RZ = {k: [] for k in KS}          # ✅ворота випадкового семпла
RN = {k: [] for k in KS}          # ⛔НЕГ: наші проміжки, перетасовані
GAP = []
rng = np.random.default_rng(2026)
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV)
         for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(770 + 13 * net)
        xs = attractor(W, g)
        zr = torch.randn(1, W_, device=DEV, generator=g)
        for src, box in ((xs, RA), (zr, RZ)):
            sp = sweep(gated(W, src))
            for k in KS:
                box[k].append(rstat(sp[k]))
                if box is RA:                     # ⛔НЕГ на тих самих рівнях
                    s = np.diff(np.sort(sp[k]))
                    RN[k].append(rstat(np.concatenate(
                        [[0.0], np.cumsum(rng.permutation(s))])))
        v = np.sort(sp[32])[::-1]
        GAP.append(float(np.exp(v[0] - v[1])))
    del W
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

# ✅ПОЗ-2 Пуассон і ✅ПОЗ-3 частокіл
poi = np.mean([rstat(rng.random(W_)) for _ in range(200)])
pick = rstat(np.arange(W_, dtype=np.float64))
goe = []
for _ in range(64):                               # ✅ПОЗ-1b: справжня GOE
    A = rng.standard_normal((W_, W_))
    goe.append(rstat(np.linalg.eigvalsh((A + A.T) / 2)))

print(f'\n{NETS} мереж. ВІДНОШЕННЯ СУСІДНІХ ПРОМІЖКІВ ⟨r⟩\n')
print('✅ЕТАЛОНИ:  Пуассон(теор) 0.3863 · GOE(теор) 0.5359 · GUE(теор) 0.6027')
print(f'✅ЗМІРЯНО:  Пуассон {poi:.4f} · GOE {np.mean(goe):.4f} · '
      f'частокіл {pick:.4f}\n')
print(f'{"глибина":>8s} {"🟢ворота атрактора":>20s} {"✅ворота семпла":>17s} '
      f'{"⛔НЕГ (перетасовані)":>21s}')
print('-' * 72)
for k in KS:
    a = np.array(RA[k]); z = np.array(RZ[k]); nn = np.array(RN[k])
    print(f'{k:8d} {a.mean():13.4f}±{a.std(ddof=1)/np.sqrt(len(a)):.4f} '
          f'{z.mean():11.4f}±{z.std(ddof=1)/np.sqrt(len(z)):.4f} '
          f'{nn.mean():15.4f}±{nn.std(ddof=1)/np.sqrt(len(nn)):.4f}')
print(f'\n🔑 розрив λ₁/λ₂ на глибині 32: медіана {np.median(GAP):.4f} '
      f'(С79 для сингулярних чисел: 2.72)')
print('📊 БАЛ 1.954e-07 — вимір КЛАСИФІКАЦІЙНИЙ, важеля не дає')
