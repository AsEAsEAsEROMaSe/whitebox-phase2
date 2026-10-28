"""СТАТИСТИКА РІВНІВ — ТРЕТЯ СПРОБА: ОБ'ЄКТ БУВ НЕ ТОЙ (С80).

## ⛔ ЧОМУ ДВІ ПОПЕРЕДНІ НЕ РАХУЮТЬСЯ
`levels.py`  — ⟨r⟩ на логарифмах, глибина 1 дала 0.3710 замість ≈0.536.
`levels2.py` — об'єм + розгортання НЕ допомогли: 0.3885. Але ✅GOE-еталон
               обидва рази пройшов (0.5341) ⟹ САМА статистика реалізована
               правильно, ламався ОБ'ЄКТ.

🔑 СПРАВЖНІЙ ДІАГНОЗ. «Глибина 1» у моєму стенді — це НЕ гаусова матриця, а
`W₀·G₀`, тобто гаусова матриця з ОБНУЛЕНИМИ приблизно половиною стовпців
(ворота = 0/1). Її ранг ≈128, отже ≈128 сингулярних чисел РІВНО НУЛЬ, у
логарифмі — купа на технічній підлозі `log(1e-300)`. Об'єм 70% ту купу
включав. Тобто я міряв не відштовхування рівнів, а ширину штучної підлоги.
🎓 УРОК: **позитивний контроль мусить бути ТИМ САМИМ об'єктом, лише з відомою
відповіддю.** Я назвав контроль «одна гаусова матриця», а підсунув туди
матрицю з воротами — і контроль чесно завалився.

## ЩО В ЦІЙ ВЕРСІЇ
 1. ✅ПОЗ: **БЕЗ воріт** — чистий добуток `W₀·…·W_{k-1}`, повний ранг.
    Глибина 1 мусить дати ≈GOE 0.536. Це і є справжні ворота приладу.
 2. 🟢 наш об'єкт із воротами — але ⟨r⟩ рахується ЛИШЕ по рівнях ВИЩЕ
    числової підлоги, і поруч друкується, СКІЛЬКИ їх лишилось;
 3. 📊 профіль виживання рангу по глибині — самостійно цікаве число.

📣 ПЕРЕДБАЧЕНО: без воріт ⟨r⟩ на глибині 1 = 0.536, і РОСТЕ з глибиною вище
   GUE 0.603 (логарифми сингулярних чисел довгого добутку самоусереднюються).
   З воротами ранг швидко вмирає, і на великій глибині питання про статистику
   рівнів просто ВТРАЧАЄ ЗМІСТ — рівнів лишиться надто мало.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/levels3.py [cuda:N] [NETS]
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
W_, D = 256, 32
KS = [1, 2, 4, 8, 16, 32]
BULK = 0.70
FLOOR = -600.0                       # log(1e-260); нижче — технічний нуль
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('Запуск')[0].split('## ЩО В ЦІЙ')[0], flush=True)


def _r(y):
    s = np.diff(np.sort(np.asarray(y, np.float64)))
    s = s[s > 0]
    if len(s) < 4:
        return np.nan
    a, b = s[:-1], s[1:]
    return float(np.mean(np.minimum(a, b) / np.maximum(a, b)))


def _live_bulk(x):
    """Лише рівні вище технічної підлоги, далі центральні BULK."""
    x = np.sort(np.asarray(x, np.float64))
    x = x[x > FLOOR]
    n = len(x)
    if n < 8:
        return x
    c = int(n * (1.0 - BULK) / 2)
    return x[c:n - c]


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


def attractor(W, gen, it=800):
    x = torch.randn(1, W_, device=DEV, generator=gen)
    x = x / x.norm()
    for _ in range(it):
        a = x
        for L in range(D):
            a = torch.relu(a @ W[L])
        x = a / a.norm().clamp_min(1e-30)
    return x


RG = {k: [] for k in KS}      # 🟢 з воротами
RP = {k: [] for k in KS}      # ✅ПОЗ: БЕЗ воріт (повний ранг)
NL = {k: [] for k in KS}      # скільки рівнів вижило (з воротами)
rng = np.random.default_rng(2026)
for net in range(NETS):
    W = [torch.tensor(np.asarray(WZ[net][q], np.float32), device=DEV)
         for q in range(D)]
    with torch.no_grad():
        g = torch.Generator(device=DEV).manual_seed(770 + 13 * net)
        x = attractor(W, g)
        mg, a = [], x
        for L in range(D):
            p = a @ W[L]
            mg.append(W[L] * (p > 0).to(p.dtype).reshape(-1).unsqueeze(0))
            a = torch.relu(p)
        sg, sp = sweep(mg), sweep(W)
        for k in KS:
            NL[k].append(int((sg[k] > FLOOR).sum()))
            RG[k].append(_r(_live_bulk(sg[k])))
            RP[k].append(_r(_live_bulk(sp[k])))
    del W
    torch.cuda.empty_cache()
    print(f'  … {net+1}/{NETS}', flush=True)

goe = []
for _ in range(64):
    A = rng.standard_normal((W_, W_))
    goe.append(_r(_live_bulk(np.linalg.eigvalsh((A + A.T) / 2))))
poi = np.mean([_r(rng.random(180)) for _ in range(200)])


def ln(v):
    v = np.array(v, float)
    v = v[~np.isnan(v)]
    return f'{v.mean():.4f}±{v.std(ddof=1)/np.sqrt(len(v)):.4f}'


print(f'\n{NETS} мереж. ⟨r⟩ — лише ЖИВІ рівні, далі об\'єм {int(100*BULK)}%\n')
print('✅ЕТАЛОНИ (теорія): Пуассон 0.3863 · GOE 0.5359 · GUE 0.6027 · '
      'частокіл 1.0000')
print(f'✅ЗМІРЯНО: Пуассон {poi:.4f} · GOE {ln(goe)}')
print(f'🔑 ВОРОТА ПРИЛАДУ: глибина 1 БЕЗ воріт = {ln(RP[1])} '
      f'← мусить бути ≈0.536\n')
print(f'{"глибина":>8s} {"✅БЕЗ воріт":>18s} {"🟢З воротами":>18s} '
      f'{"живих рівнів із 256":>21s}')
print('-' * 70)
for k in KS:
    print(f'{k:8d} {ln(RP[k]):>18s} {ln(RG[k]):>18s} '
          f'{np.median(NL[k]):20.0f}')
print('\n📊 БАЛ 1.954e-07 — вимір КЛАСИФІКАЦІЙНИЙ, важеля не дає')
