"""ЧИ НАС ЗУПИНЯЮТЬ ВІД'ЄМНІ ВАГИ: катастрофічне скорочення (ідея юзера, С78).

Після ПЕРШОГО ReLU всі активації невід'ємні. Тому на кожному наступному шарі
    p_j = Σ_{w>0} w·x  −  Σ_{w<0} |w|·x        (обидві суми ≥ 0)
— це РІЗНИЦЯ ДВОХ ВЕЛИКИХ ДОДАТНИХ СУМ. А ворота скрізь 0.500, тобто `p`
центроване біля нуля: складові майже точно гасяться, і вся відповідь живе в
залишку. Це катастрофічне скорочення, вроджене ЗАДАЧІ, а не нашому методу.

Дві частини:

  A. МАСШТАБ СКОРОЧЕННЯ по шарах:  C_ℓ = (E[p⁺] + E[|p⁻|]) / std(p).
     Це у скільки разів складові більші за розкид результату — тобто скільки
     значущих цифр з'їдається на кожному шарі.

  B. КОНТРОЛЬНИЙ ЕКСПЕРИМЕНТ (чистий тест гіпотези): частку `α` від'ємних ваг
     примусово перевертаємо в додатні і дивимось, як падає похибка вибіркової
     оцінки. У межі α→0 всі `p>0`, ReLU нічого не ріже, мережа ЛІНІЙНА і
     відповідь точна за нуль флопів. Якщо крива різко валиться — гіпотеза
     юзера підтверджена кількісно; якщо полога — скорочення НЕ головне.

🚨 Ваги перевертаються ПОШАРОВО з фіксованим жеребом, шар 0 НЕ чіпаємо (там
   вхід `z` знакозмінний, асиметрії ще нема — саме тому γ₃(шар 0)=0 рівно).

📣 ПЕРЕДБАЧЕНО: C_ℓ ≈ 3.6 з арифметики He. Для `w ~ N(0, 2/n)` і `x = relu`:
   E[p⁺] = (n/2)·E[w|w>0]·E[x] = √(n/2)·√(2/π)·0.3989σ ≈ 3.60σ, а std(p) = σ.
   Тобто на КОЖНОМУ шарі гаситься ~3.6× — це ~0.56 десяткової цифри за шар,
   за 32 шари ≈ 18 цифр, якби воно накопичувалось. Воно НЕ накопичується
   (інакше fp64 не вистачило б), бо шум переноситься разом із сигналом.
   Тому ставлю на те, що крива по α буде ПОЛОГОЮ, а справжня межа — не
   скорочення, а те, що `E[relu]` залежить від ФОРМИ поля (див.
   [[finding-edgeworth-cumulants-are-the-live-branch]]).
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
NPAIR = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
DRAWS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
W_, D, CH = 256, 32, 1 << 14
ALPHAS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0)   # частка ЗБЕРЕЖЕНИХ від'ємних ваг

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _flip(W, alpha, seed):
    """Лишити частку `alpha` від'ємних ваг, решту перевернути в додатні.

    🚨 ПОТІМ ОБОВ'ЯЗКОВО НОРМУВАТИ. При α→0 усі доданки додатні і сума росте у
    ~18× на шар (256·√(2/256)·√(2/π)), тобто 18³²≈10⁴⁰ ⟹ fp32 переповнюється і
    вимір дає NaN. Це сам по собі результат: ВІД'ЄМНІ ВАГИ ТРИМАЮТЬ МЕРЕЖУ
    СКІНЧЕННОЮ. Нормування безпечне — задача позитивно однорідна, тож множення
    W_L на константу не змінює ВІДНОСНОЇ похибки. Нормуємо ВСІ α однаково
    (включно з α=1), щоб порівняння лишалось чесним.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    out = [W[0]]
    for L in range(1, D):
        w = W[L]
        if alpha >= 1.0:
            out.append(w.clone())
        else:
            neg = w < 0
            keep = torch.rand(w.shape, device=DEV, generator=g) < alpha
            out.append(torch.where(neg & ~keep, -w, w))
    # пошарове нормування std(p)=1 на малій вибірці
    zz = torch.randn(512, W_, device=DEV,
                     generator=torch.Generator(device=DEV).manual_seed(4242))
    x = torch.cat([zz, -zz], 0)
    for L in range(D):
        p = x @ out[L]
        sd = float(p.double().std().clamp_min(1e-30))
        out[L] = out[L] / sd
        x = torch.relu(x @ out[L])
    return out


def _stat(W, seed, npairs, whiten=False, split=False):
    """E[relu] останнього шару; за split=True — ще й p⁺, |p⁻|, std(p) по шарах."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
    SP = torch.zeros(D, 4, W_, dtype=torch.float64, device=DEV) if split else None
    Wp = [w.clamp_min(0.0) for w in W] if split else None
    Wn = [w.clamp_max(0.0) for w in W] if split else None
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        if whiten:
            cov = (z.T @ z) / float(b)
            ev, U = torch.linalg.eigh(cov.double())
            z = (z.double() @ (U * ev.clamp_min(1e-8).rsqrt()) @ U.T).float()
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            if split:
                SP[L][0] += (x @ Wp[L]).double().sum(0)
                SP[L][1] += (-(x @ Wn[L])).double().sum(0)
                SP[L][2] += p.double().sum(0)
                SP[L][3] += (p.double() ** 2).sum(0)
            x = torch.relu(p)
        acc += x.double().sum(0)
        done += b
    N = float(2 * npairs)
    return acc / N, (SP / N if split else None)


# ── A. масштаб скорочення по шарах
print('A. МАСШТАБ СКОРОЧЕННЯ ПО ШАРАХ\n')
CC = np.zeros((NETS, D))
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        _, SP = _stat(W, 91_000 + 13 * n, 1 << (LOGN - 3), split=True)
        for L in range(D):
            pp, pn, m1, m2 = SP[L]
            sd = (m2 - m1 * m1).clamp_min(1e-30).sqrt()
            CC[n, L] = float(((pp + pn) / sd).mean())
    del W
    torch.cuda.empty_cache()
Cm = CC.mean(0)
print(f'{"шар":>4s} {"C = (E[p⁺]+E[|p⁻|])/std(p)":>28s} {"цифр з\'їдено":>14s}')
print('-' * 50)
for L in (0, 1, 2, 4, 8, 16, 24, 31):
    print(f'{L:4d} {Cm[L]:28.3f} {np.log10(max(Cm[L],1)):14.2f}')
print(f'\nсереднє по шарах 1..31: {Cm[1:].mean():.3f}  '
      f'(передбачено з арифметики He: 3.60)\n')

# ── B. контрольний експеримент по α
print('B. КОНТРОЛЬНИЙ ЕКСПЕРИМЕНТ: частка ЗБЕРЕЖЕНИХ від\'ємних ваг\n', flush=True)
res = np.zeros((len(ALPHAS), 3))          # MSE, ‖істина‖², C
for n in range(NETS):
    W0 = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV)
          for k in range(D)]
    for ai, a in enumerate(ALPHAS):
        W = _flip(W0, a, 7717 + n)   # нормовано ⟹ порівнянно
        with torch.no_grad():
            tA, _ = _stat(W, 91_000 + 13 * n, 1 << LOGN)
            tB, _ = _stat(W, 40_000 + 29 * n, 1 << LOGN)
            for d in range(DRAWS):
                mh, _ = _stat(W, 55_000 + 7 * n + 1013 * d, NPAIR, whiten=True)
                res[ai, 0] += float(((mh - tA) * (mh - tB)).sum())
            res[ai, 1] += float((tA * tB).sum()) * DRAWS
            _, SP = _stat(W, 91_000 + 13 * n, 1 << (LOGN - 4), split=True)
            cc = []
            for L in range(1, D):
                pp, pn, m1, m2 = SP[L]
                sd = (m2 - m1 * m1).clamp_min(1e-30).sqrt()
                cc.append(float(((pp + pn) / sd).mean()))
            res[ai, 2] += float(np.mean(cc))
        del W
    del W0
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

print(f'\n{NETS} мереж × {DRAWS} жеребів (відношення СУМ)\n')
print(f'{"α (від\'ємних)":>14s} {"скорочення C":>14s} {"відн. похибка":>15s} '
      f'{"проти α=1":>11s}')
print('-' * 60)
base = np.sqrt(res[0, 0] / res[0, 1])
for ai, a in enumerate(ALPHAS):
    rel = np.sqrt(max(res[ai, 0], 0) / res[ai, 1])
    print(f'{a:14.2f} {res[ai,2]/NETS:14.3f} {rel:15.4e} '
          f'{base/max(rel,1e-30):10.2f}×')
print('\n🔑 Якщо похибка валиться РІЗКО зі зменшенням α — скорочення і є межа.')
print('   Якщо полого — межа в іншому (форма поля).')
print('📊 БАЛ: 1.954e-07 — вимір діагностичний')
