"""RAO–BLACKWELL: ЗАМІНИТИ ХВІСТ МОНТЕ-КАРЛО АНАЛІТИКОЮ. (С87, ціль ЮЗЕРА)

## ЗВІДКИ — ЧОМУ ЦЕ АТАКУЄ ПРАВИЛЬНУ ВЕЛИЧИНУ
📊РОЗКЛАД НАШОГО БАЛУ: `V/N` = **82.5%**, `b²` = 17.5%.
Уся сесія полювала на ЗМІЩЕННЯ (`σ`, `γ`, форма), тобто на 17.5%. А домінує
ДИСПЕРСІЯ. Її ще жодного разу не атакували структурно — лише збільшенням `N`
(арифметично беззбитково вище 10% бюджету) і контрольними змінними
(ρ²=0.111 при стелі 0.926).

## 📐 ТЕОРЕМА (не евристика)
```
E[f(X)] = E[ E[f(X) | T] ]        Var( E[f|T] ) ≤ Var(f)   ЗАВЖДИ
```
Замість усереднювати `a₃₁` по `N` семплах — узяти достатню статистику
`T = (m̂_l, Ĉ_l)` на проміжному шарі `l` і дорахувати `E[a₃₁ | T]` АНАЛІТИЧНО.
🔑 ЧОМУ ДИСПЕРСІЯ ПАДАЄ: `Ĉ_l` усереднює по `n²=65536` добутках замість `n=256`
значень ⟹ її відносний шум менший за шум прямого середнього виходу.
🔑 І НАКОПИЧЕННЯ КОРОТШАЄ: старт із шару `l` дає похибку замикання
`0.35%·(1−0.976^(31−l))/0.024` замість повних 14.8%
([[finding-sigma-error-is-a-geometric-sum-of-offdiagonal-step-error]]).

## ⛔ЧОМУ ЦЕ НЕ ПОВТОР МЕРТВОГО ГІБРИДА
`hybrid87` (×0.26) підставляв ВИБІРКОВІ `σ̂,γ̂` у рекурсію НА КОЖНОМУ кроці —
змішане наближення, яке програє узгодженому (закон С87). Тут інакше: повна
`Ĉ` береться ОДИН раз, далі йде ЧИСТА аналітика без домішок.
⚠️`finding-cost-axis-midstart-dead` закрив «старт МК із середини мережі» — це
ІНШИЙ об'єкт (там МК стартувала з середини; тут МК йде з початку, а з середини
стартує АНАЛІТИКА).

## 🎯 ЩО МІРЯЄТЬСЯ
Для кожного `l`: `N` семплів проходять шари 0..l, дають `m̂_l, Ĉ_l`; далі
детермінований ланцюг до 31. Порівнюємо MSE з прямою МК на ТИХ САМИХ семплах.
```
l = 31  ⟹  тотожно пряма МК        ⛔ПЛОМБА
l = 0   ⟹  чиста аналітика (0.94%)  ⛔ПЛОМБА
десь між ними може бути оптимум — це і є питання
```

## ⛔ПЛОМБИ
 P1 `l=31` мусить дати РІВНО MSE прямої МК (тотожність).
 P2 `l=0` мусить дати 0.9416% (чисте замикання).
 P3 ШУМ ІСТИНИ окремо; істина — незалежна МК на 2²⁰.
 P4 ⛔МАСШТАБУВАННЯ: при `N×4` дисперсійна частина мусить впасти вчетверо,
    зміщення — встояти. Це відрізняє шум від систематики.
 P5 α-калібровка LOO по мережах на КОЖНІЙ сходинці (закон С87).

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) крива MSE(l) матиме МІНІМУМ усередині (не на кінцях). Шанс **50%** —
     зліва тисне зміщення замикання, справа шум МК.
 (б) мінімум буде при `l` = 20–28 (близько до кінця, бо зміщення росте швидко).
     Шанс 45%.
 (в) виграш проти прямої МК ≥×1.3. Шанс **30%**.
 (г) виграш проти прямої МК ≥×2. Шанс 12% — це вже був би важіль.
 ⚠️Головний ризик: `Ĉ` з 6000 семплів має ранг ≤6000 і шум ~1.3% в елементах;
   квадратична форма усереднює, але накопичення 0.976^k його зберігає.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/rblack87.py [cuda:N] [NETS] [NSAMP]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
NSAMP = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
NTRUE = 1 << (int(sys.argv[4]) if len(sys.argv) > 4 else 20)
BATCH = 1 << 13
L_, W_ = 32, 256
LS = (0, 4, 8, 12, 16, 20, 24, 27, 29, 30, 31)
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)
SUB97 = 1.873e-06


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def tail_chain(W, m, C, start):
    """Детермінований ланцюг від шару `start` до кінця; вхід `m, C` довільні."""
    for i in range(start, len(W)):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        t = m / s
        P, f = Phi(t), phi(t)
        m_a = m * P + s * f
        if i == len(W) - 1:
            return m_a
        Ca = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
        q2 = (m * m + s * s) * P + (m * s) * f
        Ca = Ca + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                             - torch.diagonal(Ca))
        m = m_a @ W[i + 1]
        C = W[i + 1].T @ (Ca @ W[i + 1])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · N={NSAMP} робочих семплів · істина 2^20 · {DEV}\n',
      flush=True)

TAGS = [f'l={l:2d}' for l in LS] + ['⛔пряма МК']
MS = {t: np.zeros(len(FS)) for t in TAGS}
PT = {t: np.zeros(len(FS)) for t in TAGS}
PP = {t: np.zeros(len(FS)) for t in TAGS}
TT = np.zeros(len(FS))
NRM2 = np.zeros(len(FS))
PRED = {t: np.zeros((len(FS), W_)) for t in TAGS}
TRUE_ = np.zeros((len(FS), W_))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        # ── ІСТИНА: незалежна МК на 2^20 (інший сід) ──────────────────────
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(999000 + n)
        d = 0
        while d < NTRUE:
            b = min(BATCH, NTRUE - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                a = torch.relu(a @ W32[q])
            acc += a.sum(0, dtype=torch.float64)
            d += b
        tru = acc / NTRUE
        NRM2[n] = float((tru ** 2).mean())
        TT[n] = float((tru * tru).sum())

        # ── РОБОЧА вибірка: N семплів з АНТИТЕТИКОЮ, як у сабміті ─────────
        gw = torch.Generator(device=DEV)
        gw.manual_seed(555000 + n)
        z = torch.randn(NSAMP // 2, W_, device=DEV, generator=gw)
        x = torch.cat([z, -z], 0)
        SM = [None] * (L_ + 1)
        SC = [None] * (L_ + 1)
        a = x
        SM[0] = a.double().mean(0)
        SC[0] = (a.double().T @ a.double()) / NSAMP - torch.outer(SM[0], SM[0])
        for q in range(L_):
            h = a @ W32[q]
            hd = h.double()
            SM[q + 1] = hd.mean(0)
            SC[q + 1] = (hd.T @ hd) / NSAMP - torch.outer(SM[q + 1], SM[q + 1])
            a = torch.relu(h)
        mc = a.double().mean(0)          # пряма МК-оцінка виходу

        runs = {}
        for l in LS:
            if l == 31:
                runs[f'l={l:2d}'] = mc   # ⛔P1 тотожність
            elif l == 0:
                C0 = W[0].T @ W[0]
                m0 = torch.zeros(W_, dtype=torch.float64, device=DEV)
                runs[f'l={l:2d}'] = tail_chain(W, m0, C0, 0)
            else:
                # 🚨 ІНДЕКС: `SC[q+1] = Cov(h_q)` (преактивація ПІСЛЯ q-го шару),
                #    а `tail_chain(start=l)` чекає `Cov(h_l)` ⟹ треба `SC[l+1]`.
                #    Зсув на один шар давав 16–100% похибки (спіймано пломбою).
                Cl = 0.5 * (SC[l + 1] + SC[l + 1].T)
                runs[f'l={l:2d}'] = tail_chain(W, SM[l + 1].clone(), Cl, l)
        runs['⛔пряма МК'] = mc

        TRUE_[n] = tru.cpu().numpy()
        for t, pr in runs.items():
            PRED[t][n] = pr.cpu().numpy()
            MS[t][n] = float(((pr - tru) ** 2).mean())
            PT[t][n] = float((pr * tru).sum())
            PP[t][n] = float((pr * pr).sum())
    print(f'  net{n} · l=0 {math.sqrt(MS["l= 0"][n] / NRM2[n]):.4%} · '
          f'l=24 {math.sqrt(MS["l=24"][n] / NRM2[n]):.4%} · '
          f'МК {math.sqrt(MS["⛔пряма МК"][n] / NRM2[n]):.4%}', flush=True)


def alpha_mse(t):
    out = np.zeros(len(FS))
    a_n = PT[t] / np.maximum(PP[t], 1e-30)
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        a = a_n[msk].mean() if len(FS) > 1 else a_n[i]
        out[i] = (a * a * PP[t][i] - 2.0 * a * PT[t][i] + TT[i]) / W_
    return out.mean()


print(f'\n{"═" * 72}\n⛔ПЛОМБИ')
print(f' P1 l=31 ≡ пряма МК: |Δ| = '
      f'{abs(MS["l=31"].mean() - MS["⛔пряма МК"].mean()):.3e}  '
      f'{"✅" if abs(MS["l=31"].mean() - MS["⛔пряма МК"].mean()) < 1e-15 else "⛔"}')
e0 = math.sqrt(MS['l= 0'].mean() / NRM2.mean())
print(f' P2 l=0 (чисте замикання) = {e0:.4%}  (очік. 0.9416%)  '
      f'{"✅" if abs(e0 - 0.009416) < 0.0010 else "⛔"}')

print(f'\n{"═" * 72}')
print(f'КРИВА: ДЕ ОБІРВАТИ МК І ПЕРЕЙТИ НА АНАЛІТИКУ (N={NSAMP})')
mk = MS['⛔пряма МК'].mean()
mka = alpha_mse('⛔пряма МК')
print(f'{"старт аналітики":>16s} {"віднос.":>9s} {"MSE":>11s} {"ПІСЛЯ α":>11s} '
      f'{"vs МК":>8s} {"vs sub97":>9s}')
for t in TAGS:
    m_ = MS[t].mean()
    a_ = alpha_mse(t)
    print(f'{t:>16s} {math.sqrt(m_ / NRM2.mean()):8.4%} {m_:11.3e} '
          f'{a_:11.3e} {"×%.3f" % (mka / a_):>8s} {"×%.3f" % (SUB97 / a_):>9s}')

# 🔑 БЛЕНД: похибки аналітики від шару `l` і прямої МК мають РІЗНІ джерела
#    (зміщення замикання проти шуму вибірки). Якщо вони не повністю
#    корельовані, оптимальна суміш б'є обидві. Вага — LOO ПО МЕРЕЖАХ.
print(f'\n{"═" * 72}')
print('🔑 ОПТИМАЛЬНИЙ БЛЕНД `w·аналітика(l) + (1−w)·МК`, вага LOO по мережах')
print(f'{"l":>6s} {"w опт":>8s} {"MSE бленду":>12s} {"vs МК":>8s} {"vs sub97":>9s}')
for t in TAGS[:-1]:
    dA = PRED[t] - TRUE_       # похибка аналітики
    dM = PRED['⛔пряма МК'] - TRUE_
    num = (dM * (dM - dA)).sum(1)
    den = ((dM - dA) ** 2).sum(1)
    w_n = num / np.maximum(den, 1e-30)
    out = np.zeros(len(FS))
    for i in range(len(FS)):
        msk = np.arange(len(FS)) != i
        w = float(np.clip(w_n[msk].mean(), -0.5, 1.5)) if len(FS) > 1 else float(w_n[i])
        r = w * dA[i] + (1.0 - w) * dM[i]
        out[i] = float((r * r).mean())
    mb = out.mean()
    print(f'{t:>6s} {w_n.mean():8.4f} {mb:12.3e} {"×%.3f" % (mk / mb):>8s} '
          f'{"×%.3f" % (SUB97 / mb):>9s}')

best = min((alpha_mse(t), t) for t in TAGS if t != '⛔пряма МК')
print(f'\n📣 ВИРОК: найкращий обрив — {best[1]}, MSE після α = {best[0]:.3e}')
print(f'   проти прямої МК ×{mka / best[0]:.3f} · проти sub97 ×{SUB97 / best[0]:.3f}')
if mka / best[0] >= 1.3:
    print('   🟢 RAO-BLACKWELL ПРАЦЮЄ — аналітичний хвіст б\'є семплінг')
else:
    print('   ⛔ обрив не окупається: зміщення замикання з\'їдає виграш дисперсії')
