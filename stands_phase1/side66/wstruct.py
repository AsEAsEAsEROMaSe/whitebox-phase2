"""ОЗНАКИ З САМИХ ВАГ — ТЕ, ЧОГО ЗАМИКАННЯ НЕ БАЧИТЬ (С84, теза ЮЗЕРА «все в вагах»)

## 🚨 ЛОГІЧНА ДІРА, ЯКУ ВИЯВИЛА ТЕЗА
⛔`side66/qstruct.py` фітував `b_q` з 14 ознак — `t,Φ,φ,s,q_замик,γ` — тобто з
ВИХОДІВ САМОГО ЗАМИКАННЯ. Але зміщення замикання ЗА ВИЗНАЧЕННЯМ є тим, чого
замикання НЕ ВЛОВИЛО: якби воно було функцією власних виходів, замикання могло
б себе виправити. ⟹ 🎓 **`R²`=0.26 — НЕ ВИПАДКОВІСТЬ, А ПОЧТИ ТАВТОЛОГІЧНА
МЕЖА: НЕ МОЖНА ФІТУВАТИ ПОХИБКУ МЕТОДУ ФУНКЦІЄЮ ВИХОДІВ ТОГО САМОГО МЕТОДУ.**

## 🔑 ЩО ЗАМИКАННЯ ВІДКИДАЄ
З матриці `W` (256×256 на шар) воно бере ЛИШЕ квадратичну форму `WᵀCW`.
Індивідуальна структура стовпців і вищі моменти — відкидаються за побудовою.
А саме там і живе те, чого бракує:
    `κ₄(h_i) = Σ_{jklm} W_ji W_ki W_li W_mi · κ₄(a_j,a_k,a_l,a_m)`
діагональний член ⟹ **`Σ_j W_ji⁴`** прямо керує негаусовістю нейрона `i`, і
замикання його НІДЕ не використовує.
🔑🔑 ГОЛОВНЕ: такі ознаки **ДЕТЕРМІНОВАНІ** (з ваг), а не вибіркові ⟹ вони НЕ
дублюють `λ_Q` і не з'їдають КЗ. Це виводить із замкненого кола
[[finding-capacity-law-and-why-q-has-no-pattern]] §4.

## ЩО ДОДАНО ДО БАЗИСУ (8 нових ознак, УСІ з ваг стовпця `i` шару `ℓ`)
```
w4   = Σ_j W_ji⁴ / (Σ_j W_ji²)²      куртозис стовпця (керує κ₄ напряму)
ipr  = (Σ_j W_ji²)² / Σ_j W_ji⁴      ефективне число внесків у суму
w3   = Σ_j W_ji³ / (Σ_j W_ji²)^1.5   скошеність (керує κ₃)
wmax = max_j |W_ji| / √(Σ_j W_ji²)   найбільший одиничний внесок
cw4  = Σ_j C_jj² W_ji⁴ / (Σ_j C_jj W_ji²)²   те саме, зважене дисперсіями
n2   = Σ_j W_ji²                      сама норма стовпця
row4 = Σ_k W_ik⁴ / (Σ_k W_ik²)²      куртозис РЯДКА (вихідний бік)
cor  = Σ_j W_ji² m_j / (s·Σ_j W_ji²)  вирівняність стовпця з середнім входу
```
📌 Порівнюємо ЧЕСНО: спершу 14 старих (замиканні), потім +8 нових ⟹ приріст
`R²` від ваг видно окремо.

## СТАРА ШАПКА
ЧИ ІСНУЄ ФУНКЦІЯ `b_q = f(ознаки нейрона)` — ГЕЙТ ПЕРЕД ГЕНЕРАЦІЄЮ МЕРЕЖ (С84)

## ЗВІДКИ — ПИТАННЯ ЮЗЕРА ПРО УНІВЕРСАЛЬНУ АПРОКСИМАЦІЮ
«Наскільки детальну криву може відтворити наша модель і скільки патернів?»
🔑 ЗАКОН ПЕРЕОБТЯЖЕННЯ, зміряний на двох незалежних прогонах `qfeat` (24 і 48
мереж, k=1..9): **штраф ≈ 11.76% · k/N** (коефіцієнт тримається 9.6–15.4%).
⟹ при `N=48` окупається лише **3.6** параметра — ось чому ВСІ фіти сесії
вмирали однаково. Обмежує не виразність, а ЄМНІСТЬ ДАНИХ.
⟹ при `N=2689` окупалося б ~200 параметрів, а це вже серйозна модель.
💰 Мережі можна ГЕНЕРУВАТИ: істина ≈ 8.4e12 FLOP ⟹ ~1.4 с на P104 ⟹ 2689
мереж ≈ 16 хв на чотирьох GPU. Ваги банку `pool3` НЕ відтворюються з `seed`
(перевірено 4 рецепти, max|Δ|=0.68), але свої мережі ми робимо самі.

## 🚨 ГЕЙТ ПЕРЕД ТИМ, ЯК ЦЕ РОБИТИ
Більше мереж допоможе ЛИШЕ якщо функція `b_q = f(ознаки)` існує. С70 міряла
`R²=0.911` для зміщення **`m`** ([[finding-closure-bias-is-a-universal-function-of-8-neuron-features]]),
для **`q`** — НІКОЛИ. А С84 вже показала, що `m` і `q` поводяться по-різному:
стала складова дає 0.923 для `m` і НУЛЬ для `q`
([[finding-the-q-channel-is-poisoned-by-closure-bias]]).
⟹ переносити `R²` з `m` на `q` НЕ МОЖНА — це та сама помилка, що зі сталою.

## ЩО МІРЯЄМО
Ціль: `b_q = E[a²]_істина − q_замик` понейронно, шари 1..8, 48 мереж.
«Істина» — середнє по **32** жеребах (оракул; шум оракула друкується).
Ознаки (усі ДОСТУПНІ в рантаймі): `1, t, Φ, φ, tφ, s, s², q_замик, γ₁, γ₂,
ŝ²−s², t², φ², tΦ` — 14 штук, як у С70 плюс наші кумулянти.
Фіт: зважений МНК, ЦІЛЬ БЕЗРОЗМІРНА `b_q/s²` (інакше домінують великі `s`),
**LOO ПО МЕРЕЖАХ**, спільні коефіцієнти для всіх мереж.
📊 Головні числа: `R²_oos` для 1 / 8 / 14 ознак і ЯКА частка — сама стала.

⛔ ПЛОМБИ:
 (а) «ознака = ціль» мусить дати `R² = 1.000` (перевірка машинерії фіту);
 (б) ⛔ROLL-контроль: ті самі ознаки, зсунуті по нейронах, — `R²` мусить
     впасти до рівня самої константи (роль `roll` НЕ руйнує стовпець одиниць!);
 (в) шум оракула проти дисперсії `b_q` — щоб не фітувати шум.

📣 ПЕРЕДБАЧЕНО (нове): старі 14 дають 0.2644. Якщо ознаки з ВАГ додають
   ≥+0.15 — теза «все в вагах» підтверджена практично, і відкривається шлях,
   якого не було (детермінована поправка, що не дублює канал). Якщо приріст
   ≤+0.05 — інформація про похибку не лежить у ЛОКАЛЬНІЙ структурі стовпця, і
   тоді лишається тільки нелокальне (уся історія ланцюга).
   Ставлю обережно: `w4` для гаусових ваг ≈ 3/n з флуктуацією ~1/√n, тобто
   сигнал слабкий за побудовою; але `ipr` і `cw4` можуть нести більше.

## СТАРЕ ПЕРЕДБАЧЕННЯ: якщо `R²_oos` для 14 ознак вийде ≥0.7 — структура є, генерація
   мереж виправдана, і можна цілитись у помітну частку стелі ×1.0896. Якщо
   ≤0.3 — функції нема, `b_q` понейронно НЕ передбачається з локальних ознак,
   і тоді генерація мереж НЕ ДОПОМОЖЕ: проблема не в даних, а в тому, що
   зміщення `q` — властивість НАКОПИЧЕННЯ, тобто залежить від усієї історії
   ланцюга, а не від стану нейрона. Ставлю на друге (~0.3), бо С84 вже
   показала, що зміщення `q` — це накопичення, а не локальна форма.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qstruct.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, PILOT_PAIRS)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
L_, NORA = 32, 32
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = sorted(glob.glob('/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
FS = [f for f in FS if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж × {NORA} жеребів (оракул), {DEV}\n', flush=True)

NAMES = ['1', 't', 'Φ', 'φ', 'tφ', 's', 's²', 'q_cl',
         'γ₁', 'γ₂', 'ŝ²−s²', 't²', 'φ²', 'tΦ',
         'w4', 'ipr', 'w3', 'wmax', 'cw4', 'n2', 'row4', 'cor']
NOLD = 14
NFE = len(NAMES)
X, Y, WT, GRP = [], [], [], []
ORN = []

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        s_cl, m_cl, q_cl, P_cl, _ = closure91(wc, KSIG, DEV, kdeep=len(wc))
        # опори t, φ, Φ із замикання
        TT, PH, PHI, SS, WF = [], [], [], [], []
        m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=DEV)
        C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=DEV)
        for i in range(len(wc)):
            w = wc[i].double()
            cp = w.T @ (C @ w)
            s = cp.diagonal().clamp_min(1e-20).sqrt()
            mp = m @ w
            t = mp / s
            ph, Ph = _phi(t), _Phi(t)
            if 1 <= i <= KSIG:
                TT.append(t); PH.append(ph); PHI.append(Ph); SS.append(s)
                # 🔑 ОЗНАКИ З САМИХ ВАГ — того, чого замикання не бачить
                w2 = (w * w).sum(0)
                w4 = (w ** 4).sum(0)
                WF.append(torch.stack([
                    w4 / (w2 * w2).clamp_min(1e-300),
                    (w2 * w2) / w4.clamp_min(1e-300),
                    (w ** 3).sum(0) / w2.pow(1.5).clamp_min(1e-300),
                    w.abs().max(0).values / w2.sqrt().clamp_min(1e-300),
                    ((C.diagonal() ** 2)[:, None] * w ** 4).sum(0)
                    / (((C.diagonal())[:, None] * w * w).sum(0) ** 2).clamp_min(1e-300),
                    w2,
                    (w ** 4).sum(1).mean() * torch.ones_like(w2)
                    / ((w * w).sum(1).mean() ** 2).clamp_min(1e-300),
                    (w2 * 0 + (m[:, None] * w * w).sum(0)
                     / (s * w2).clamp_min(1e-300)),
                ], 1))
            m = mp * Ph + s * ph
            if i < len(wc) - 1:
                d2 = (mp*mp + s*s)*Ph + mp*s*ph
                C = cp*torch.outer(Ph, Ph) + (0.5*cp*cp)*torch.outer(ph/s, ph/s)
                C = C + torch.diag((d2 - m*m) - C.diagonal())
        # ── ОРАКУЛ: істинні E[a²], ŝ², γ₁, γ₂ по 32 жеребах
        acc = [torch.zeros_like(q_cl[k]) for k in range(KSIG)]
        acc2 = [torch.zeros_like(q_cl[k]) for k in range(KSIG)]
        sh2a = [torch.zeros_like(q_cl[k]) for k in range(KSIG)]
        g1a = [torch.zeros_like(q_cl[k]) for k in range(KSIG)]
        g2a = [torch.zeros_like(q_cl[k]) for k in range(KSIG)]
        for d in range(NORA):
            rng = np.random.default_rng(7_000_000 + 733*n + d)
            rng.standard_normal((PILOT_PAIRS, W_))
            z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                             device=DEV)
            cov = (z.T @ z) / float(npair)
            ev, V = torch.linalg.eigh(cov.double())
            wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
            y = z @ (wh @ wc[0])
            x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
            den = float(2*npair)
            for i in range(1, min(KSIG+1, len(wc))):
                p = x @ wc[i]
                pd = p.double(); p2 = pd*pd
                mh = pd.sum(0)/den
                e2 = p2.sum(0)/den
                sh2 = (e2 - mh*mh).clamp_min(1e-30)
                e3 = (p2*pd).sum(0)/den
                e4 = (p2*p2).sum(0)/den
                mh2 = mh*mh
                k3 = e3 - 3.0*mh*e2 + 2.0*mh2*mh
                k4 = e4 - 4.0*mh*e3 + 6.0*mh2*e2 - 3.0*mh2*mh2 - 3.0*sh2*sh2
                x = torch.relu(p)
                xq = x.double().pow(2).sum(0)/den
                acc[i-1] += xq/NORA
                acc2[i-1] += xq*xq/NORA
                sh2a[i-1] += sh2/NORA
                g1a[i-1] += (k3/(sh2*sh2.sqrt()))/NORA
                g2a[i-1] += (k4/(sh2*sh2))/NORA
        for i in range(KSIG):
            s = SS[i]; t = TT[i]; ph = PH[i]; Ph = PHI[i]
            b = acc[i] - q_cl[i]                       # зміщення q
            one = torch.ones_like(s)
            fe = torch.cat([torch.stack(
                [one, t, Ph, ph, t*ph, s, s*s, q_cl[i],
                 g1a[i], g2a[i], sh2a[i]-s*s, t*t, ph*ph, t*Ph], 1),
                WF[i]], 1)
            X.append(fe.cpu().numpy())
            Y.append((b/(s*s)).cpu().numpy())          # БЕЗРОЗМІРНА ціль
            WT.append((s*s).cpu().numpy() ** 2)        # вага ∝ внеску в δ
            GRP.append(np.full(len(s), n))
            ORN.append(float(((acc2[i]-acc[i]**2).clamp_min(0)/NORA).sum().sqrt()
                              / b.norm().clamp_min(1e-300)))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

X = np.concatenate(X); Y = np.concatenate(Y)
WT = np.concatenate(WT); GRP = np.concatenate(GRP)
WT = WT / WT.mean()
np.savez('/mnt/win_d/llm/White-Box/arena62/logs/qstruct.npz',
         X=X, Y=Y, WT=WT, GRP=GRP)
print(f'\n⛔ПЛОМБА (в) шум оракула / ‖b_q‖: сер {np.mean(ORN):.4f} '
      f'{"✅" if np.mean(ORN) < 0.2 else "🚨 ФІТУЄМО ЧАСТКОВО ШУМ"}')
print(f'   рядків {len(Y)}, мереж {GRP.max()+1}\n')


def r2_loo(Xa, Ya, Wa, G, k, roll=False):
    """LOO ПО МЕРЕЖАХ, перші `k` ознак. `roll` — зсув ознак по нейронах."""
    Xk = Xa[:, :k].copy()
    if roll:
        Xk[:, 1:] = np.roll(Xk[:, 1:], 137, axis=0)
    num = den = 0.0
    for g in range(int(G.max()) + 1):
        tr, te = G != g, G == g
        A = Xk[tr] * Wa[tr, None]
        M = Xk[tr].T @ A + 1e-10 * np.eye(k)
        c = np.linalg.solve(M, A.T @ Ya[tr])
        r = Ya[te] - Xk[te] @ c
        num += float((Wa[te] * r * r).sum())
        den += float((Wa[te] * (Ya[te] - np.average(Ya[tr], weights=Wa[tr]))**2).sum())
    return 1.0 - num / max(den, 1e-300)


print('⛔ПЛОМБА (а) «ознака = ціль»: R² = '
      f'{r2_loo(np.column_stack([np.ones(len(Y)), Y]), Y, WT, GRP, 2):.4f} '
      '(мусить бути 1.0000)')
print(f'\n{"k ознак":>9s} {"остання":>10s} {"R²_oos":>9s} {"⛔ROLL":>9s}')
print('─' * 42)
for k in [1, 8, 11, NOLD, NOLD+2, NOLD+5, NFE]:
    print(f'{k:9d} {NAMES[k-1]:>10s} {r2_loo(X, Y, WT, GRP, k):9.4f} '
          f'{r2_loo(X, Y, WT, GRP, k, roll=True):9.4f}')

ro = r2_loo(X, Y, WT, GRP, NOLD)
rf = r2_loo(X, Y, WT, GRP, NFE)
rr = r2_loo(X, Y, WT, GRP, NFE, roll=True)
# лише ваги (стала + 8 вагових), без замиканних
Xw = np.column_stack([X[:, 0], X[:, NOLD:]])
rw = r2_loo(Xw, Y, WT, GRP, Xw.shape[1])
print('\n🎯 ВИРОК — ЧИ ДОДАЮТЬ ОЗНАКИ З ВАГ')
print(f'   14 ЗАМИКАННИХ (як було):        R² = {ro:.4f}')
print(f'   +8 З ВАГ (усі {NFE}):            R² = {rf:.4f}   ПРИРІСТ {rf-ro:+.4f}')
print(f'   ЛИШЕ ваги (стала + 8):          R² = {rw:.4f}')
print(f'   ⛔ROLL усіх {NFE}:                R² = {rr:.4f}')
if rf - ro >= 0.15:
    print('   ⟹ 🟢 ТЕЗА «ВСЕ В ВАГАХ» ПІДТВЕРДЖЕНА: ваги несуть те, чого')
    print('      замикання не бачить, і ознака ДЕТЕРМІНОВАНА (не дублює λ_Q)')
elif rf - ro >= 0.05:
    print('   ⟹ ⚠️ приріст помірний — рахувати, чи окупається за законом ємності')
else:
    print('   ⟹ ⛔ ЛОКАЛЬНА структура стовпця НЕ несе похибки замикання')
    print('      ⟹ лишається лише НЕЛОКАЛЬНЕ: уся історія ланцюга')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
