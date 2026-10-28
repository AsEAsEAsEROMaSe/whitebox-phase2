"""ЧИ ІСНУЄ ФУНКЦІЯ `b_q = f(ознаки нейрона)` — ГЕЙТ ПЕРЕД ГЕНЕРАЦІЄЮ МЕРЕЖ (С84)

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

📣 ПЕРЕДБАЧЕНО: якщо `R²_oos` для 14 ознак вийде ≥0.7 — структура є, генерація
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
         'γ₁', 'γ₂', 'ŝ²−s²', 't²', 'φ²', 'tΦ']
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
        TT, PH, PHI, SS = [], [], [], []
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
            fe = torch.stack([one, t, Ph, ph, t*ph, s, s*s, q_cl[i],
                              g1a[i], g2a[i], sh2a[i]-s*s, t*t, ph*ph, t*Ph], 1)
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
for k in [1, 2, 3, 5, 8, 11, NFE]:
    print(f'{k:9d} {NAMES[k-1]:>10s} {r2_loo(X, Y, WT, GRP, k):9.4f} '
          f'{r2_loo(X, Y, WT, GRP, k, roll=True):9.4f}')

r1 = r2_loo(X, Y, WT, GRP, 1)
rf = r2_loo(X, Y, WT, GRP, NFE)
rr = r2_loo(X, Y, WT, GRP, NFE, roll=True)
print('\n🎯 ВИРОК')
print(f'   сама стала: R² = {r1:.4f} · усі {NFE} ознак: R² = {rf:.4f} · '
      f'⛔ROLL = {rr:.4f}')
print(f'   ПРИРІСТ ФОРМИ понад сталу й ROLL: {rf - max(r1, rr):+.4f}')
if rf >= 0.7:
    print('   ⟹ 🟢 СТРУКТУРА Є ⟹ генерація ~2700 мереж виправдана')
    print(f'      (закон 11.76%·k/N ⟹ при N=2689 окупається ~200 параметрів)')
elif rf >= 0.3:
    print('   ⟹ ⚠️ структура ЧАСТКОВА — рахувати, яку частку стелі це дає')
else:
    print('   ⟹ ⛔ ФУНКЦІЇ НЕМА: `b_q` НЕ передбачається з локальних ознак')
    print('      ⟹ ГЕНЕРАЦІЯ МЕРЕЖ НЕ ДОПОМОЖЕ — проблема не в даних.')
    print('      Причина: зміщення `q` — НАКОПИЧЕННЯ (залежить від усієї')
    print('      історії ланцюга), а не властивість стану нейрона.')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
