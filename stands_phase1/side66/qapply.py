"""ПОПРАВКА ДО `δ_Q` З ФІТУ ПРОТИ ОРАКУЛА — ЧИ ВИЖИВЕ В БАЛІ (С84)

## ЗВІДКИ
`side66/qstruct.py`: зміщення `q` як функція локальних ознак має `R²_oos`
лише **0.2644** (проти 0.911 для `m` у С70) ⟹ функції майже нема, і
генерація мереж НЕ допоможе. АЛЕ коефіцієнти фіту вийшли ДУЖЕ стабільними
(LOO-розкид 0.2–1.1% для головних), і домінує ОДНА ознака:
    `b_q ≈ 0.559·(ŝ² − s²) + 0.077·γ₁·s² + …`
`ŝ²−s²` — різниця ВИБІРКОВОЇ і ЗАМИКАНОЇ дисперсії передактивації. Фізика
пряма: `q = E[a²]` живе з `s²`, тож похибка дисперсії переходить у `q`.
🚨 РИЗИК: `ŝ²−s²` — ВИБІРКОВА величина ⟹ віднімання може з'їсти КЗ (саме так
померла проєкція якоря для `m` у С70). Тому міряємо В БАЛІ, а не в `R²`.

## 🔑 ЧОМУ ЦЕ ІНШИЙ ШЛЯХ, НІЖ ⛔`qfeat`
Там коефіцієнти шукались МІНІМІЗАЦІЄЮ MSE на 8 жеребах (шумно, LOO вмирав).
Тут вони взяті з ФІТУ `b_q` ПРОТИ ОРАКУЛА (32 жереби) — на порядок точніше,
і вони ФІКСОВАНІ, не підганяються під бал.

⛔ ПЛОМБИ: (а) множник 0 ⟹ РІВНО sub93; (б) ⛔ROLL коефіцієнтів; (в) ⛔лише
   домінантна ознака проти всіх 14; (г) банк, крос-ціль.

📣 ПЕРЕДБАЧЕНО: `R²=0.26` ⟹ якщо КЗ не постраждає, це має дати ~26% стелі
   ×1.0896, тобто ×1.02. Але `ŝ²−s²` корелює з шумом вибірки, а `λ_Q` уже
   налаштований гасити саме цей шум ⟹ найімовірніше поправка ЧАСТКОВО
   дублює те, що канал уже робить, і чистий виграш буде ×1.00–1.01.
   Якщо оптимальний множник вийде ВІД'ЄМНИМ — поправка діє проти каналу.

## СТАРА ШАПКА (для контексту)
ЗМІЩЕННЯ ОБРІЗАННЯ — АНАЛІТИЧНО, ДВОМА ЗАМИКАННЯМИ (С84)

## ЗВІДКИ (метафора ЮЗЕРА: «мікробугорок на початку зміщує всю траєкторію»)
У прямому прочитанні метафора СПРОСТОВАНА: коефіцієнт передачі збурення через
шар `G = 0.22` (`side66/selfcon.py`) — крок ГАСИТЬ бугорок у 4.5×, а якір із
шару 0 приходить до виходу у 8.1× меншим (С75).
🔑 АЛЕ вона працює буквально для РІШЕННЯ, а не для величини: `cut_budget91`
дивиться 128 пілотних пар і НАЗАВЖДИ викидає нейрони з `max h < 0`. Це не
доданок до траєкторії — це ЗМІНА САМОЇ МЕРЕЖІ, крізь яку йде решта вибірки,
і воно не гасне.

## 📊 РОЗМІР УЖЕ ЗМІРЯНИЙ, ПОПРАВКА — НІ
С82 ([[reference-bank-truth-is-the-right-target]]): зсув «обрізана проти повної»
= **1.5e-04…6.1e-04** при RMS робочого 1.4e-03 ⟹ **до 20% MSE**, і він
БІЛЬШИЙ за обидва шуми, тобто систематичний.
С81: «нейрон із `p_on≈1%` на 256 пілотних семплах виглядає мертвим, пропуск
7.6%» — і там же дослівно: **«як внесок У БАЛ НЕ МІРЯНО»**.
📌 Обрізання САМЕ ПО СОБІ — чистий плюс (без нього −30.25% MSE), бо дає більше
пар за той самий бюджет. Питання не «чи обрізати», а «чи компенсувати зсув».

## 🔑 ІДЕЯ: ЗСУВ ОБЧИСЛЮВАНИЙ БЕЗ ВИБІРКИ
Замикання вже рахується на ОБРІЗАНИХ вагах (`m_deep`). Порахувати його ще раз
на ПОВНИХ — і різниця
    `D = m_deep(ПОВНІ W) − m_deep(ОБРІЗАНІ wc)`
є АНАЛІТИЧНОЮ оцінкою зсуву обрізання: детермінована, без краплі шуму вибірки.
💰 ЦІНА: 32 матмули 256×256 = 6.7e7 флоп ≈ **0.8% бюджету** (вибірка — 7400×32
таких), і йде ПОВЗ `n_pairs`, бо `per_sample_cut` не змінюється.
🚨 Вихід останнього шару НЕ обрізається по стовпцях (`if i < depth-1`), тож
обидва `m_deep` — 256-вимірні й віднімаються прямо.

## ЩО МІРЯЄМО
`pr' = pr + β·D`, `β` — один скаляр, ОПТИМУМ АНАЛІТИЧНИЙ (прогноз лінійний):
    `MSE(β) = ⟨e₁e₂⟩ + β⟨D(e₁+e₂)⟩ + β²⟨D²⟩`
⛔ ПЛОМБИ І КОНТРОЛІ:
 (а) `β=0` ⟹ РІВНО базовий MSE;
 (б) ⛔ЧУЖА МЕРЕЖА: `D` від сусідньої мережі (зсув обрізання — ВЛАСТИВІСТЬ
     МЕРЕЖІ, тож жеребний контроль тут не є нулем — урок С75);
 (в) ⛔ШУМ тієї ж норми;
 (г) ЧЕСНИЙ LOO по мережах — `β` фітується без поточної мережі;
 (д) істина — БАНК (ПОВНА мережа!), крос-ціль.
📌 Міряємо на sub93 як базі (найкращий готовий кандидат).

📣 ПЕРЕДБАЧЕНО: знак `β` має вийти ДОДАТНИМ (обрізання ВІДКИДАЄ внесок ⟹ наш
   прогноз занижений, `D>0` його повертає) і СТАБІЛЬНИМ по мережах — це
   головна ознака, що ловимо систематику, а не шум. Розмір: зсув ~20% MSE, але
   замикання знає його неточно (власна похибка форми 0.19% для `m`), тож чекаю
   ×1.01–1.04. Якщо `β` вийде близьким до 1.0 — різниця замикань відтворює
   зсув майже точно, і це сильний результат. Якщо LOO ≈ ⛔чужа мережа —
   різниця несе лише спільну корпусну складову, і треба буде дивитись, чи
   виживає вона як ОДНА КОНСТАНТА для всіх мереж.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/trimbias.py [cuda:N] [NETS]
"""
import glob
import sys

import numpy as np
import torch

sys.path.insert(0, '/mnt/win_d/llm/White-Box/side66')
import sub91_port as SP                                            # noqa: E402
from sub91_port import (cut_budget91, closure91, _phi, _Phi,       # noqa: E402
                        KSIG, W_, GATE_N, INV_SQRT_2PI, PILOT_PAIRS,
                        ALPHA, W_BLEND)

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:3'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 48
L_, NDRAW = 32, 8
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)

_TT = np.load('/mnt/win_d/llm/White-Box/bank_s53/truth2.npz')
TRUTH2 = {int(v): (_TT['T1'][i], _TT['T2'][i])
          for i, v in enumerate(_TT['seeds'])}
WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have
    and int(np.load(f)['seed']) in TRUTH2][:NETS]
print(f'{len(FS)} мереж × {NDRAW} жеребів, {DEV}\n', flush=True)


def aux_parts(wc, dev):
    """+ опори `s`,`t`,`φ`,`Φ` — потрібні для ознак фіту."""
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    AA, BB, SS, TT_, PHc, PHIc = [], [], [], [], [], []
    for i in range(len(wc)):
        w = wc[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        mp = m @ w
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        if 1 <= i <= KSIG:
            s2ph = (s * s) * ph
            AA.append(s2ph); BB.append(s2ph * t)
            SS.append(s); TT_.append(t); PHc.append(ph); PHIc.append(Ph)
        m = mp * Ph + s * ph
        if i < len(wc) - 1:
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
    return AA, BB, SS, TT_, PHc, PHIc


QC = np.load('/tmp/claude-1000/-mnt-win-d-llm-White-Box/'
             '5379fdc5-ed40-4d43-9d92-b7da4256f2e6/scratchpad/qcoef.npy')


def run93(wc, npair, dev, seed, cl, aux, mu=0.0, only_dom=False, roll=False):
    """sub93 + `mu`·b̂_q, де b̂_q — фіт проти оракула (14 ознак)."""
    depth = len(wc)
    rng = np.random.default_rng(int(seed))
    rng.standard_normal((PILOT_PAIRS, W_))
    z = torch.tensor(rng.standard_normal((npair, W_)).astype(np.float32),
                     device=dev)
    cov = (z.T @ z) / float(npair)
    ev, V = torch.linalg.eigh(cov.double())
    wh = ((V * ev.clamp_min(1e-8).rsqrt()) @ V.T).float()
    y = z @ (wh @ wc[0])
    x = torch.cat([torch.relu(y), torch.relu(-y)], 0)
    den = float(2 * npair)
    sg0 = wc[0].double().pow(2).sum(0).clamp_min(1e-30).sqrt()
    v = x.double().sum(0) / den - sg0 * INV_SQRT_2PI
    s_cl, m_cl, q_cl, P_cl, m_deep = cl
    AA, BB, SS, TT_, PHc, PHIc = aux
    cf = np.roll(QC, 5) if roll else QC
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        pd = p.double()
        p2 = pd * pd
        mh = pd.sum(0) / den
        e2 = p2.sum(0) / den
        sh2 = (e2 - mh * mh).clamp_min(1e-30)
        sh = sh2.sqrt()
        x = torch.relu(p)
        if i <= KSIG:
            e3 = (p2 * pd).sum(0) / den
            e4 = (p2 * p2).sum(0) / den
            mh2 = mh * mh
            k3 = e3 - 3.0 * mh * e2 + 2.0 * mh2 * mh
            k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                  - 3.0 * mh2 * mh2 - 3.0 * sh2 * sh2)
            dq = ((k3 / (sh2 * sh)) * (AA[i - 1] / 3.0)
                  - (k4 / (sh2 * sh2)) * (BB[i - 1] / 12.0))
            if mu != 0.0:
                s2 = SS[i - 1] * SS[i - 1]
                t_ = TT_[i - 1]; ph_ = PHc[i - 1]; Ph_ = PHIc[i - 1]
                g1 = k3 / (sh2 * sh)
                g2 = k4 / (sh2 * sh2)
                one = torch.ones_like(s2)
                FE = [one, t_, Ph_, ph_, t_ * ph_, SS[i - 1], s2,
                      q_cl[i - 1], g1, g2, sh2 - s2, t_ * t_, ph_ * ph_,
                      t_ * Ph_]
                if only_dom:
                    bq = float(cf[10]) * FE[10] * s2
                else:
                    bq = sum(float(cf[j]) * FE[j] for j in range(14)) * s2
                dq = dq + mu * bq
            xm = x.double().sum(0) / den
            xq = x.double().pow(2).sum(0) / den
            v = (v + (SP.LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + SP.LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + SP.LAM_Q[i - 1] * ((xq - q_cl[i - 1]) - dq)
                 + SP.LAM_G[i - 1] * (gate - P_cl[i - 1]))
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    return ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)


MUS = [0.0, 0.25, 0.5, 1.0, 1.5]
VAR = [('🎯 усі 14 ознак', False, False),
       ('🔵 лише ŝ²−s² (домінантна)', True, False),
       ('⛔ ROLL коефіцієнтів', False, True)]
MSE = np.zeros((len(FS), len(VAR), len(MUS)))
SEAL0 = 0.0

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
         for q in range(L_)]
    with torch.no_grad():
        wc, npair, _ = cut_budget91(W, DEV)
        cl = closure91(wc, KSIG, DEV, kdeep=len(wc))
        aux = aux_parts(wc, DEV)
        b1, b2 = TRUTH2[sd0]
        T1 = torch.tensor(b1, dtype=torch.float64, device=DEV)
        T2 = torch.tensor(b2, dtype=torch.float64, device=DEV)
        for d in range(NDRAW):
            sd = 103_000_000 + 733 * n + d
            for v, (_, dom, rl) in enumerate(VAR):
                for k, mu in enumerate(MUS):
                    pr = run93(wc, npair, DEV, sd, cl, aux, mu, dom, rl)
                    MSE[n, v, k] += float(((pr - T1) * (pr - T2)).sum()) / NDRAW
            if d == 0:   # ⛔ПЛОМБА (а): mu=0 ⟹ РІВНО sub93 в усіх варіантах
                a0 = run93(wc, npair, DEV, sd, cl, aux, 0.0, False, False)
                a1 = run93(wc, npair, DEV, sd, cl, aux, 0.0, True, True)
                SEAL0 = max(SEAL0, float((a0 - a1).abs().max()))
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/qapply.npz', MSE=MSE,
         MUS=np.array(MUS))

print(f'\n⛔ПЛОМБА (а) mu=0 ⟹ РІВНО sub93: max|Δ| = {SEAL0:.2e} '
      f'{"✅" if SEAL0 == 0 else "🚨"}')
base = MSE[:, 0, 0].mean()
print(f'⛔база sub93 = {base:.6e}\n')
print(f'{"mu":>6s} ' + ' '.join(f'{v[0]:>26s}' for v in VAR))
print('─' * 90)
for k2, mu in enumerate(MUS):
    print(f'{mu:6.2f} ' + ' '.join(
        f'{MSE[:, v, k2].mean():13.6e} ×{base/MSE[:, v, k2].mean():6.4f}'
        for v in range(len(VAR))))
print('\n🎯 ВИРОК')
bi = [int(np.argmin(MSE[:, v, :].mean(0))) for v in range(len(VAR))]
g = [base / MSE[:, v, bi[v]].mean() for v in range(len(VAR))]
for v in range(len(VAR)):
    print(f'   {VAR[v][0]:>26s}: оптимум mu={MUS[bi[v]]:.2f} ⟹ ×{g[v]:.4f}')
net = g[0] / max(g[2], 1e-9)
print(f'   ЧИСТИЙ понад ⛔ROLL: ×{net:.4f}')
if g[0] > 1.003 and net > 1.003:
    print(f'   ⟹ 🟢 ЖИВЕ ⟹ бал 1.928e-07 → {1.928e-07/g[0]:.4e}')
else:
    print('   ⟹ ⛔ поправка дублює те, що λ_Q вже робить (з\'їдає КЗ)')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
