"""ФІТ ЗМІЩЕННЯ `q` З ОЗНАК НЕЙРОНА — БАГАТОВИМІРНЕ УЗАГАЛЬНЕННЯ (С84)

## ЗВІДКИ
⛔`side66/qtune.py`: підбір ДВОХ скалярів (масштаб поправки, масштаб `λ_Q`) НЕ
переживає LOO — ×0.9957 до sub93, тобто теоретичні `1/3`,`1/12` кращі за
підігнані. АЛЕ оптимальний масштаб поправки `c_e = 2.605` СТАБІЛЬНИЙ (розкид
11%), а на ⛔шумі відповідний коефіцієнт падає до −0.196 ⟹ 🔑 сигнал у поправці
Є, Еджворт просто НЕДООЦІНЮЄ зміщення в ~2.6× (узгоджено з тим, що він бере
4.5% оракульної стелі ×1.0896).
⟹ 🎯 замість підбирати МАСШТАБ однієї поправки — дати ланцюгу БАЗИС ознак і
розв'язати лінійну систему. Та сама точна лінійність, що в `qtune`.

## ІСТОРІЯ (для контексту)
`estimator_sub93.py` дав ×1.0134 проти sub87 (sub92 — ×1.0046), але два
параметри в ньому НЕ підбирались:
1. коефіцієнти поправки стоять ТЕОРЕТИЧНІ (`γ₁/3`, `γ₂/12`) — виведені для
   ідеального Еджворта, а не для нашої ознаки;
2. `_LAM_Q` лишились ті, що підганялись під БРУДНУ ознаку, хоч `whatfeeds.py`
   зміряв: після чистки оптимум `λ` зсувається ×1.0 → **×1.5**.
Стеля чистки з ОРАКУЛОМ — ×1.0896, узято ×1.0088 ⟹ лежить ще ~90%.

## 🔑 БАЗИС ОЗНАК (усі ДОСТУПНІ В РАНТАЙМІ — із замикання або з вибірки)
`s,t,φ,Φ` — із ЗАМИКАННЯ (детерміновані); `γ₁,γ₂,ŝ,m̂` — з ТОГО Ж проходу:
```
V₁ = (γ₁/3)·s²φ      ← теоретичний член κ₃ (те, що стоїть у sub93)
V₂ = −(γ₂/12)·s²tφ   ← теоретичний член κ₄
V₃ = s²φ             ← сталий зсув форми (С70: стала складова дає 92% на 1..8)
V₄ = s²tφ
V₅ = s²(t²−1)φ
V₆ = (ŝ²−s²)·Φ       ← розбіжність вибіркової й замиканої дисперсії
V₇ = q_замик         ← сам рівень
V₈ = γ₂·s²φ          ← ексцес без теоретичного множника
```
🚨 С70 навчила: **фіт мусить бути в ТІЙ нормі, в якій живе бал**
([[finding-closure-bias-is-a-universal-function-of-8-neuron-features]] §головна
помилка). Тому тут НЕ фітується `b_q` покомпонентно — кожна ознака ПЕРЕНОСИТЬСЯ
ланцюгом воріт до виходу, і мінімізується САМ MSE. Норма правильна за побудовою.

## 🔑 ЧОМУ НЕ ПОТРІБНА СІТКА: ПРОГНОЗ ЛІНІЙНИЙ ПО ОБОХ
`pr = (1−w)·ours + w(1+α)·m_deep`, `ours = last − v`, а `v` входить лінійно:
    `v ← v + λ_Q·(δ_Q − c_e·Δq)`  ⟹  `pr(A,B) = pr₀ + A·U + B·V`
де `pr₀` — прогноз при `λ_Q=0`, `U` — внесок родини `Q` з нинішніми `λ_Q`,
`V` — внесок поправки. Гейти й вибірка ТІ САМІ ⟹ лінійність ТОЧНА, не наближена.
⟹ три прогони на жереб дають ПОВНУ поверхню, оптимум — 2×2 система:
```
MSE(A,B) = ⟨e₁e₂⟩ + A⟨U·(e₁+e₂)⟩ + B⟨V·(e₁+e₂)⟩ + A²⟨U²⟩ + B²⟨V²⟩ + 2AB⟨UV⟩
```
Переклад назад: `A = c_λ`, `B = −c_λ·c_e` ⟹ `c_e = −B/A`.

⛔ ПЛОМБИ:
 (а) ЛІНІЙНІСТЬ: `pr(1,−1)` з формули мусить збігтися з ПРЯМИМ прогоном sub93
     побітово — якщо ні, розклад хибний і вся поверхня недійсна;
 (б) `(A,B) = (1,0)` мусить відтворити sub92, `(1,−1)` — sub93;
 (в) LOO ПО МЕРЕЖАХ: оптимум фітується без поточної мережі;
 (г) ⛔контроль: та сама оптимізація на ЧИСТОМУ ШУМІ замість `V`;
 (д) істина — БАНК, крос-ціль.

📣 ПЕРЕДБАЧЕНО: 8 ознак на 24 мережах — межа переобтяження, тож головним
   числом буде LOO, а не оптимум на всіх. Чекаю LOO ×1.005–1.02 до sub93.
   ⛔КОНТРОЛЬ: той самий фіт на 8 ЧИСТО ШУМОВИХ ознаках тієї ж норми — він
   покаже, скільки дає САМ ФАКТ восьми вільних параметрів. Виграш
   засвідчується ЛИШЕ понад цей контроль.
   Якщо LOO ≈ контроль — гілка «фіт зміщення `q`» закривається, і лишається
   єдиний невипробуваний шлях: коефіцієнти з БАНКУ (2454 мережі), де LOO
   надійний за побудовою.

## СТАРЕ ПЕРЕДБАЧЕННЯ (`qtune`, для протоколу): `c_λ` вийде **>1** (бо чистка зменшила зміщення ⟹ можна
   довіряти ознаці більше) і `c_e` близько 1..2. Чекаю сумарно ×1.01–1.03 до
   sub93, тобто помітно менше за стелю ×1.0896: два скаляри не можуть
   замінити ПОШАРОВУ оцінку зміщення. Якщо LOO дасть <1.005 — важелі порожні
   і треба одразу йти в апарат С70 (фіт зміщення `q` з ознак нейрона).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/qtune.py [cuda:N] [NETS]
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
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
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


NF = 8


def run(wc, npair, dev, seed, cl, aux, lamq_on, edge_on, basis=False):
    """Прогін сабміту. `basis=True` — ще й переносить БАЗИС ознак до виходу."""
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
    AA, BB, S2, TT2, PHc, PHIc = aux
    Bm = (torch.zeros(NF, wc[0].shape[1], dtype=torch.float64, device=dev)
          if basis else None)
    for i in range(1, depth):
        p = x @ wc[i]
        gate = (p[:GATE_N] > 0).to(p.dtype).sum(0).double() / GATE_N
        wd = wc[i].double()
        v = (v @ wd) * gate
        if Bm is not None:
            Bm = (Bm @ wd) * gate.unsqueeze(0)
        pd = p.double()
        p2 = pd * pd
        mh = pd.sum(0) / den
        e2 = p2.sum(0) / den
        sh2 = (e2 - mh * mh).clamp_min(1e-30)
        sh = sh2.sqrt()
        x = torch.relu(p)
        xm = x.double().sum(0) / den
        xq = x.double().pow(2).sum(0) / den
        if i <= KSIG:
            dq = 0.0
            if edge_on:
                e3 = (p2 * pd).sum(0) / den
                e4 = (p2 * p2).sum(0) / den
                mh2 = mh * mh
                k3 = e3 - 3.0 * mh * e2 + 2.0 * mh2 * mh
                k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                      - 3.0 * mh2 * mh2 - 3.0 * sh2 * sh2)
                dq = ((k3 / (sh2 * sh)) * (AA[i - 1] / 3.0)
                      - (k4 / (sh2 * sh2)) * (BB[i - 1] / 12.0))
            if Bm is not None:
                e3 = (p2 * pd).sum(0) / den
                e4 = (p2 * p2).sum(0) / den
                mh2 = mh * mh
                k3 = e3 - 3.0 * mh * e2 + 2.0 * mh2 * mh
                k4 = (e4 - 4.0 * mh * e3 + 6.0 * mh2 * e2
                      - 3.0 * mh2 * mh2 - 3.0 * sh2 * sh2)
                g1 = k3 / (sh2 * sh)
                g2 = k4 / (sh2 * sh2)
                lq = SP.LAM_Q[i - 1]
                FV = ((g1 / 3.0) * AA[i - 1],
                      -(g2 / 12.0) * BB[i - 1],
                      AA[i - 1],
                      BB[i - 1],
                      S2[i - 1] * (TT2[i - 1] * TT2[i - 1] - 1.0) * PHc[i - 1],
                      (sh2 - S2[i - 1]) * PHIc[i - 1],
                      q_cl[i - 1],
                      g2 * AA[i - 1])
                for kk in range(NF):
                    Bm[kk] = Bm[kk] + lq * FV[kk]
            v = (v + (SP.LAM_S[i - 1] * _phi(mh / sh)) * (sh - s_cl[i - 1])
                 + SP.LAM_A[i - 1] * (xm - m_cl[i - 1])
                 + SP.LAM_G[i - 1] * (gate - P_cl[i - 1]))
            if lamq_on:
                v = v + SP.LAM_Q[i - 1] * ((xq - q_cl[i - 1]) - dq)
        if i == depth - 1:
            last = x.double().sum(0) / den
    ours = last - v
    pr = ours + W_BLEND * ((1.0 + ALPHA) * m_deep - ours)
    # 🔑 ознака входить у `v` зі знаком «−» у `ours = last − v`, а далі бленд
    #    масштабує на (1−w) ⟹ внесок ознаки у `pr` = −(1−w)·Bm
    return (pr, -(1.0 - W_BLEND) * Bm) if basis else pr


def aux_parts(wc, dev):
    """`s²φ(t)` і `s²tφ(t)` із замикання на шарах 1..KSIG (як у sub93)."""
    m = torch.zeros(wc[0].shape[0], dtype=torch.float64, device=dev)
    C = torch.eye(wc[0].shape[0], dtype=torch.float64, device=dev)
    AA, BB, S2, TT2, PHc, PHIc = [], [], [], [], [], []
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
            S2.append(s * s); TT2.append(t); PHc.append(ph); PHIc.append(Ph)
        m = mp * Ph + s * ph
        if i < len(wc) - 1:
            d2 = (mp * mp + s * s) * Ph + mp * s * ph
            C = cp * torch.outer(Ph, Ph) + (0.5*cp*cp) * torch.outer(ph/s, ph/s)
            C = C + torch.diag((d2 - m * m) - C.diagonal())
    return AA, BB, S2, TT2, PHc, PHIc


# регресори: R₀ = U (родина Q як є) + NF ознак базису ⟹ NR = 1 + NF
NR = 1 + NF
E12 = np.zeros(len(FS))                 # ⟨e₁e₂⟩ при λ_Q = 0
LIN = np.zeros((len(FS), NR))           # ⟨R_k·(e₁+e₂)⟩
QUA = np.zeros((len(FS), NR, NR))       # ⟨R_k·R_l⟩
E12c = np.zeros(len(FS))
LINc = np.zeros((len(FS), NR))
QUAc = np.zeros((len(FS), NR, NR))
SEAL_LIN = 0.0

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
        gen = torch.Generator(device=DEV); gen.manual_seed(31 + n)
        for d in range(NDRAW):
            sd = 97_000_000 + 733 * n + d
            p0 = run(wc, npair, DEV, sd, cl, aux, False, False)   # λ_Q = 0
            pu, Bm = run(wc, npair, DEV, sd, cl, aux, True, False, basis=True)
            pv = run(wc, npair, DEV, sd, cl, aux, True, True)     # sub93
            U = pu - p0
            # ⛔ПЛОМБА (а): V₁+V₂, перенесені базисом, мусять ТОЧНО дати sub93
            if d == 0:
                SEAL_LIN = max(SEAL_LIN, float(
                    (pu - (Bm[0] + Bm[1]) - pv).abs().max()))
            e1, e2 = p0 - T1, p0 - T2
            es = e1 + e2
            R = torch.stack([U] + [Bm[k] for k in range(NF)], 0)
            rnd = torch.randn(R.shape, dtype=torch.float64, device=DEV,
                              generator=gen)
            Rc = torch.stack([U] + [rnd[k] * (Bm[k].norm()
                                              / rnd[k].norm().clamp_min(1e-300))
                                    for k in range(NF)], 0)
            for eA, lA, qA, Rx in ((E12, LIN, QUA, R),
                                   (E12c, LINc, QUAc, Rc)):
                eA[n] += float((e1 * e2).sum()) / NDRAW
                lA[n] += (Rx * es).sum(1).cpu().numpy() / NDRAW
                qA[n] += (Rx @ Rx.T).cpu().numpy() / NDRAW
    del W, wc
    torch.cuda.empty_cache()
    print(f'  … {n+1}/{len(FS)}  seed {sd0}', flush=True)

np.savez('/mnt/win_d/llm/White-Box/arena62/logs/qfeat.npz',
         E12=E12, LIN=LIN, QUA=QUA, E12c=E12c, LINc=LINc, QUAc=QUAc)


def mse_c(E, L, Q, idx, c):
    """MSE при коефіцієнтах `c` (довжина NR) на підмножині мереж."""
    return float((E[idx] + L[idx] @ c
                  + np.einsum('i,nij,j->n', c, Q[idx], c)).mean())


def opt_c(E, L, Q, idx, k, ridge=1e-12):
    """Оптимум для ПЕРШИХ `k` регресорів (решта = 0)."""
    A = 2.0 * Q[idx][:, :k, :k].sum(0)
    b = -L[idx][:, :k].sum(0)
    A = A + ridge * np.trace(A) / max(k, 1) * np.eye(k)
    c = np.zeros(L.shape[1])
    try:
        c[:k] = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        c[0] = 1.0
    return c


ALL = np.arange(len(FS))
c92 = np.zeros(NR); c92[0] = 1.0
c93 = np.zeros(NR); c93[0] = 1.0; c93[1] = -1.0; c93[2] = -1.0
b92 = mse_c(E12, LIN, QUA, ALL, c92)
b93 = mse_c(E12, LIN, QUA, ALL, c93)
print(f'\n⛔ПЛОМБА (а) базис відтворює sub93: max|Δ| = {SEAL_LIN:.2e} '
      f'{"✅" if SEAL_LIN < 1e-15 else "🚨 БАЗИС НЕ ТОЙ, ПОВЕРХНЯ НЕДІЙСНА"}')
print(f'⛔ПЛОМБА (б) sub92 = {b92:.6e} · sub93 = {b93:.6e} · '
      f'sub93/sub92 = ×{b92/b93:.4f}\n')

print(f'{"k ознак":>9s} {"опт. на всіх":>14s} {"×":>8s} {"ЧЕСНИЙ LOO":>14s} '
      f'{"×до93":>8s} {"⛔КОНТРОЛЬ LOO":>15s} {"×":>8s}')
print('─' * 84)
best = (0.0, 0)
for k in range(1, NR + 1):
    ca = opt_c(E12, LIN, QUA, ALL, k)
    va = mse_c(E12, LIN, QUA, ALL, ca)
    lo = lc = 0.0
    for n in range(len(FS)):
        o = np.array([j for j in range(len(FS)) if j != n])
        lo += mse_c(E12, LIN, QUA, np.array([n]), opt_c(E12, LIN, QUA, o, k))
        lc += mse_c(E12c, LINc, QUAc, np.array([n]),
                    opt_c(E12c, LINc, QUAc, o, k))
    lo /= len(FS); lc /= len(FS)
    print(f'{k:9d} {va:14.6e} {b93/va:8.4f} {lo:14.6e} {b93/lo:8.4f} '
          f'{lc:15.6e} {b93/lc:8.4f}')
    if b93 / lo > best[0]:
        best = (b93 / lo, k)

print('\n🎯 ВИРОК')
kb = best[1]
lc = 0.0
for n in range(len(FS)):
    o = np.array([j for j in range(len(FS)) if j != n])
    lc += mse_c(E12c, LINc, QUAc, np.array([n]),
                opt_c(E12c, LINc, QUAc, o, kb))
lc /= len(FS)
print(f'   найкраще k = {kb}: LOO ×{best[0]:.4f} до sub93, '
      f'⛔контроль ×{b93/lc:.4f}')
net = best[0] / max(b93 / lc, 1e-12)
print(f'   ЧИСТИЙ виграш понад контроль: ×{net:.4f} ⟹ бал 1.928e-07 → '
      f'{1.928e-07/max(net,1e-9):.4e}')
print(f'   📌 стеля чистки `q` з ОРАКУЛОМ: ×1.0896')
ca = opt_c(E12, LIN, QUA, ALL, kb)
print(f'   коефіцієнти: ' + ' '.join(f'{x:+.3f}' for x in ca[:kb]))
if net < 1.005:
    print('   ⟹ ⛔ БАЗИС НЕ ДАЄ ПОНАД КОНТРОЛЬ на 24 мережах')
    print('      ⟹ лишається шлях коефіцієнтів із БАНКУ (2454 мережі)')
else:
    print('   ⟹ 🟢 є що брати; далі — зашити коефіцієнти й перевірити ab_files')
print(f'\n📊 БАЛ: sub87 1.954e-07 · sub92 1.946e-07 · sub93 1.928e-07')
