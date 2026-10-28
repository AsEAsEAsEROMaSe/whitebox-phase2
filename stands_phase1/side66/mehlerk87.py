"""ЧИ ВИНЕН ОБРИВ РЯДУ МЕЛЕРА В ЗАНИЖЕННІ `σ`. (С87)

## ЗВІДКИ — ТРИ ФАКТИ, ЯКІ РАЗОМ ДАЮТЬ ГІПОТЕЗУ
1. `ladder87.py`: масштаб нашої `σ` падає **1.0000 → 0.8724** ⟹ `σ` СИСТЕМАТИЧНО
   ЗАНИЖЕНА, і це не шум (розкид по мережах лише 1.2%).
2. ТЗ: `σ` на 3% хибна ⟹ ×176. Отже 12.8% заниження — головний ворог.
3. 📐 РЯД МЕЛЕРА (формула Прайса, ТОЧНА для бівар. гауса):
```
Cov(f(X),g(Y)) = Σ_{k≥1} (Cov(X,Y)^k / k!) · E[f^(k)(X)] · E[g^(k)(Y)]
```
Для `f = ReLU`, `X ~ N(m,σ²)`, `t = m/σ` — ВИВЕДЕНО через δ-функції:
```
E[f']   = Φ(t)                E[f'''] = −(t/σ²)φ(t)
E[f'']  = φ(t)/σ              E[f⁗]  = ((t²−1)/σ³)φ(t)
```
🔑 Члени `k=1` і `k=2` — це РІВНО те, що стоїть у сабміті. Обрив на `K=2`
ВІДКИДАЄ `k≥3`, а вони при `ρ→1` НЕ малі. І член `k=3` має вигляд
`(C³/6)·(t_jφ_j/σ_j²)(t_kφ_k/σ_k²)` — ДОБУТОК ДВОХ ОДНАКОВИХ СТРУКТУР ⟹
на діагоналі й при однакових знаках ДОДАТНИЙ ⟹ обрив ЗАНИЖУЄ.

## 🚨 ЧОМУ ЦЕ НЕ ПОВТОР ЗАКРИТОГО
[[finding-error-is-born-from-variance-error]] каже «`K=4` не змінює нічого».
АЛЕ там міряли ІНШИЙ об'єкт — «похибку `s`, поділену на похибку вибірки» на
шарах 1–31, тобто відношення, у якому масштабний дрейф не видно. Тут міряємо
ПРЯМО те, що виявилось вирішальним: **масштаб `r` і накопичення по ланцюгу**.
🎓 Це [[feedback-controls-have-three-distinct-diseases]] — «не той об'єкт».

## 💰 ЦІНА (гейт до реалізації)
Кожен додатковий член — одне елементне множення матриці `256²` плюс `outer`:
**6.6e4 флопів на шар × 32 = 2.1e6** = 0.0008% бюджету. Практично безкоштовно.
⟹ якщо працює, реалізується одразу.

## 🎯 ЩО МІРЯЄМО
Ланцюг при `K = 2,3,4,5,6`: масштаб `r` по шарах, похибка `σ`, фінальна
похибка `E[a_last]`. Плюс `K=3..6` разом з ОРАКУЛЬНОЮ формою (`L10`-режим).

⛔ПЛОМБИ:
 M1 `K=2` мусить відтворити 0.9416% і `r[31]=0.8724` (`ladder87.py`).
 M2 ДІАГОНАЛЬ вписується точною формулою НЕЗАЛЕЖНО від `K` ⟹ зміни мусять
    приходити ТІЛЬКИ з позадіагоналі. Перевірка: `‖diag(C_a)‖` не змінюється.
 M3 ⛔КОНТРОЛЬ ЗНАКУ: член `k=3` із ПЕРЕВЕРНУТИМ знаком мусить ЗАНИЖУВАТИ `r`
    ще сильніше. Якщо ні — формула не та.
 M4 шар 1: `p₁` гаусів ТОЧНО ⟹ `r[0] = 1.0000` при будь-якому `K`.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) `K=3` підніме `r[31]` з 0.872 у бік 1. Шанс **60%** — бо на глибині
     еф. ранг 3.2 ⟹ `ρ` близькі до 1, і саме там ряд збігається найгірше.
 (б) Приріст буде НЕ монотонним по `K`: непарні члени додатні на діагоналі,
     але позадіагональ має знаки `ρ³`. Шанс 50%.
 (в) 🎯Якщо `r[31] → 0.97+` і похибка `σ` впаде нижче 5%, фінальна похибка
     піде до 0.3–0.5% ⟹ ГІЛКА ЖИВА і коштує 0.0008% бюджету. Шанс **35%**.
 (г) ⚠️РИЗИК: ряд може РОЗБІГАТИСЬ при `ρ→1` (він асимптотичний), і `K=5,6`
     зроблять гірше за `K=2`. Шанс 40%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/mehlerk87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = 1 << 12
L_, W_ = 32, 256
KLIST = (2, 3, 4, 5, 6)
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def edge_relu(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    return s * ((t * P + f) + (g3 / 6.0) * (-t * f)
                + (g4 / 24.0) * ((t * t - 1.0) * f)
                + (g3 * g3 / 72.0) * ((t ** 4 - 6.0 * t * t + 3.0) * f))


def edge_relu2(m, s, g3, g4):
    t = m / s
    P, f = Phi(t), phi(t)
    s2 = s * s
    return ((m * m + s2) * P + m * s * f
            + (g3 / 3.0) * s2 * f - (g4 / 12.0) * s2 * t * f
            + (g3 * g3 / 36.0) * s2 * (3.0 * t - t ** 3) * f)


def relu_derivs(m, s, K):
    """`E[ReLU^(k)(X)]` для `k = 1..K` — ВИВЕДЕНО через δ-функції.

    `p(x)` — щільність `N(m,σ²)`; `E[δ^(k−2)(X)] = (−1)^(k−2) p^(k−2)(0)`, а
    `p^(j)(0) = φ(t)/σ^{j+1} · He_j(−t)·(−1)^j`. Розкрито до `k=6`."""
    t = m / s
    P, f = Phi(t), phi(t)
    d = [P, f / s]
    if K >= 3:
        d.append(-(t / (s * s)) * f)
    if K >= 4:
        d.append(((t * t - 1.0) / s ** 3) * f)
    if K >= 5:
        d.append((-(t ** 3 - 3.0 * t) / s ** 4) * f)
    if K >= 6:
        d.append(((t ** 4 - 6.0 * t * t + 3.0) / s ** 5) * f)
    return d[:K]


def chain(W, K=2, orc_g=None, orc_s=None, trace=None, flip3=False):
    dev = W[0].device
    n = W[0].shape[1]
    zero = torch.zeros(n, dtype=torch.float64, device=dev)
    C = W[0].T @ W[0]
    m = zero.clone()
    for i in range(1, len(W) + 1):
        s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
        if orc_s is not None:
            s = orc_s[i - 1]
        g3, g4 = orc_g[i - 1] if orc_g is not None else (zero, zero)
        if trace is not None:
            so = trace['orc'][i - 1]
            trace['r'].append(float((s * so).sum() / (so * so).sum()))
            trace['e'].append(float(torch.sqrt(((s - so) ** 2).mean()
                                               / (so * so).mean())))
        m_a = edge_relu(m, s, g3, g4)
        if i == len(W):
            return m_a
        # ── ряд Прайса-Мелера до порядку K ──
        dv = relu_derivs(m, s, K)
        Ck = C.clone()
        C_a = Ck * torch.outer(dv[0], dv[0])
        fact = 1.0
        for k in range(2, K + 1):
            Ck = Ck * C
            fact *= k
            sgn = -1.0 if (flip3 and k == 3) else 1.0
            C_a = C_a + (sgn / fact) * Ck * torch.outer(dv[k - 1], dv[k - 1])
        # 🔑 ДІАГОНАЛЬ — ТОЧНОЮ формулою (там `ρ=1`, ряд збігається найгірше)
        d2 = edge_relu2(m, s, g3, g4) - m_a * m_a
        C_a = C_a + torch.diag(torch.clamp(d2, min=1e-30) - torch.diagonal(C_a))
        m = m_a @ W[i]
        C = W[i].T @ (C_a @ W[i])
    raise RuntimeError


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · K={KLIST} · {DEV}\n', flush=True)

TAGS = [f'K={k}' for k in KLIST] + [f'K={k}+γ' for k in KLIST] + ['⛔K=3 знак−']
E = {t: np.zeros(len(FS)) for t in TAGS}
R = {t: np.zeros(len(FS)) for t in TAGS}
RTR = {k: np.zeros((len(FS), L_)) for k in KLIST}
ETR = {k: np.zeros((len(FS), L_)) for k in KLIST}

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        acc = torch.zeros(W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(8800 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                hd = h.double()
                p = hd
                for kk in range(4):
                    MOM[q, kk] += p.sum(0)
                    p = p * hd
                a = torch.relu(h)
            acc += a.sum(0, dtype=torch.float64)
            d += b
        MOM /= NS
        tru = acc / NS
        nrm = float(torch.sqrt((tru ** 2).mean()))
        OS, OG = [], []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
                  - 3.0 * c2 * c2)
            sq = torch.sqrt(c2)
            OS.append(sq)
            OG.append((c3 / sq ** 3, c4 / sq ** 4))

        for k in KLIST:
            tr = {'orc': OS, 'r': [], 'e': []}
            pr = chain(W, K=k, trace=tr)
            RTR[k][n] = np.array(tr['r'])
            ETR[k][n] = np.array(tr['e'])
            E[f'K={k}'][n] = float(torch.sqrt(((pr - tru) ** 2).mean())) / nrm
            R[f'K={k}'][n] = float((pr * tru).sum() / (tru * tru).sum())
            pg = chain(W, K=k, orc_g=OG)
            E[f'K={k}+γ'][n] = float(torch.sqrt(((pg - tru) ** 2).mean())) / nrm
            R[f'K={k}+γ'][n] = float((pg * tru).sum() / (tru * tru).sum())
        pf = chain(W, K=3, flip3=True)
        E['⛔K=3 знак−'][n] = float(torch.sqrt(((pf - tru) ** 2).mean())) / nrm
        R['⛔K=3 знак−'][n] = float((pf * tru).sum() / (tru * tru).sum())
    print(f'  net{n} · ' + ' · '.join(f'K{k} {E[f"K={k}"][n]:.4%}'
                                      for k in KLIST), flush=True)


def loo_resid(err, rat):
    out = np.zeros_like(err)
    for i in range(len(err)):
        msk = np.arange(len(err)) != i
        out[i] = math.sqrt(max(err[i] ** 2 - (rat[i] - 1.0) ** 2
                               + (rat[i] - rat[msk].mean()) ** 2, 1e-30))
    return out


print('\n══ МАСШТАБ `r` ПО ШАРАХ (мусить бути 1.0000; K=2 дає 0.8724 на ш.31) ══')
print('шар :' + ' '.join(f'{q+1:7d}' for q in range(0, L_, 5)))
for k in KLIST:
    print(f'K={k} :' + ' '.join(f'{RTR[k][:, q].mean():7.4f}'
                                for q in range(0, L_, 5)))
print('\n══ ПОХИБКА `σ` ПО ШАРАХ ══')
for k in KLIST:
    print(f'K={k} :' + ' '.join(f'{ETR[k][:, q].mean():6.2%}'
                                for q in range(0, L_, 5)))

print('\n══ ФІНАЛЬНА ПОХИБКА `E[a_last]` ══')
print('варіант      сира     після α    MSE        бал(×0.1)   vs sub97')
for t in TAGS:
    rs = loo_resid(E[t], R[t])
    mse = 1.698e-05 * (rs.mean() / 0.00941) ** 2
    print(f'{t:12s} {E[t].mean():7.4%} {rs.mean():7.4%}  {mse:.3e}  '
          f'{mse * 0.1:.3e}   ×{1.873e-07 / (mse * 0.1):6.3f}')
print(f'{"ЛІДЕР":12s} {0.0521 / 100:7.4%} {0.0521 / 100:7.4%}  {5.211e-08:.3e}  '
      f'{5.102e-08:.3e}   ×{1.873e-07 / 5.102e-08:6.3f}')
