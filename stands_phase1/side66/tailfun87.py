"""РОЗКЛАД ПОХИБКИ ЗАМИКАННЯ НА СКЛАДОВІ ЧЕРЕЗ ХВОСТОВІ ФУНКЦІОНАЛИ. (С87)

## ЗВІДКИ
`stepcov87.py`: похибка одного кроку 1.6%, і вона сидить у ПОЗАДІАГОНАЛІ
(0.81→1.69%), а не в діагоналі (0.32→0.23%). Маргінальні `γ₃,γ₄` її не
пояснюють (R²≈0). Треба зрозуміти, З ЧОГО САМЕ вона складається.

## 📐 ТОЧНА ТОТОЖНІСТЬ (виведено, не вгадано)
`ReLU(x) = x − x·1_{x<0}`. Хай `u = p·1_{p<0}`. Тоді ТОЧНО:
```
E[a_j a_k] = C_jk + m_j m_k − A_jk − A_kj + Q_jk     A_jk = E[u_j p_k]
E[a_j]     = m_j − T₁_j                              Q_jk = E[u_j u_k]
⟹ Cov(a)_jk = C_jk − A_jk − A_kj + Q_jk + m_j T₁_k + m_k T₁_j − T₁_j T₁_k
```
🔑 `A` і `Q` — ГРАМ-МАТРИЦІ від `u`, тобто той самий матмул, що й для `C`.
Жодних тензорів `n³`.

📐 ЛІНЕАРИЗАЦІЯ регресії `E[p_k|p_j] = m_k + (C_jk/s_j²)(p_j−m_j)` дає
`A_jk ≈ m_k T₁_j + (C_jk/s_j²)·D_j`, `D = T₂ − m·T₁`, звідки
`Cov(a)_jk ≈ C_jk[1 − D_j/s_j² − D_k/s_k²] − T₁_j T₁_k + Q_jk`.
✅ПЛОМБА НА ГАУСАХ: `D = s²Φ(−t)` ⟹ `C_jk[1−Φ(−t_j)−Φ(−t_k)]`, а Мелер дає
`Φ(t_j)Φ(t_k)C_jk`; різниця = `Φ(−t_j)Φ(−t_k)C_jk` ≈ 1.8e-6 при t=3 ⟹ це та
сама формула. Отже перепараметризація — ТОЧНА, а не нова апроксимація.

## 🎯 НАВІЩО ЦЕ ПОТРІБНО — ЧОМУ `T` ЛЕГШЕ ЗА `σ`
Пряма вимога до `σ` — 0.465%. Але `σ` входить у відповідь як `s² − (щось малe)`;
розрив `s² − Var(a)` при `t=3` = лише 0.2496% від `s²`. Тому відносна точність,
потрібна від хвостових функціоналів `T`, послаблена приблизно в **×80** (5.8%
замість 0.465%). Це і є важіль перепараметризації.
🔑 І `T₁,T₂` — ОДНОВИМІРНІ хвостові інтеграли маргіналі, а хвіст суми описується
СЕДЛОВОЮ ТОЧКОЮ (валідна на 3σ, на відміну від Еджворта, чий ряд там
розбігається: `|γ₃He₃(−3)/6| = 1.5 > 1`).

## 🎯 ЩО САМЕ МІРЯЄТЬСЯ — ІЗОЛЯЦІЯ КОЖНОГО ДЖЕРЕЛА
На істинних `m, C` шару рахуємо `σ'` кількома способами і дивимось похибку:
```
V0 ТОЧНА тотожність (A,Q,T₁ з МК)          ⛔ПЛОМБА: мусить дати РІВНО 0
V1 лінеаризація A, решта з МК               ← ціна ЛІНЕАРИЗАЦІЇ регресії
V2 V1 без Q (спільний хвіст викинуто)       ← ціна викидання Q
V3 гаусові T₁,D (це наше НИНІШНЄ замикання) ← має дати ~1.6%
V4 ОРАКУЛЬНІ T₁,D + лінеаризація + без Q    ← 🎯 СТЕЛЯ перепараметризації
```
🔑 `V4` — головне число: скільки лишиться, якщо `T` знати ТОЧНО. Якщо `V4` не
набагато кращий за `V3` — уся гілка мертва, бо седлова точка дасть щонайбільше
оракульні `T`.

## ⛔ПЛОМБИ
 P1 `V0` = рівно 0 (машинна точність) — інакше тотожність зібрана неправильно.
 P2 `V3` ≈ 1.6% — має відтворити `stepcov87`, інакше не той об'єкт.
 P3 шум істини окремою колонкою.
 P4 ⛔КОНТРОЛЬ: `T` ІНШОГО шару замість свого — мусить бути помітно гірше.

📣 ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
 (а) `V4` дасть 0.3–0.8% (проти 1.6% зараз) ⟹ виграш ×2–5 у похибці кроку.
     Шанс **45%**.
 (б) лінеаризація (`V0`→`V1`) коштуватиме менше за 0.3%. Шанс 55%.
 (в) `Q` виявиться незначущим (`V1`≈`V2`). Шанс 70% — його придушення
     `φ_ρ(−t,−t)` ~1.6e-4.
 (г) якщо `V4` ≈ `V3`, гілка мертва одразу і далі седлову точку не рахуємо.
     Шанс на це — 40%.

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/tailfun87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
LOG2N = int(sys.argv[3]) if len(sys.argv) > 3 else 19
NS = 1 << LOG2N
BATCH = min(1 << 13, max(1 << 8, NS // 4))
L_, W_ = 32, 256
PROBE = (3, 7, 11, 15, 19, 23, 27, 30)
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('Запуск')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def qform(W, C):
    return torch.clamp((C @ W * W).sum(0), min=1e-30)


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · 2^{LOG2N} семплів · {DEV} · шари {PROBE}\n', flush=True)

TAGS = ('V0 ТОЧНА тотожність', 'V1 +лінеаризація A', 'V2 БЕЗ Q (контроль)',
        'V3 гаусові T (наше)', 'V4 оракульні T +лін +Q',
        '⛔V5 T чужого шару', 'шум істини')
NP = len(PROBE)
E = {t: np.zeros((len(FS), NP)) for t in TAGS}
TREL = np.zeros((len(FS), NP))     # наскільки гаусова T₁ бреше проти істинної
DREL = np.zeros((len(FS), NP))

for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        Z = {k: torch.zeros(NP, W_, W_, dtype=torch.float64, device=DEV)
             for k in ('GH', 'GA', 'AUP', 'QUU', 'GAH')}
        S = {k: torch.zeros(NP, W_, dtype=torch.float64, device=DEV)
             for k in ('SH', 'SA', 'SU', 'SU2', 'SAH')}
        g = torch.Generator(device=DEV)
        g.manual_seed(6100 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                a = torch.relu(h)
                if q in PROBE:
                    j = PROBE.index(q)
                    hd, ad = h.double(), a.double()
                    ud = torch.where(hd < 0, hd, torch.zeros_like(hd))
                    S['SH'][j] += hd.sum(0)
                    S['SA'][j] += ad.sum(0)
                    S['SU'][j] += ud.sum(0)
                    S['SU2'][j] += (ud * ud).sum(0)
                    Z['GH'][j] += hd.T @ hd
                    Z['GA'][j] += ad.T @ ad
                    Z['AUP'][j] += ud.T @ hd        # A_jk = E[u_j p_k]
                    Z['QUU'][j] += ud.T @ ud        # Q_jk = E[u_j u_k]
                    if d < NS // 2:
                        S['SAH'][j] += ad.sum(0)
                        Z['GAH'][j] += ad.T @ ad
            d += b

        for j, q in enumerate(PROBE):
            mh = S['SH'][j] / NS
            Ch = Z['GH'][j] / NS - torch.outer(mh, mh)
            Ch = 0.5 * (Ch + Ch.T)
            ma = S['SA'][j] / NS
            Ca = Z['GA'][j] / NS - torch.outer(ma, ma)
            Ca = 0.5 * (Ca + Ca.T)
            Wn = W[q + 1]
            s_true = torch.sqrt(qform(Wn, Ca))

            def err(Cx):
                return float(torch.sqrt(
                    (((torch.sqrt(qform(Wn, Cx)) - s_true) ** 2).mean()
                     / (s_true * s_true).mean())))

            T1 = S['SU'][j] / NS
            T2 = S['SU2'][j] / NS
            A = Z['AUP'][j] / NS - torch.outer(T1, mh)
            Q = Z['QUU'][j] / NS - torch.outer(T1, T1)
            s = torch.sqrt(torch.clamp(torch.diagonal(Ch), min=1e-30))
            Dv = T2 - mh * T1

            # V0 — ТОЧНА тотожність (усе з МК)
            C0 = Ch - A - A.T + Q
            E['V0 ТОЧНА тотожність'][n, j] = err(C0)

            # V1 — лінеаризація A: A_jk ≈ (C_jk/s_j²)·D_j  (після зняття m-члена)
            lin = (Ch / (s * s)[:, None]) * Dv[:, None]
            C1 = Ch - lin - lin.T + Q
            E['V1 +лінеаризація A'][n, j] = err(C1)
            C2 = Ch - lin - lin.T
            E['V2 БЕЗ Q (контроль)'][n, j] = err(C2)

            # V3 — НАШЕ НИНІШНЄ ЗАМИКАННЯ (Мелер K=2 + точна діагональ),
            #      слово в слово як у сабміті — це база для порівняння.
            t = mh / s
            P_, f_ = Phi(t), phi(t)
            C3 = Ch * torch.outer(P_, P_) + (0.5 * (Ch * Ch)) * torch.outer(
                f_ / s, f_ / s)
            m_a = mh * P_ + s * f_
            q2 = (mh * mh + s * s) * P_ + (mh * s) * f_
            C3 = C3 + torch.diag(torch.clamp(q2 - m_a * m_a, min=1e-30)
                                 - torch.diagonal(C3))
            E['V3 гаусові T (наше)'][n, j] = err(C3)

            # V4 — ОРАКУЛЬНІ T + лінеаризація + ТОЧНИЙ Q ⟹ СТЕЛЯ гілки.
            #      🚨 Q НЕ ВИКИДАЄМО: димовий прогін показав, що без нього
            #      похибка 65–81% (половина нейронів має t≈−3 ⟹ 1_{p<0}≈1
            #      тотожно, і Q_jk ≈ C_jk є ГОЛОВНИМ членом, а не поправкою).
            C4 = C1 + torch.diag(torch.diagonal(Ca) - torch.diagonal(C1))
            E['V4 оракульні T +лін +Q'][n, j] = err(C4)

            # ⛔V5 — T ЧУЖОГО шару
            jo = (j + 3) % NP
            so = torch.sqrt(torch.clamp(
                torch.diagonal(Z['GH'][jo] / NS
                               - torch.outer(S['SH'][jo] / NS,
                                             S['SH'][jo] / NS)), min=1e-30))
            Do = (S['SU2'][jo] / NS
                  - (S['SH'][jo] / NS) * (S['SU'][jo] / NS)) * (s * s) / (so * so)
            lino = (Ch / (s * s)[:, None]) * Do[:, None]
            C5 = Ch - lino - lino.T + Q
            E['⛔V5 T чужого шару'][n, j] = err(
                C5 + torch.diag(torch.diagonal(Ca) - torch.diagonal(C5)))

            mah = S['SAH'][j] / (NS // 2)
            Cah = Z['GAH'][j] / (NS // 2) - torch.outer(mah, mah)
            E['шум істини'][n, j] = err(0.5 * (Cah + Cah.T))

            # наскільки ГАУСОВІ хвостові функціонали брешуть проти істинних —
            # це і є вимога до седлової точки: якщо гаусові вже точніші за
            # потрібні 5.8%, то уточнювати `T` НЕМА ЧОГО.
            T1g = s * (t * Phi(-t) - phi(t))
            Dg = s * s * Phi(-t)
            TREL[n, j] = float(torch.sqrt(((T1g - T1) ** 2).mean()
                                          / (T1 * T1).mean()))
            DREL[n, j] = float(torch.sqrt(((Dg - Dv) ** 2).mean()
                                          / (Dv * Dv).mean()))
    print(f'  net{n} · V0 {E["V0 ТОЧНА тотожність"][n].mean():.2e} · '
          f'V3 {E["V3 гаусові T (наше)"][n].mean():.3%} · '
          f'V4 {E["V4 оракульні T +лін +Q"][n].mean():.3%}', flush=True)

print(f'\n{"═" * 76}\n⛔ПЛОМБИ')
v0 = E['V0 ТОЧНА тотожність'].mean()
print(f' P1 V0 (точна тотожність) = {v0:.3e}  '
      f'{"✅" if v0 < 1e-12 else "⛔ тотожність зібрана невірно"}')
v3 = E['V3 гаусові T (наше)'].mean()
print(f' P2 V3 = {v3:.4%}  (мусить ≈1.6% зі stepcov87)  '
      f'{"✅" if 0.008 < v3 < 0.030 else "⛔ не той об єкт"}')
print(f' P3 шум істини = {E["шум істини"].mean():.4%}')
v4, v5 = E['V4 оракульні T +лін +Q'].mean(), E['⛔V5 T чужого шару'].mean()
print(f' P4 ⛔чужі T = {v5:.4%} проти своїх {v4:.4%}  '
      f'{"✅" if v5 > v4 * 1.3 else "⛔ свої T не кращі за чужі"}')

print(f'\n{"═" * 76}')
print('РОЗКЛАД ПОХИБКИ `σ` НА КРОЦІ — ПО ШАРАХ')
print('шар             : ' + ' '.join(f'{q + 1:8d}' for q in PROBE))
for t in TAGS:
    print(f'{t:16s}: ' + ' '.join(f'{E[t][:, j].mean():8.3%}'
                                  for j in range(NP)))
print('\nнаскільки гаусові T брешуть проти істинних:')
print('T₁ віднос.      : ' + ' '.join(f'{TREL[:, j].mean():8.3%}'
                                      for j in range(NP)))
print('D  віднос.      : ' + ' '.join(f'{DREL[:, j].mean():8.3%}'
                                      for j in range(NP)))

print(f'\n📣 ГОЛОВНЕ ЧИСЛО: V3 (наше) {v3:.3%} → V4 (оракульні T) {v4:.3%}'
      f'  ⟹ ×{(v3 / max(v4, 1e-12)) ** 2:.2f} у MSE кроку')
print(f'   якщо ×<1.5 — гілку закрито, седлову точку не рахуємо')
print(f'   потрібна точність T: ~5.8% (проти 0.465% для прямої σ) — послаблення ×80')
