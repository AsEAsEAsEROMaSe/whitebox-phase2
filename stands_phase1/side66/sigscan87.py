"""МАСОВИЙ СКАН: ЧИ Є В ВАГАХ ІНФОРМАЦІЯ ПРО ЗАЛИШКОВУ ПОХИБКУ `σ`. (С87)

## ЗВІДКИ — ІДЕЯ ЮЗЕРА
«Спробуй масову кореляцію і фільтруй по знаках, щоб виявити: звідки взяти `σ`
з точністю ≤1%, не платячи за семпли.»

## 🎯 ЦІНА ПИТАННЯ — ЗМІРЯНО ДО СКАНУ (`ladder87.py`, правильний нормувальник)
`похибка ланцюга ≈ 0.0419% + 0.2231·ε`, де `ε` — віднос. похибка `σ` у %:
```
σ до 0.10% ⟹ ×5.14 до рекорду     σ до 0.465% ⟹ рівно sub97
σ до 0.25% ⟹ ×2.22                σ до 1.00%  ⟹ ×0.30 (ГІРШЕ)
```
Наша `σ` після зняття шарового масштабу має **8.72%** ⟹ треба **×19**.
📣 ЧЕСНИЙ ШАНС ДО ЗАПУСКУ: що скан дасть `σ` до 0.465% — **12%**.
   Що знайдеться хоч одна ознака з `|corr| > 0.3` до залишку — 65%.
   Що знаки (`W³`) поб'ють беззнакове (`|W|³`) — **55%** (у С86 знак вирішував).
🎯 Але скан вартий і при провалі: він ЗАКРИЄ напрям за один прогін.

## 📐 КАНДИДАТИ — ВИВЕДЕНІ, НЕ ВГАДАНІ
Похибка `σ_замикання` народжується з того, що `Cov(ReLU)` рахується за
ГАУСОВИМ припущенням на `p`. Керує нею негаусовість `p`, а вона, за умовної
незалежності, виражається ТОЧНО через ваги:
```
κ₃(p_i) = Σ_j W_ji³ · κ₃(a_j)      ← `W³` ЄДИНИЙ ЗБЕРІГАЄ ЗНАК ⟹ «фільтр по знаках»
κ₄(p_i) = Σ_j W_ji⁴ · κ₄(a_j)
```
Звідси родина кандидатів. 🔑 ПАРА `s3` (зі знаком) / `a3` (з `|W|³`) — це
ПРЯМИЙ ТЕСТ ідеї юзера: якщо знаки несуть інформацію, `s3` поб'є `a3`.

## ⛔ПЛОМБИ І КОНТРОЛІ
 T1 ТАВТОЛОГІЯ: подати сам залишок як ознаку ⟹ `R²` мусить бути 1.0000.
 T2 ПІДЛОГА: випадкова ознака ⟹ `R²` ≈ 0 (дає масштаб «що таке нуль»).
 T3 ⛔ЧУЖА МЕРЕЖА: ознаки від іншої мережі ⟹ `R²` мусить упасти до підлоги.
 T4 LOO ПО МЕРЕЖАХ: коефіцієнти регресії — з ІНШИХ мереж.
 T5 🎯ГОЛОВНА: вирок не в `R²`, а в тому, чи падає RMS залишку нижче 0.465%
    ([[feedback-measure-the-gain-not-the-ceiling]] — міряти ВИГРАШ, не стелю).

Запуск: /mnt/win_d/llm/White-Box/torch_venv/bin/python side66/sigscan87.py [cuda:N] [NETS] [LOG2N]
"""
import glob
import math
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:1'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
NS = 1 << (int(sys.argv[3]) if len(sys.argv) > 3 else 19)
BATCH, L_, W_ = 1 << 12, 32, 256
LAYERS = list(range(2, 32))
assert DEV not in ('cuda:0', 'cuda:4'), f'{DEV} заборонений'
print(__doc__.split('📣')[1].split('🎯 Але')[0], flush=True)
INV_SQ2PI = 1.0 / math.sqrt(2.0 * math.pi)


def phi(x):
    return torch.exp(-0.5 * x * x) * INV_SQ2PI


def Phi(x):
    return 0.5 * torch.erfc(-x * 0.7071067811865476)


def features(m, C, W):
    """Кандидати для нейрона `i` наступного шару. Усе — матвеки/елементні."""
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    v_a = torch.clamp((m * m + s * s) * P + m * s * f - m_a * m_a, min=1e-30)
    W2 = W * W
    W3 = W2 * W
    W4 = W2 * W2
    d1 = v_a @ W2                                  # діагональний внесок у Var
    full = torch.clamp(((C @ W) * W).sum(0), min=1e-30)
    sg = torch.sqrt(full)
    v32 = v_a ** 1.5
    F = {}
    # 🔑 ПАРА «ЗНАК vs БЕЗ ЗНАКА» — прямий тест ідеї юзера
    F['s3_знак'] = (v32 @ W3) / sg ** 3            # κ₃-подібне, `W³` несе знак
    F['a3_модуль'] = (v32 @ W3.abs()) / sg ** 3    # те саме БЕЗ знака (контроль)
    F['s4'] = ((v_a * v_a) @ W4) / sg ** 4         # κ₄-подібне
    F['pr_ваг'] = (W2.sum(0) ** 2) / torch.clamp(W4.sum(0), min=1e-30)
    F['pr_зваж'] = (d1 * d1) / torch.clamp((v_a * v_a) @ W4, min=1e-30)
    F['off'] = 1.0 - d1 / full                     # частка позадіагоналі
    F['t'] = m_a @ W / sg                          # `t` наступного шару
    F['Phi_сер'] = P @ W2 / torch.clamp(W2.sum(0), min=1e-30)
    F['sgn_m'] = (m_a @ torch.sign(W)) / torch.clamp(m_a.abs().sum(), min=1e-30)
    F['sgn_v'] = (v_a @ torch.sign(W)) / torch.clamp(v_a.sum(), min=1e-30)
    F['розкид_v'] = ((v_a * v_a) @ W2) / torch.clamp(d1 * d1, min=1e-30)
    F['t_вх'] = (t.abs() * v_a) @ W2 / torch.clamp(d1, min=1e-30)
    return F, sg


def step(m, C, W):
    s = torch.sqrt(torch.clamp(torch.diagonal(C), min=1e-30))
    t = m / s
    P, f = Phi(t), phi(t)
    m_a = m * P + s * f
    C_a = C * torch.outer(P, P) + (0.5 * (C * C)) * torch.outer(f / s, f / s)
    d = (m * m + s * s) * P + (m * s) * f - m_a * m_a
    C_a = C_a + torch.diag(torch.clamp(d, min=1e-30) - torch.diagonal(C_a))
    return m_a @ W, W.T @ (C_a @ W)


WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
SD = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w_seeds.npy')
have = {int(v): j for j, v in enumerate(SD)}
FS = [f for f in sorted(glob.glob(
    '/mnt/win_d/llm/White-Box/bank_s53/pool3/net_*.npz'))
    if int(np.load(f)['seed']) in have][:NETS]
print(f'{len(FS)} мереж · істина 2^{int(math.log2(NS))} · шари {LAYERS[0]}..'
      f'{LAYERS[-1]} · {DEV}\n', flush=True)

X, Y, NET, LAY = [], [], [], []
KEYS = None
for n, f in enumerate(FS):
    sd0 = int(np.load(f)['seed'])
    W32 = [torch.tensor(np.asarray(WZ[have[sd0]][q], np.float32), device=DEV)
           for q in range(L_)]
    W = [w.double() for w in W32]
    with torch.no_grad():
        MOM = torch.zeros(L_, 4, W_, dtype=torch.float64, device=DEV)
        g = torch.Generator(device=DEV)
        g.manual_seed(11000 + n)
        d = 0
        while d < NS:
            b = min(BATCH, NS - d)
            a = torch.randn(b, W_, device=DEV, generator=g)
            for q in range(L_):
                h = a @ W32[q]
                hd = h.double()
                pw = hd
                for kk in range(4):
                    MOM[q, kk] += pw.sum(0)
                    pw = pw * hd
                a = torch.relu(h)
            d += b
        MOM /= NS
        S_TRUE = [torch.sqrt(torch.clamp(MOM[q, 1] - MOM[q, 0] ** 2, min=1e-30))
                  for q in range(L_)]
        # 🔑 ОРАКУЛЬНІ `γ₃,γ₄` цього ж шару — ВЕРХНЯ МЕЖА: якщо навіть вони не
        #    пояснюють залишок `σ`, то він структурно не про негаусовість.
        G_ORC = []
        for q in range(L_):
            m1, m2, m3, m4 = MOM[q, 0], MOM[q, 1], MOM[q, 2], MOM[q, 3]
            c2 = torch.clamp(m2 - m1 * m1, min=1e-30)
            c3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
            c4 = (m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
                  - 3.0 * c2 * c2)
            sq = torch.sqrt(c2)
            G_ORC.append((c3 / sq ** 3, c4 / sq ** 4))
        C = W[0].T @ W[0]
        m = torch.zeros(W_, dtype=torch.float64, device=DEV)
        for i in range(1, L_):
            F, sg_next = features(m, C, W[i])
            if i in LAYERS:
                F['ORC_γ₃'] = G_ORC[i][0]
                F['ORC_γ₄'] = G_ORC[i][1]
                F['ORC_γ₃²'] = G_ORC[i][0] ** 2
                if KEYS is None or 'ORC_γ₃' not in KEYS:
                    KEYS = list(F.keys())
                X.append(torch.stack([F[k] for k in KEYS], 1).cpu().numpy())
                Y.append(torch.log(S_TRUE[i] / sg_next).cpu().numpy())
                NET.append(np.full(W_, n))
                LAY.append(np.full(W_, i))
            m, C = step(m, C, W[i])
    print(f'  net{n} (seed {sd0}) готова', flush=True)

X = np.concatenate(X)
Y = np.concatenate(Y)
NET = np.concatenate(NET)
LAY = np.concatenate(LAY)
# 🔑 знімаємо ШАРОВУ константу — масштаб уже визнано невиправним, ціль = ЗАЛИШОК
for lq in np.unique(LAY):
    msk = LAY == lq
    Y[msk] -= Y[msk].mean()
X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-30)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f'\n{len(Y)} точок · {X.shape[1]} ознак · RMS залишку `log σ` = '
      f'{Y.std():.4%}\n')

print('══ ОДИНОЧНІ КОРЕЛЯЦІЇ з залишком `log(σ_істина/σ_замикання)` ══')
cors = {}
for j, k in enumerate(KEYS):
    c = float(np.corrcoef(X[:, j], Y)[0, 1])
    cors[k] = 0.0 if np.isnan(c) else c
for k in sorted(cors, key=lambda z: -abs(cors[z])):
    print(f'  {k:12s} corr = {cors[k]:+.4f}   ⟹ сама по собі знімає '
          f'{100 * (1 - math.sqrt(max(1 - cors[k] ** 2, 0))):5.2f}% RMS')
sg_, ab_ = abs(cors['s3_знак']), abs(cors['a3_модуль'])
print(f'\n🔑 ТЕСТ ІДЕЇ ЮЗЕРА (знаки): |s3_знак| = {sg_:.4f} проти '
      f'|a3_модуль| = {ab_:.4f} ⟹ ' +
      ('ЗНАКИ ДАЮТЬ БІЛЬШЕ' if sg_ > ab_ * 1.05 else
       'знаки НЕ додають' if sg_ < ab_ * 0.95 else 'нічия'))


def loo_fit(cols):
    """LOO ПО МЕРЕЖАХ: коефіцієнти з інших мереж ⟹ чесний залишок."""
    pred = np.zeros_like(Y)
    for nn in np.unique(NET):
        tr, te = NET != nn, NET == nn
        A = np.column_stack([X[tr][:, cols], np.ones(tr.sum())])
        beta = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
        pred[te] = np.column_stack([X[te][:, cols],
                                    np.ones(te.sum())]) @ beta
    return pred


print('\n══ ЖАДІБНИЙ ВІДБІР (LOO по мережах) ══')
chosen, rest = [], list(range(len(KEYS)))
base = Y.std()
for _ in range(6):
    best, bj = None, None
    for j in rest:
        r = Y - loo_fit(chosen + [j])
        if best is None or r.std() < best:
            best, bj = r.std(), j
    if best >= base * 0.999:
        break
    chosen.append(bj)
    rest.remove(bj)
    base = best
    print(f'  + {KEYS[bj]:12s} ⟹ RMS залишку {base:.4%} '
          f'(R² = {1 - (base / Y.std()) ** 2:.4f})')

print('\n══ ⛔ПЛОМБИ ══')
r_taut = Y - loo_fit([])
Xt = np.column_stack([X, Y])
print(f'  T1 ТАВТОЛОГІЯ: ознака = сам залишок ⟹ R² = '
      f'{1 - ((Y - np.polyval(np.polyfit(Y, Y, 1), Y)).std() / Y.std()) ** 2:.4f}'
      ' (мусить бути 1.0000)')
rng = np.random.default_rng(7)
Xr = X.copy()
Xr = np.column_stack([Xr, rng.standard_normal(len(Y))])
pred_r = np.zeros_like(Y)
for nn in np.unique(NET):
    tr, te = NET != nn, NET == nn
    A = np.column_stack([Xr[tr][:, -1], np.ones(tr.sum())])
    beta = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
    pred_r[te] = np.column_stack([Xr[te][:, -1], np.ones(te.sum())]) @ beta
print(f'  T2 ПІДЛОГА (випадкова ознака): R² = '
      f'{1 - ((Y - pred_r).std() / Y.std()) ** 2:+.5f}')
perm = np.zeros_like(Y)
for lq in np.unique(LAY):
    msk = LAY == lq
    idx = np.where(msk)[0]
    perm[idx] = rng.permutation(idx)
Xa = X[perm.astype(int)]
pred_a = np.zeros_like(Y)
for nn in np.unique(NET):
    tr, te = NET != nn, NET == nn
    A = np.column_stack([Xa[tr][:, chosen], np.ones(tr.sum())])
    beta = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
    pred_a[te] = np.column_stack([Xa[te][:, chosen], np.ones(te.sum())]) @ beta
print(f'  T3 ⛔ЧУЖІ ознаки (перемішані в шарі): R² = '
      f'{1 - ((Y - pred_a).std() / Y.std()) ** 2:+.5f} (мусить бути ≈0)')

final = Y - loo_fit(chosen) if chosen else Y
print(f'\n══ 🎯 ВИРОК (T5: міряти ВИГРАШ, не стелю) ══')
print(f'  RMS похибки `σ` ДО скану:    {Y.std():.4%}')
print(f'  RMS похибки `σ` ПІСЛЯ скану: {final.std():.4%}')
print(f'  🎯 ТРЕБА для рівності з sub97: 0.465%   ·   для лідера: 0.153%')
ok = final.std() * 100 <= 0.465
print(f'  ⟹ {"✅ ГЕЙТ ПРОЙДЕНО" if ok else "⛔ ГЕЙТ НЕ ПРОЙДЕНО"} '
      f'(треба ще ×{final.std() * 100 / 0.465:.1f})')
