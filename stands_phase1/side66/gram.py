"""КУТИ МІЖ ГІПЕРПЛОЩИНАМИ: точна arc-cosine формула проти нашого Мелера K=2.

Ідея юзера: 256 ваг нейрона задають ГІПЕРПЛОЩИНУ, і ці площини мають
співставлятись між собою, даючи точне число. Це буквально так — для ПАРИ
нейронів існує замкнена формула без наближень:

    E[relu(u)·relu(v)] = (σᵤσᵥ/2π)·[√(1−ρ²) + ρ·(π/2 + arcsin ρ)]

(перевірка: ρ=1 → σ²/2 ✓; ρ=0 → σᵤσᵥ/2π ✓), де ρ — косинус кута між площинами
у метриці входу.

🚨 НАШ САБМІТ ЦЬОГО НЕ ВИКОРИСТОВУЄ. `estimator_sub87.py:140` бере ряд Мелера
   K=2. Розклад точної формули при малих ρ:
       1/(2π) + ρ/4 + ρ²/(4π) + ρ⁴/(12π) + …
   а Мелер K=2 дає рівно перші ТРИ члени ⟹ похибка O(ρ⁴), тобто ρ⁴/(12π).
   При ρ≈0.06 (типове для iid у 256 вимірах) це 2.6e-6 — мізер. Але кореляції
   активацій РОСТУТЬ вглиб (усі x≥0 ⟹ ρ додатні), і при ρ≈0.5 похибка ~1% —
   підозріло близько до 0.849%, які ми давно приписуємо замиканню.

Стенд рахує ланцюг (m, C) двічі — Мелером K=2 і ТОЧНОЮ формулою — і порівнює
обидва `s_замик` з істинним `s` з великого MC, пошарово.

📣 ПЕРЕДБАЧЕНО: типове |ρ| між нейронами росте з глибиною; на шарах 1–3 різниці
   не буде (ρ малі), а десь із шару 5–8 точна формула має стати помітно кращою.
   🚨 АЛЕ це НЕ обіцяє балу: `field.py` показав, що навіть при ТОЧНИХ (m,s)
   гаусове замикання дає 1.116%, а наша вибірка — 0.177%. Тобто замикання як
   ЗАМІНА вибірки лишається мертвим; сенс лише в тому, що канал δs = ŝ − s_замик
   стає чистішою оцінкою шуму `ŝ` (замикання зміщене БЕЗ шуму, вибірка шумна
   БЕЗ зміщення — див. докстрінг sub87).
"""
import sys

import numpy as np
import torch

DEV = sys.argv[1] if len(sys.argv) > 1 else 'cuda:2'
NETS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
LOGN = int(sys.argv[3]) if len(sys.argv) > 3 else 17
KMAX = int(sys.argv[4]) if len(sys.argv) > 4 else 12      # скільки шарів замикати
W_, D, CH = 256, 32, 1 << 14
S2PI = float(1.0 / np.sqrt(2 * np.pi))
SQ2 = float(np.sqrt(2.0))
PI = float(np.pi)

WZ = np.load('/mnt/win_d/llm/White-Box/arena60/fs_w.npy', mmap_mode='r')
print(__doc__.split('📣')[1], flush=True)


def _truth(W, seed, npairs):
    """Істинні s і середні кореляції |ρ| по шарах."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    Z = [torch.zeros(2, W_, dtype=torch.float64, device=DEV) for _ in range(D)]
    G = [torch.zeros(W_, W_, dtype=torch.float64, device=DEV) for _ in range(KMAX)]
    done = 0
    while done < npairs:
        b = min(CH, npairs - done)
        z = torch.randn(b, W_, device=DEV, generator=gen)
        x = torch.cat([z, -z], 0)
        for L in range(D):
            p = x @ W[L]
            pd = p.double()
            Z[L][0] += pd.sum(0)
            Z[L][1] += (pd * pd).sum(0)
            if L < KMAX:
                G[L] += pd.T @ pd
            x = torch.relu(p)
        done += b
    N = float(2 * npairs)
    s, rho = [], []
    for L in range(D):
        m = Z[L][0] / N
        sd = (Z[L][1] / N - m * m).clamp_min(1e-30).sqrt()
        s.append(sd)
        if L < KMAX:
            cov = G[L] / N - torch.outer(m, m)
            dd = cov.diagonal().clamp_min(1e-30).sqrt()
            r = cov / torch.outer(dd, dd)
            r.fill_diagonal_(0.0)
            rho.append(float(r.abs().mean()))
    return s, rho


TT = {}


def _closure(W, exact):
    """Ланцюг (m, C) на KMAX шарів. exact=False — Мелер K=2 (як у sub87)."""
    m = torch.zeros(W_, dtype=torch.float64, device=DEV)
    C = torch.eye(W_, dtype=torch.float64, device=DEV)
    out = []
    for i in range(KMAX + 1):
        w = W[i].double()
        cp = w.T @ (C @ w)
        s = cp.diagonal().clamp_min(1e-20).sqrt()
        if i >= 1:
            out.append(s.clone())
        mp = m @ w
        t = mp / s
        if i >= 1:
            TT[i - 1] = float(t.abs().mean())
        ph = torch.exp(-0.5 * t * t) * S2PI
        Ph = 0.5 * (1.0 + torch.erf(t / SQ2))
        m = mp * Ph + s * ph
        if exact:
            # 🔑 ТОЧНА arc-cosine — але вона справедлива ЛИШЕ при mp=0.
            #    Тому беремо її для ЦЕНТРОВАНОЇ частини і віднімаємо ВІДПОВІДНЕ
            #    центроване середнє s·φ(0), а зсув доносимо множником Ph⊗Ph, як
            #    у Мелері. Інакше формула неузгоджена сама з собою.
            r = (cp / torch.outer(s, s)).clamp(-1.0, 1.0)
            k = (torch.outer(s, s) / (2 * PI)) * (
                (1 - r * r).clamp_min(0).sqrt() + r * (PI / 2 + torch.asin(r)))
            cc = k - torch.outer(s * S2PI, s * S2PI)     # центрована коваріація
            C = cc * (4.0 * torch.outer(Ph, Ph))         # зсув, як у Мелері
        else:
            C = cp * torch.outer(Ph, Ph) + (0.5 * (cp * cp)) * torch.outer(
                ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + torch.diag(d - C.diagonal())
    return out


acc = np.zeros((KMAX, 4))       # похибка Мелера, точної, |ρ|, |t|=|mp|/s
for n in range(NETS):
    W = [torch.tensor(np.asarray(WZ[n][k], np.float32), device=DEV) for k in range(D)]
    with torch.no_grad():
        st, rho = _truth(W, 91_000 + 13 * n, 1 << LOGN)
        sm = _closure(W, exact=False)
        se = _closure(W, exact=True)
        for L in range(KMAX):
            tr = st[L]
            acc[L, 0] += float(((sm[L] - tr) / tr).abs().mean())
            acc[L, 1] += float(((se[L] - tr) / tr).abs().mean())
            acc[L, 2] += rho[L]
            acc[L, 3] += TT.get(L, 0.0)
    del W
    torch.cuda.empty_cache()
    print(f'  … мережа {n+1}/{NETS}', flush=True)

acc /= NETS
print(f'\n{NETS} мереж, істина 2·2^{LOGN} семплів\n')
print(f'{"шар":>4s} {"сер.|ρ|":>9s} {"сер.|t|":>9s} {"Мелер K=2":>12s} '
      f'{"ТОЧНА":>12s} {"краще у":>9s}')
print('-' * 62)
for L in range(KMAX):
    g = acc[L, 0] / max(acc[L, 1], 1e-30)
    star = ' <<<' if g > 1.2 else ''
    print(f'{L+1:4d} {acc[L,2]:9.4f} {acc[L,3]:9.4f} {acc[L,0]:12.4e} '
          f'{acc[L,1]:12.4e} {g:8.2f}×{star}')
tot = acc[:, 0].sum() / max(acc[:, 1].sum(), 1e-30)
print(f'\n🔑 СУМАРНО точна формула краща у **{tot:.3f}×**')
print(f'   середнє |ρ|: шар 1 = {acc[0,2]:.4f} → шар {KMAX} = {acc[-1,2]:.4f}')
print('🚨 Це точність ЗАМИКАННЯ. Стеля гаусового замикання 1.116% ≫ нашої '
      'вибіркової 0.177%\n   ⟹ як ЗАМІНА мертве; сенс лише в каналі δs = ŝ − s_замик.')
print('📊 БАЛ: 1.954e-07 — вимір прайсує заміну формули')
