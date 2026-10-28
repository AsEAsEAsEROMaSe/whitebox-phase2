"""lamval92 — ЧЕСНА ПЕРЕВІРКА λ-РОЗШИРЕННЯ: ПОДІЛ НАВПІЛ, ГРЕБІНЬ ЛИШЕ НА ТРЕНІ.

════════════════ ДВІ ПІДОЗРИ, ЯКІ ЗНІМАЄ ЦЕЙ СТЕНД ════════════════
`lamana92` дав ×1.051 над сабмітом, але:
1. 🚨 ГРЕБІНЬ обирався мінімізацією ТІЄЇ САМОЇ LOO-похибки (9 значень) ⟹
   зміщення відбору. Тут гребінь обирається ЛИШЕ на треновій половині.
2. 🚨 LOO по мережах усе одно бачить усі 600 мереж у виборі СТРУКТУРИ
   (які шари/родини). Тут структура фіксується наперед, а число дається
   на мережах, яких фіт не бачив ЖОДНОГО РАЗУ.
⟹ поділ 300/300: `λ` і гребінь із першої половини, MSE — на другій.
Плюс дзеркальний прогін (2↔1) і середнє — щоб не залежати від жеребу поділу.

════════════════ ВАРІАНТИ (усі проти ДІЮЧОГО сабміту) ════════════════
```
S0  сабміт як є (λ зашиті, 4 родини, шари 1..8)          база
V1  ПЕРЕФІТ тих самих 4 родин на 1..8                    ціна 0 (лише константи)
V2  4 родини на 1..31                                    ціна ~0.12% B
V3  лише δE[a²] на 1..31                                 ціна ~0.055% B
V4  δE[a²] на 1..31 + решта 3 родин на 1..8              ціна ~0.07% B
V5  3 родини (без δмаски) на 1..31                       ціна ~0.09% B
⛔F  чужа матриця R                                       контроль
```

════════════════ ГЕЙТ ОКУПНОСТІ — ЯВНО ════════════════
Стоїмо на підлозі множника ⟹ ціна `c` (у частках B) віднімається з util
0.0817: `n_pairs` падає на `c/0.0817`, `V` росте так само, і
```
MSE(V→V·(1+u)) / MSE(V) = (1+u)·(V+E²)/(V(1+u)+E²),  V=2.050e-6, E²=2.118e-5
```
Чистий виграш = брутто / цей множник. Рахую для кожного варіанта.

════════════════ ПЛОМБИ ════════════════
P1 ⛔ЧУЖА мусить дати MSE = «λ вимкнені» (±3%).
P2 V1 із гребенем→∞ мусить зійтись до «λ вимкнені», не до сабміта.
P3 дзеркальні половини мусять дати той самий знак; розкид між ними — це
   і є чесна похибка виміру.
P4 сума ціни: V2 мусить коштувати рівно вчетверо більше за V3 у редукціях.
"""
import numpy as np

d = np.load("/mnt/win_d/llm/White-Box/arena62/logs/lamdeep92.npz")
BASE, ZERO, DEEPM = d["BASE"], d["ZERO"], d["DEEPM"]
TRUTH, RR = d["TRUTH"], d["RR"]
Q = len(TRUTH)
W_B, AL = 0.0883, -0.00751
V_, E2_, UTIL = 2.050e-06, 2.118e-05, 0.0817
ours0 = (ZERO - W_B * (1.0 + AL) * DEEPM) / (1.0 - W_B)
const = (1.0 - W_B) * ours0 + W_B * (1.0 + AL) * DEEPM
TGT = (const - TRUTH).reshape(Q, 256)
DES = -(1.0 - W_B) * RR
RIDGE = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0)


def cols(layers, fams=(0, 1, 2, 3)):
    return [4 * k + f for k in layers for f in fams]


VAR = (
    ('V1 перефіт 4×(1..8)', cols(range(8)), 0.0),
    ('V2 4×(1..31)', cols(range(31)), 0.00120),
    ('V3 лише δE[a²]×(1..31)', cols(range(31), (2,)), 0.00055),
    ('V4 δE[a²]×31 + 3×(1..8)', sorted(set(cols(range(31), (2,)))
                                       | set(cols(range(8), (0, 1, 3)))), 0.00070),
    ('V5 3 родини ×(1..31)', cols(range(31), (0, 1, 2)), 0.00090),
)


def fit_eval(cols_, tr_idx, te_idx, ridge, foreign=False):
    D = DES[:, cols_, :]
    if foreign:
        D = np.roll(D, 1, axis=0)
    p = len(cols_)
    A = D[tr_idx]
    G = np.einsum('qik,qjk->ij', A, A)
    b = np.einsum('qik,qk->i', A, TGT[tr_idx])
    coef = np.linalg.solve(G + ridge * np.trace(G) / p * np.eye(p), b)
    e_tr = TGT[tr_idx] - np.einsum('qik,i->qk', A, coef)
    e_te = TGT[te_idx] - np.einsum('qik,i->qk', D[te_idx], coef)
    return float((e_tr ** 2).mean()), float((e_te ** 2).mean())


def inner_ridge(cols_, tr_idx, foreign=False):
    """Гребінь — ЛИШЕ на треновій половині, через її власну LOO по мережах."""
    D = DES[:, cols_, :][tr_idx]
    T = TGT[tr_idx]
    p, n = len(cols_), len(tr_idx)
    Gq = np.einsum('qik,qjk->qij', D, D)
    bq = np.einsum('qik,qk->qi', D, T)
    G, b = Gq.sum(0), bq.sum(0)
    out = []
    for r in RIDGE:
        tr = r * np.trace(G) / p
        e = np.zeros((n, 256))
        for q in range(n):
            e[q] = T[q] - D[q].T @ np.linalg.solve(G - Gq[q] + tr * np.eye(p),
                                                   b - bq[q])
        out.append((float((e * e).mean()), r))
    return min(out)[1]


half = Q // 2
SPL = ((np.arange(half), np.arange(half, Q)),
       (np.arange(half, Q), np.arange(half)))
mse_base = np.array([((BASE[te] - TRUTH[te]) ** 2).mean() for _, te in SPL])
mse_zero = np.array([((ZERO[te] - TRUTH[te]) ** 2).mean() for _, te in SPL])

print("=" * 100)
print(f"ЧЕСНИЙ ПОДІЛ {half}/{half} · гребінь лише на треновій половині · "
      f"структура фіксована наперед")
print("=" * 100)
print(f"  сабміт (S0)  MSE половин: {mse_base[0]:.5e} · {mse_base[1]:.5e}")
print(f"  λ вимкнені   MSE половин: {mse_zero[0]:.5e} · {mse_zero[1]:.5e}\n")


def net_gain(gross, cost):
    u = cost / UTIL
    pen = (1 + u) * (V_ + E2_) / (V_ * (1 + u) + E2_)
    return gross / pen, pen


print(f"{'варіант':>26s}{'MSE пол.1':>13s}{'MSE пол.2':>13s}{'брутто ×':>11s}"
      f"{'ціна %B':>10s}{'штраф ×':>10s}{'НЕТТО ×':>10s}{'бал':>12s}")
for nm, cl, cost in VAR:
    te_mse, rg = [], []
    for tr, te in SPL:
        r = inner_ridge(cl, tr)
        rg.append(r)
        te_mse.append(fit_eval(cl, tr, te, r)[1])
    te_mse = np.array(te_mse)
    gross = float((mse_base / te_mse).mean())
    net, pen = net_gain(gross, cost)
    print(f"{nm:>26s}{te_mse[0]:>13.5e}{te_mse[1]:>13.5e}{gross:>11.4f}"
          f"{100*cost:>10.3f}{pen:>10.4f}{net:>10.4f}"
          f"{1.86762e-07/net:>12.4e}   гребінь {rg}")

fm = np.array([fit_eval(cols(range(31)), tr, te, inner_ridge(cols(range(31)), tr,
              True), foreign=True)[1] for tr, te in SPL])
print(f"\n⛔P1 ЧУЖА: {fm[0]:.5e} · {fm[1]:.5e} проти λ вимкнених "
      f"{mse_zero[0]:.5e} · {mse_zero[1]:.5e} "
      f"{'✅' if np.allclose(fm, mse_zero, rtol=0.03) else '🚨'}")
big = fit_eval(cols(range(8)), SPL[0][0], SPL[0][1], 1e9)[1]
print(f"⛔P2 V1 гребінь→∞: {big:.5e} проти λ вимкнених {mse_zero[1]:.5e} "
      f"{'✅' if abs(big/mse_zero[1]-1) < 0.02 else '🚨'}")
