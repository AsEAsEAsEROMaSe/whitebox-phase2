"""learn102an — ВИРОК: чи можна поставити `a_k` ДЕТЕРМІНОВАНО?

Дані: 1000 мереж спліту `full` з ТОЧНОЮ істиною (`final_means`, n=1e9).
Для кожної збережено G = BᵀB, h = Bᵀy, yy = yᵀy, де
B = [m_deep | (v₁·m)v₁ | … | (v₁₆·m)v₁₆] ⟹ похибку будь-якого `a` рахуємо
точно: ‖Ba − y‖² = aᵀGa − 2aᵀh + yy, без перепрогону мереж.

🚨 a₀ ЗАВЖДИ ВІЛЬНИЙ (це `_ai`, масштаб береться даром) ⟹ при фіксованих
a_{1:} мінімізуємо по a₀ аналітично. Так вимір іде на КОСИНУСНІЙ похибці
(закон С101-в), а не на сирій.

СХЕМИ, ВІД НАЙПРОСТІШОЇ:
  base        лише a₀                                  0 параметрів
  const       a_k = c для всіх k                       1
  vec         a_k = c_k (свій на кожен k)              K
  ora         a_k фітується ПО МЕРЕЖІ проти істини     стеля, нереалізовно
Вирок для const/vec — КРОС A→B (парні/непарні мережі), не in-sample.
🔒 контроль: перемішані по мережах a_k (форма зберігається, зв'язок ні).
"""
import numpy as np

d = np.load("/home/trim/learn102.npz", allow_pickle=True)
G, H, YY = d["GG"], d["HH"], d["YY"]
N, M = G.shape[0], G.shape[1]
K = M - 1
print(f"learn102an · {N} мереж · K={K}")


def err(n, ar):
    """відносна похибка мережі n при фіксованих a_{1:}=ar, a₀ ВІЛЬНИЙ"""
    g, h, yy = G[n], H[n], YY[n]
    g00 = g[0, 0]
    if g00 <= 0:
        return 1.0
    b = np.concatenate([[0.0], ar])
    # a₀* = (h₀ − g₀ᵀb) / g₀₀
    a0 = (h[0] - g[0] @ b) / g00
    a = b.copy()
    a[0] = a0
    r = a @ g @ a - 2.0 * a @ h + yy
    return np.sqrt(max(r, 0.0) / yy)


def mean_err(idx, ar):
    return float(np.mean([err(n, ar) for n in idx]))


ALL = np.arange(N)
IA, IB = ALL[0::2], ALL[1::2]
base = mean_err(ALL, np.zeros(K))
print(f"\nбаза (лише a₀): {100*base:.4f}%")

# ---- ora: стеля ----
ora = []
for n in range(N):
    g, h = G[n], H[n]
    try:
        a = np.linalg.solve(g + 1e-14 * np.trace(g) / M * np.eye(M), h)
    except np.linalg.LinAlgError:
        a = np.zeros(M)
    ora.append(err(n, a[1:]))
print(f"ОРАКУЛ (a по мережі): {100*np.mean(ora):.4f}%  ×{base/np.mean(ora):.4f}")


def fit_const(idx, kk):
    grid = np.concatenate([-np.logspace(-1, -5, 60), [0.0], np.logspace(-5, -1, 60)])
    best, bc = 1e9, 0.0
    for c in grid:
        ar = np.zeros(K)
        ar[:kk] = c
        e = mean_err(idx, ar)
        if e < best:
            best, bc = e, c
    return bc


def fit_vec(idx, kk, ridge=0.0):
    """спільний МНК: мінімізуємо Σ_n ‖B_n a_n − y_n‖²/yy_n при спільних a_{1:}"""
    Q = np.zeros((kk, kk))
    r = np.zeros(kk)
    for n in idx:
        g, h, yy = G[n], H[n], YY[n]
        g00 = g[0, 0]
        if g00 <= 0:
            continue
        # проектуємо геть напрямок a₀ (він вільний)
        gs = g[1:kk + 1, 1:kk + 1] - np.outer(g[1:kk + 1, 0], g[0, 1:kk + 1]) / g00
        hs = h[1:kk + 1] - g[1:kk + 1, 0] * h[0] / g00
        Q += gs / yy
        r += hs / yy
    Q = Q + ridge * np.trace(Q) / kk * np.eye(kk)
    return np.linalg.solve(Q, r)


print("\n" + "=" * 74)
print("СХЕМА                     in-sample      ×      крос A→B      ×    B→A")
print("-" * 74)
for kk in (1, 2, 4, 8, 16):
    c_in = fit_const(ALL, kk)
    e_in = mean_err(ALL, np.pad([c_in] * kk, (0, K - kk)))
    ca, cb = fit_const(IA, kk), fit_const(IB, kk)
    e_ab = mean_err(IB, np.pad([ca] * kk, (0, K - kk)))
    e_ba = mean_err(IA, np.pad([cb] * kk, (0, K - kk)))
    print(f"const K={kk:<2} c={c_in:+.5f}  {100*e_in:>9.4f}% {base/e_in:>6.4f}  "
          f"{100*e_ab:>9.4f}% {base/e_ab:>6.4f} {base/e_ba:>6.4f}")

for kk in (1, 2, 4, 8, 16):
    for rg in (1e-3,):
        v_in = fit_vec(ALL, kk, rg)
        e_in = mean_err(ALL, np.pad(v_in, (0, K - kk)))
        va, vb = fit_vec(IA, kk, rg), fit_vec(IB, kk, rg)
        e_ab = mean_err(IB, np.pad(va, (0, K - kk)))
        e_ba = mean_err(IA, np.pad(vb, (0, K - kk)))
        print(f"vec   K={kk:<2}          {100*e_in:>9.4f}% {base/e_in:>6.4f}  "
              f"{100*e_ab:>9.4f}% {base/e_ab:>6.4f} {base/e_ba:>6.4f}")

# 🔒 контроль: коефіцієнти від ПЕРЕМІШАНОГО набору мереж (та сама величина,
#    але зв'язок із конкретною мережею знищено) — має дати ≈базу
rng = np.random.default_rng(11)
v8 = fit_vec(IA, 8, 1e-3)
sc = rng.permutation(v8)
e_sh = mean_err(IB, np.pad(sc, (0, K - 8)))
print("-" * 74)
print(f"🔒 ПЕРЕМІШАНІ коефіцієнти (K=8, крос): {100*e_sh:>9.4f}% ×{base/e_sh:.4f}")

best_v = fit_vec(ALL, 8, 1e-3)
print(f"\nнайкращі спільні a_k (K=8): " + " ".join(f"{x:+.5f}" for x in best_v))
e_best = mean_err(ALL, np.pad(best_v, (0, K - 8)))
g = base / e_best
V, E2 = 1.9227e-06, 2.3385e-05
new = V * (E2 / g ** 2) / (V + E2 / g ** 2)
print(f"У БАЛІ при ×{g:.4f}: MSE {V*E2/(V+E2):.4e} → {new:.4e} "
      f"= ×{(V*E2/(V+E2))/new:.4f}")
