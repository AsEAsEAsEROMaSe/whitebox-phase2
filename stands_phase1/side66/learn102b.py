"""learn102b — ОСТАННІЙ ВАРІАНТ: `a_k` як ФУНКЦІЯ детермінованих ознак

learn102an показав: спільна КОНСТАНТА `a_k` дає рівно нуль (крос 0.991/1.011)
при оракульній стелі ×1.7177. Лишилось перевірити те, що ЮЗЕР і пропонував:
не константа, а МОДЕЛЬ, що видає `a_k` з ознак мережі.

ОЗНАКИ (усі детерміновані, замикання їх уже має):
   LM  16  λ_k/tr(C) — спектр фінальної коваріації
   CS  16  |v_k·m|/‖m‖ — як сильно кожен напрямок змішування несе сигнал
   EFF  5  eff_rank(C) на шарах 4,8,16,24,31 — ПРОФІЛЬ ЗМІШУВАННЯ
   FRZ  1  ‖m‖²/(‖m‖²+trC) — «наскільки фарба вже одного кольору»
   TQ  15  квантилі t (25/50/75) на тих самих шарах
   NRM  3  ‖m‖, tr C, ‖істина‖ ⟹ БЕРЕМО ЛИШЕ ПЕРШІ ДВА (третє — витік!)
⟹ 55 ознак, 1000 мереж, вирок — КРОС A→B у ПОХИБЦІ (не в R² для a_k).

🚨 ЦІЛЬ МАЄ ВАЖКІ ХВОСТИ (sd a_5 = 13.9 при медіані −0.003) ⟹ фітуємо
робастно: ціль обрізається на 5/95 перцентилях, ознаки стандартизуються,
ридж обов'язковий.
🔒 КОНТРОЛЬ: ті самі ознаки, ПЕРЕМІШАНІ по мережах ⟹ має дати ≤ база.

ВИРОК: крос < ×1.01 ⟹ `a_k` не є функцією нічого, що замикання знає,
і вся гілка закрита остаточно — з обох боків.
"""
import numpy as np

d = np.load("/home/trim/learn102.npz", allow_pickle=True)
G, H, YY = d["GG"], d["HH"], d["YY"]
N, M = G.shape[0], G.shape[1]
K = M - 1
X = np.concatenate([d["LM"], d["CS"], d["EFF"], d["FRZ"][:, None],
                    d["TQ"], d["NRM"][:, :2]], axis=1)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print(f"learn102b · {N} мереж · K={K} · ознак {X.shape[1]}")


def err(n, ar):
    g, h, yy = G[n], H[n], YY[n]
    g00 = g[0, 0]
    if g00 <= 0:
        return 1.0
    b = np.concatenate([[0.0], ar])
    a = b.copy()
    a[0] = (h[0] - g[0] @ b) / g00
    return np.sqrt(max(a @ g @ a - 2.0 * a @ h + yy, 0.0) / yy)


ALL = np.arange(N)
IA, IB = ALL[0::2], ALL[1::2]
base = float(np.mean([err(n, np.zeros(K)) for n in ALL]))

# оракульні a_k по мережі
AO = np.zeros((N, K))
for n in range(N):
    g, h = G[n], H[n]
    a = np.linalg.solve(g + 1e-14 * np.trace(g) / M * np.eye(M), h)
    AO[n] = a[1:] / a[0] if abs(a[0]) > 1e-30 else 0.0
ora = float(np.mean([err(n, AO[n] * 1.0) for n in ALL]))
print(f"база {100*base:.4f}%   оракул {100*ora:.4f}% ×{base/ora:.4f}")

Xs = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
Xs = np.concatenate([Xs, np.ones((N, 1))], axis=1)


def predict(tr, te, kk, lam, shrink):
    P = np.zeros((len(te), K))
    for k in range(kk):
        y = AO[tr, k].copy()
        lo, hi = np.percentile(y, [5, 95])
        y = np.clip(y, lo, hi)
        A_ = Xs[tr]
        w = np.linalg.solve(A_.T @ A_ + lam * np.eye(A_.shape[1]), A_.T @ y)
        P[:, k] = np.clip(Xs[te] @ w, lo, hi)
    return P * shrink


print("\n" + "=" * 72)
print(f"{'K':>3} {'lam':>7} {'shrink':>7} {'in-sample':>11} {'×':>7} "
      f"{'крос A→B':>11} {'×':>7} {'B→A':>7}")
print("-" * 72)
BEST = (1.0, None)
for kk in (1, 4, 8, 16):
    for lam in (10.0, 100.0, 1000.0):
        for sh in (0.3, 1.0):
            Pin = predict(ALL, ALL, kk, lam, sh)
            e_in = float(np.mean([err(n, Pin[i]) for i, n in enumerate(ALL)]))
            Pab = predict(IA, IB, kk, lam, sh)
            e_ab = float(np.mean([err(n, Pab[i]) for i, n in enumerate(IB)]))
            Pba = predict(IB, IA, kk, lam, sh)
            e_ba = float(np.mean([err(n, Pba[i]) for i, n in enumerate(IA)]))
            cross = 0.5 * (base / e_ab + base / e_ba)
            if cross > BEST[0]:
                BEST = (cross, (kk, lam, sh))
            print(f"{kk:>3} {lam:>7.0f} {sh:>7.2f} {100*e_in:>10.4f}% "
                  f"{base/e_in:>7.4f} {100*e_ab:>10.4f}% {base/e_ab:>7.4f} "
                  f"{base/e_ba:>7.4f}")
print("-" * 72)
print(f"НАЙКРАЩЕ (середнє по двох напрямках кросу): ×{BEST[0]:.4f} при {BEST[1]}")

rng = np.random.default_rng(5)
Xp = Xs.copy()
Xp[:, :-1] = Xs[rng.permutation(N), :-1]
Xs_real = Xs
Xs = Xp
kk, lam, sh = BEST[1] if BEST[1] else (8, 100.0, 1.0)
Pab = predict(IA, IB, kk, lam, sh)
e_sh = float(np.mean([err(n, Pab[i]) for i, n in enumerate(IB)]))
Xs = Xs_real
print(f"🔒 ПЕРЕМІШАНІ ОЗНАКИ на тих самих (K,lam,sh): ×{base/e_sh:.4f}")

g = BEST[0]
V, E2 = 1.9227e-06, 2.3385e-05
new = V * (E2 / g ** 2) / (V + E2 / g ** 2)
print(f"\nУ БАЛІ при ×{g:.4f}: {V*E2/(V+E2):.4e} → {new:.4e} "
      f"= ×{(V*E2/(V+E2))/new:.4f}")
