"""lamana92 — перерахунок `lamdeep92` із ПРАВИЛЬНИМ перекладом у бал + розклади.

🚨 У `lamdeep92` (перший запуск) вирок рахувався формулою `(1+10.33j)/11.33`
ПОВЕРХ уже повного MSE прогнозу — подвійний рахунок. Тут: **виграш на балі =
відношення повних MSE**, бо множник лишається на підлозі 0.1.

Розклади, яких не було:
  * по ГРУПАХ ШАРІВ  (1..8 · 9..16 · 17..24 · 25..31) — де живе інформація
  * по РОДИНАХ       (δσ · δμ · δE[a²] · δмаска) — яка з чотирьох працює
  * НАКОПИЧУВАЛЬНО   (1..8 → 1..16 → 1..24 → 1..31)
"""
import sys

import numpy as np

d = np.load("/mnt/win_d/llm/White-Box/arena62/logs/lamdeep92.npz")
BASE, ZERO, DEEPM = d["BASE"], d["ZERO"], d["DEEPM"]
TRUTH, RR = d["TRUTH"], d["RR"]
Q = len(TRUTH)
W_B, AL = 0.0883, -0.00751
ours0 = (ZERO - W_B * (1.0 + AL) * DEEPM) / (1.0 - W_B)
const = (1.0 - W_B) * ours0 + W_B * (1.0 + AL) * DEEPM
TGT = (const - TRUTH).reshape(Q, 256)
DES = -(1.0 - W_B) * RR
mse_base = float(((BASE - TRUTH) ** 2).mean())
mse_zero = float(((ZERO - TRUTH) ** 2).mean())
z = float((TRUTH * TRUTH).mean())
FAM = ('δσ', 'δμ', 'δE[a²]', 'δмаска')
RIDGE = (1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def fit(cols, ridge, foreign=False):
    D = DES[:, cols, :]
    if foreign:
        D = np.roll(D, 1, axis=0)
    p = len(cols)
    Gq = np.einsum('qik,qjk->qij', D, D)
    bq = np.einsum('qik,qk->qi', D, TGT)
    G, b = Gq.sum(0), bq.sum(0)
    tr = ridge * np.trace(G) / p
    err = np.zeros((Q, 256))
    for q in range(Q):
        err[q] = TGT[q] - D[q].T @ np.linalg.solve(
            G - Gq[q] + tr * np.eye(p), b - bq[q])
    return float((err * err).mean())


def best(cols, foreign=False):
    return min((fit(cols, r, foreign), r) for r in RIDGE)


def cols(layers, fams=range(4)):
    return [4 * k + f for k in layers for f in fams]


print("=" * 92)
print(f"λ-РОДИНИ — ПРАВИЛЬНИЙ ВИРОК · {Q} мереж · LOO по мережах")
print("=" * 92)
print(f"  λ вимкнені            MSE {mse_zero:.6e}  відн {100*np.sqrt(mse_zero/z):.4f}%")
print(f"  λ ЗАШИТІ (сабміт)     MSE {mse_base:.6e}  відн {100*np.sqrt(mse_base/z):.4f}%"
      f"   ⟹ родини дають ×{mse_zero/mse_base:.4f}")
print(f"\n{'варіант':>26s}{'MSE':>14s}{'бал ×':>10s}{'бал':>13s}{'гребінь':>10s}")
for nm, cl in (('FIT 1..8 (як сабміт)', cols(range(8))),
               ('FIT 1..16', cols(range(16))),
               ('FIT 1..24', cols(range(24))),
               ('FIT 1..31 (ідея)', cols(range(31))),
               ('лише 9..31', cols(range(8, 31)))):
    v, r = best(cl)
    print(f"{nm:>26s}{v:>14.5e}{mse_base/v:>10.4f}"
          f"{1.86762e-07*v/mse_base:>13.4e}{r:>10.0e}")
vf, rf = best(cols(range(31)), foreign=True)
print(f"{'⛔ЧУЖА 1..31':>26s}{vf:>14.5e}{mse_base/vf:>10.4f}"
      f"{1.86762e-07*vf/mse_base:>13.4e}{rf:>10.0e}"
      f"   {'✅ = λ вимкнені' if abs(vf/mse_zero-1) < 0.03 else '🚨'}")

print("\n" + "=" * 92)
print("ДЕ ЖИВЕ ІНФОРМАЦІЯ — ПО ГРУПАХ ШАРІВ (кожна ОКРЕМО, поверх λ=0)")
print("=" * 92)
print(f"{'шари':>12s}{'MSE':>14s}{'× над λ=0':>13s}{'× над сабмітом':>17s}")
for lo, hi in ((0, 8), (8, 16), (16, 24), (24, 31)):
    v, _ = best(cols(range(lo, hi)))
    print(f"{f'{lo+1}..{hi}':>12s}{v:>14.5e}{mse_zero/v:>13.4f}{mse_base/v:>17.4f}")

print("\n" + "=" * 92)
print("ЯКА З ЧОТИРЬОХ РОДИН ПРАЦЮЄ (усі 31 шар, родина ОКРЕМО)")
print("=" * 92)
print(f"{'родина':>12s}{'MSE':>14s}{'× над λ=0':>13s}")
for f, nm in enumerate(FAM):
    v, _ = best(cols(range(31), [f]))
    print(f"{nm:>12s}{v:>14.5e}{mse_zero/v:>13.4f}")

v31, _ = best(cols(range(31)))
v8, _ = best(cols(range(8)))
print("\n" + "=" * 92)
print("📣 ВИРОК")
print("=" * 92)
print(f"  розширення 8 → 31 шар:  {v8:.5e} → {v31:.5e}  = ×{v8/v31:.4f}")
print(f"  проти ДІЮЧОГО сабміту:  {mse_base:.5e} → {v31:.5e}  = ×{mse_base/v31:.4f}")
print(f"  ціна ≈ 0.12% B ⟹ треба ×1.0012 лише щоб вийти в нуль")
print(f"  потрібно ×3.66 щоб побити лідера")
