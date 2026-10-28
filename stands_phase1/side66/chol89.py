"""chol89 — ВИБІЛЮВАННЯ ХОЛЕЦЬКИМ ЗАМІСТЬ `eigh`.

САБМІТ: `eigh(cov)` 150,994,944 + `matmul(evecs*inv_sqrt, evecs.T)` 33,554,432
        = 184.5e6 = 0.0678% B  (ZCA-вибілювання, симетричне)
ЗАМІНА: `cholesky(cov)` 5,592,405 + обернення трикутної
        ⟹ `z @ L^{-T}` теж дає ТОЧНО одиничну емпіричну коваріацію.

📐 ЧОМУ ЦЕ ТА САМА МАТЕМАТИКА, А НЕ НАБЛИЖЕННЯ:
   мета вибілювання — зробити емпіричну `cov` РІВНО `I`. Обидві матриці це
   роблять точно: ZCA `cov^{-1/2}` і Холецька `L^{-T}`. Різниця — поворот,
   а `z ~ N(0,I)` інваріантний до поворотів ⟹ РОЗПОДІЛ той самий.
   ⚠️ АЛЕ вибірка КІНЦЕВА: конкретні реалізації різні ⟹ бал зміниться на
   рівні МК-шуму. Це НЕ бітова пломба, а статистична.

ПЕРЕДБАЧЕННЯ (шанс 65%): економія ~0.05% B ⟹ +0.6% семплів ⟹ ×1.005.
Ризик: `fnp.linalg.inv` трикутної може коштувати повних 256³ ⟹ перевіряємо
ТРИ шляхи обернення, а не один.

ПЛОМБА: `cov(z@Wᵀ)` мусить дати `I` з тією ж точністю, що й ZCA.
"""
import numpy as np

import flopscope as flops
import flopscope.numpy as fnp

B = 2.72e11
n = 256
rng = np.random.default_rng(3)
npair = 2709
znp = rng.standard_normal((npair, n)).astype(np.float32)
z = fnp.asarray(znp)
covnp = (znp.T @ znp / npair).astype(np.float32)
cov = fnp.asarray(covnp)


def cost(fn, name):
    with flops.BudgetContext(flop_budget=10**13) as c:
        r = fn()
    print(f"{name:48s} {c.flops_used:>13,d}  ({100 * c.flops_used / B:.4f}% B)")
    return r, c.flops_used


print("=" * 80)
print("A) ПОТОЧНИЙ ШЛЯХ (ZCA через eigh)")
print("=" * 80)


def zca():
    ev, U = fnp.linalg.eigh(cov)
    inv = 1.0 / fnp.sqrt(fnp.maximum(ev, 1e-8))
    return fnp.matmul(U * inv, U.T)


Wz, fz = cost(zca, "eigh + matmul(U*inv, U.T)   [САБМІТ]")

print()
print("=" * 80)
print("B) ХОЛЕЦЬКИЙ — ТРИ СПОСОБИ ОБЕРНЕННЯ")
print("=" * 80)


def chol_inv():
    L = fnp.linalg.cholesky(cov)
    return fnp.linalg.inv(L).T


def chol_solve():
    L = fnp.linalg.cholesky(cov)
    return fnp.linalg.solve(L, fnp.eye(n, dtype=fnp.float32)).T


def chol_solve_T():
    L = fnp.linalg.cholesky(cov)
    return fnp.linalg.solve(L.T, fnp.eye(n, dtype=fnp.float32))


W1, f1 = cost(chol_inv, "cholesky + inv(L).T")
W2, f2 = cost(chol_solve, "cholesky + solve(L, I).T")
W3, f3 = cost(chol_solve_T, "cholesky + solve(L.T, I)")

print()
print("=" * 80)
print("C) ПЛОМБА: ЧИ СПРАВДІ ВИБІЛЮЄ (cov(z@Wᵀ) = I)?")
print("=" * 80)
for nm, Wm in (("ZCA (eigh)", Wz), ("chol inv", W1), ("chol solve", W2),
               ("chol solve.T", W3)):
    Wa = np.asarray(Wm)
    zw = znp @ Wa.T
    cw = zw.T @ zw / npair
    err = np.abs(cw - np.eye(n)).max()
    print(f"   {nm:16s} max|cov(z@Wᵀ) − I| = {err:.3e}")

print()
print("=" * 80)
print("D) ЩО ЦЕ ДАЄ")
print("=" * 80)
best = min(f1, f2, f3)
save = fz - best
samp = 0.0838 * B
print(f"   найдешевший шлях = {best:,d} проти {fz:,d} ⟹ економія {save:,d}"
      f" = {100 * save / B:.4f}% B")
if save > 0:
    g = save / samp
    prev = 0.001957                                   # нетто С89 від тегів
    gtot = (save / B + prev) * B / samp
    print(f"   сам по собі: +{100 * g:.2f}% семплів")
    print(f"   РАЗОМ із тегами: +{100 * gtot:.2f}% семплів")
    mse = 1.542e-06 / (1 + gtot) + 3.27e-07
    print(f"   ⟹ MSE {mse:.4e} ⟹ бал {mse * 0.1:.4e}  (×{1.873e-07 / (mse * 0.1):.4f})")
else:
    print("   ⛔ ХОЛЕЦЬКИЙ НЕ ДЕШЕВШИЙ — гілка мертва")
