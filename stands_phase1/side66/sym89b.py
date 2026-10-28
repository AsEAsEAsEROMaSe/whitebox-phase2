"""sym89b — ЧИ ДАЄ flopscope ЗНИЖКУ ЗА ДОВЕДЕНУ СИМЕТРІЮ?

`sym89` показав: einsum НЕ дешевший за пару матмулів (Δ = 0 флопів рівно).
Але flopscope 0.8 кинув `SymmetryLossWarning` і показав API `as_symmetric()`.
⟹ знижка існує, але вмикається ТЕГОМ, а не формою запису.

СТАВКА: перенос коваріації = 2.143e9 = 98.9% усього замикання і 7.9% ПІДЛОГИ
бюджету. Якщо симетрія дає −25%, звільняється 0.0020·B ⟹ +2.1% семплів МК
⟹ V ×0.979 ⟹ бал 1.873e-07 → ~1.837e-07 (×1.019). Безкоштовно, без математики.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (шанс 55%):
  тег на `C` сам по собі НЕ здешевить `w.T@C@w` (результат симетричний, але
  flopscope мусить це ДОВЕСТИ через спільні групи операндів, а `w` не симетр.).
  Найімовірніший робочий шлях — форма `X.T @ X` (syrk), де симетрія виводиться
  з ТОТОЖНОСТІ операндів. Тоді через Холецького:
     C = L Lᵀ  ⟹  WᵀCW = (LᵀW)ᵀ(LᵀW)   — chol + 1 матмул + 1 syrk.
"""
import shim  # noqa: F401
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

flops.configure(symmetry_warnings=False)

n = 256
rng = np.random.default_rng(0)
W = (rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32)
A = (rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32)
Cnp = (A @ A.T + np.eye(n, dtype=np.float32)).astype(np.float32)
Wf = fnp.asarray(W)
Cf = fnp.asarray(Cnp)
Cs = flops.as_symmetric(fnp.asarray(Cnp), symmetry=(0, 1))
BASE = 66977792.0


def cost(fn, name, note=""):
    try:
        with flops.BudgetContext(flop_budget=10**13) as ctx:
            r = fn()
        u = ctx.flops_used
        print(f"{name:46s} {u:>14,d}  {u / BASE:6.3f}×  {note}")
        return np.asarray(r), u
    except Exception as e:                                   # noqa: BLE001
        print(f"{name:46s} {'ПОМИЛКА':>14s}  {type(e).__name__}: {str(e)[:44]}")
        return None, None


print("=" * 84)
print("A) ЧИ ЗНАЄ flopscope ПРО СИМЕТРІЮ РЕЗУЛЬТАТУ `XᵀX`?")
print("=" * 84)
cost(lambda: fnp.matmul(Wf, Wf), "matmul(w, w)            [несиметр. результат]")
cost(lambda: fnp.matmul(Wf.T, Wf), "matmul(w.T, w)          [СИМЕТР. результат]")
cost(lambda: fnp.einsum("ia,ib->ab", Wf, Wf), "einsum('ia,ib->ab', w, w)  [той самий]")

print()
print("=" * 84)
print("B) ТЕГ `as_symmetric` НА `C` — ЧИ ЗНИЖУЄ ПЕРЕНОС?")
print("=" * 84)
r0, f0 = cost(lambda: fnp.matmul(Wf.T, fnp.matmul(Cf, Wf)), "БАЗА: matmul(w.T, matmul(C, w))")
cost(lambda: fnp.matmul(Wf.T, fnp.matmul(Cs, Wf)), "тег C: matmul(w.T, matmul(Cs, w))")
cost(lambda: fnp.einsum("ij,ia,jb->ab", Cs, Wf, Wf), "тег C: einsum('ij,ia,jb->ab', Cs, w, w)")

print()
print("=" * 84)
print("C) ХОЛЕЦЬКИЙ: C = L Lᵀ ⟹ WᵀCW = (LᵀW)ᵀ(LᵀW)")
print("=" * 84)
_, fchol = cost(lambda: fnp.linalg.cholesky(Cf), "  cholesky(C) окремо")


def chol_path():
    L = fnp.linalg.cholesky(Cf)
    M = fnp.matmul(L.T, Wf)
    return fnp.matmul(M.T, M)


rC, fC = cost(chol_path, "ПОВНИЙ ШЛЯХ ЧЕРЕЗ ХОЛЕЦЬКОГО")
if rC is not None:
    print(f"   🔬 ПЛОМБА тотожності: {np.abs(rC - r0).max() / np.abs(r0).max():.3e}")

print()
print("=" * 84)
print("D) ЕЛЕМЕНТНІ ОПЕРАЦІЇ НА ТЕГОВАНІЙ МАТРИЦІ")
print("=" * 84)
v = fnp.asarray(np.abs(rng.standard_normal(n)).astype(np.float32) + 0.5)
cost(lambda: Cf * Cf, "  C * C            (без тега)")
cost(lambda: Cs * Cs, "  Cs * Cs          (з тегом)")
cost(lambda: Cf * fnp.outer(v, v), "  C * outer(v,v)   (без тега)")
cost(lambda: Cs * flops.as_symmetric(fnp.outer(v, v), symmetry=(0, 1)),
     "  Cs * outer_s     (обидва теговані)")

print()
print("=" * 84)
print("E) ЩО ЦЕ ДАЄ В БЮДЖЕТІ")
print("=" * 84)
B = 2.72e11
floor = 0.1 * B
mc = 0.1002 * B - 2.1637e9
print(f"   підлога 0.1·B = {floor:.4g} · замикання = 2.164e9 ({100 * 2.1637e9 / floor:.2f}%)")
print(f"   МК = {mc:.4g} ({100 * mc / floor:.2f}% підлоги)")
print("   економія X флопів/шар ⟹ приріст семплів = 32·X/МК:")
for x in (8.4e6, 16.7e6, 25.1e6, 33.5e6):
    g = 32 * x / mc
    print(f"     {x / 1e6:5.1f}e6/шар → +{100 * g:5.2f}% семплів → V×{1 / (1 + g):.4f}"
          f" → бал ≈ {1.873e-07 * (0.9117 / (1 + g) + 0.0883):.4e}")
