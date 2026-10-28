"""sym89 — БУХГАЛТЕРІЯ ФЛОПІВ ПЕРЕНОСУ КОВАРІАЦІЇ.

ПИТАННЯ: `w.T @ (C @ w)` проти `einsum("ij,ia,jb->ab", C, w, w)` — чи справді
flopscope тарифікує другий дешевше, бо може ДОВЕСТИ симетрію результату?
(запис у CLAUDE.md §ДАНІ; ніколи не перевірявся НАМИ на живому).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (шанс 45%):
  einsum дешевший на ~25% від пари матмулів (33.5e6 замість 67.1e6 було б −50%,
  але einsum трилінійний ⟹ швидше за все він РОЗГОРТАЄТЬСЯ у ті самі два
  матмули і дасть РІВНО те саме, або навіть ДОРОЖЧЕ).
  Якщо дешевший бодай на 10% — це 0.8% усього бюджету ⟹ +0.9% семплів МК.

ПЛОМБА: обидва шляхи мусять дати ЧИСЛОВО той самий результат (rel < 1e-5).
"""
import shim  # noqa: F401
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

n = 256
rng = np.random.default_rng(0)
W = (rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32)
A = (rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32)
C = (A @ A.T + np.eye(n, dtype=np.float32)).astype(np.float32)
Wf, Cf = fnp.asarray(W), fnp.asarray(C)


def cost(fn, name):
    with flops.BudgetContext(flop_budget=10**13) as ctx:
        r = fn()
    print(f"{name:44s} {ctx.flops_used:>14,d}")
    return np.asarray(r), ctx.flops_used


print("=" * 66)
print("A) ПЕРЕНОС КОВАРІАЦІЇ — ЧОТИРИ ШЛЯХИ ДО ОДНОГО ЧИСЛА")
print("=" * 66)
r1, f1 = cost(lambda: fnp.matmul(Wf.T, fnp.matmul(Cf, Wf)), "matmul(w.T, matmul(C, w))   [САБМІТ]")
r2, f2 = cost(lambda: fnp.einsum("ij,ia,jb->ab", Cf, Wf, Wf), "einsum('ij,ia,jb->ab', C, w, w)")
r3, f3 = cost(lambda: fnp.matmul(fnp.matmul(Wf.T, Cf), Wf), "matmul(matmul(w.T, C), w)")
r4, f4 = cost(lambda: fnp.einsum("ia,ij,jb->ab", Wf, Cf, Wf), "einsum('ia,ij,jb->ab', w, C, w)")

print("\n🔬 ПЛОМБА ЧИСЛОВОЇ ТОТОЖНОСТІ (усі мусять збігтися):")
base = np.abs(r1).max()
for nm, r in [("einsum ij,ia,jb", r2), ("matmul(w.T,C),w", r3), ("einsum ia,ij,jb", r4)]:
    print(f"   |{nm:18s} − сабміт|/max = {np.abs(r - r1).max() / base:.3e}")

print("\n📊 ВІДНОСНА ЦІНА (сабміт = 1.000):")
for nm, f in [("einsum ij,ia,jb", f2), ("matmul(w.T,C),w", f3), ("einsum ia,ij,jb", f4)]:
    print(f"   {nm:20s} {f / f1:.4f}   (Δ = {f - f1:+,d})")

print()
print("=" * 66)
print("B) ЩЕ ТРИ ФОРМИ, ЯКІ МОЖУТЬ ЗНАТИ ПРО СИМЕТРІЮ")
print("=" * 66)
Cs = fnp.asarray(C)
cost(lambda: fnp.matmul(Cf, Wf), "  один матмул 256^3 (опора)")
cost(lambda: fnp.einsum("ij,jb->ib", Cf, Wf), "  einsum одного матмула")
cost(lambda: fnp.einsum("ia,ib->ab", fnp.matmul(Cf, Wf), Wf), "  einsum('ia,ib->ab', C@w, w)  [СИМЕТР. ЯВНО]")

print()
print("=" * 66)
print("C) ПОВНА ЦІНА `_closure_parts` НА 32 ШАРАХ — СКІЛЬКИ ЦЕ ВІД БЮДЖЕТУ")
print("=" * 66)
B = 2.72e11


def _phi(t):
    return fnp.exp(-0.5 * t * t) * 0.3989422804014327


def _Phi(t):
    return 0.5 * (1.0 + fnp.tanh(0.7978845608028654 * (t + 0.044715 * t * t * t)))


def closure(ws, use_einsum):
    m = fnp.zeros(n, dtype=fnp.float32)
    Cc = fnp.asarray(np.eye(n, dtype=np.float32))
    for w in ws:
        if use_einsum:
            cp = fnp.einsum("ij,ia,jb->ab", Cc, w, w)
        else:
            cp = fnp.matmul(w.T, fnp.matmul(Cc, w))
        s = fnp.sqrt(fnp.maximum(fnp.diagonal(cp), 1e-20))
        mp = fnp.matmul(m, w)
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        Cc = cp * fnp.outer(Ph, Ph) + (0.5 * (cp * cp)) * fnp.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        Cc = Cc + fnp.diag(d - fnp.diagonal(Cc))
    return m


ws = [fnp.asarray((rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32))
      for _ in range(32)]
_, fa = cost(lambda: closure(ws, False), "замикання 32 шари, matmul  [САБМІТ]")
_, fb = cost(lambda: closure(ws, True), "замикання 32 шари, einsum")
print(f"\n   частка бюджету (B={B:.3g}, підлога 0.1·B={0.1 * B:.4g}):")
print(f"     matmul : {fa / B:.5f}  = {100 * fa / (0.1 * B):.3f}% ПІДЛОГИ")
print(f"     einsum : {fb / B:.5f}  = {100 * fb / (0.1 * B):.3f}% ПІДЛОГИ")
print(f"   економія: {fa - fb:+,d} флопів = {100 * (fa - fb) / (0.1 * B):.3f}% підлоги")
print(f"\n   ⟹ якщо МК займає решту підлоги, приріст семплів "
      f"≈ {100 * (fa - fb) / (0.1 * B - fa):.3f}%")
