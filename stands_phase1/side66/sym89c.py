"""sym89c — END-TO-END: скільки СПРАВДІ економлять теги симетрії у САБМІТІ.

`sym89b` знайшов дві знижки:
   einsum('ij,ia,jb->ab', Cs, w, w)  ×0.751  (тег на C)          −16.7e6/шар
   einsum('ia,ib->ab', X, X)         ×0.502  (симетрія з тотожності)
але також показав ПАСТКУ: `Cs * as_symmetric(outer(v,v))` коштує 524,543
замість 98,432 — тобто ВАЛІДАЦІЯ тега платна і може з'їсти весь виграш.

ТУТ МІРЯЄМО ПОВНИЙ ЛАНЦЮГ, а не окремі операції.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (шанс 60%):
  чистий виграш замикання буде 20–24% (не 25%), бо симетрія рветься на
  `cp*outer(Ph,Ph)` і її треба або відновлювати (платно), або не відновлювати
  (тоді наступний крок втрачає знижку). Якщо ДРУГИЙ варіант ще дає знижку —
  тег треба ставити лише на ВХІД einsum, і виграш повний.

ПЛОМБА: обидва ланцюги мусять дати той самий `m` (rel < 1e-4 у fp32).
"""
import shim  # noqa: F401
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

n = 256
rng = np.random.default_rng(1)
ws = [fnp.asarray((rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32))
      for _ in range(32)]


def _phi(t):
    return fnp.exp(-0.5 * t * t) * 0.3989422804014327


def _Phi(t):
    return 0.5 * (1.0 + fnp.tanh(0.7978845608028654 * (t + 0.044715 * t * t * t)))


def closure(mode):
    """mode: 'base' | 'tag_each' | 'tag_once'."""
    m = fnp.zeros(n, dtype=fnp.float32)
    C = fnp.asarray(np.eye(n, dtype=np.float32))
    if mode != "base":
        C = flops.as_symmetric(C, symmetry=(0, 1))
    for w in ws:
        if mode == "base":
            cp = fnp.matmul(w.T, fnp.matmul(C, w))
        else:
            cp = fnp.einsum("ij,ia,jb->ab", C, w, w)
        s = fnp.sqrt(fnp.maximum(fnp.diagonal(cp), 1e-20))
        mp = fnp.matmul(m, w)
        t = mp / s
        ph, Ph = _phi(t), _Phi(t)
        m = mp * Ph + s * ph
        C = cp * fnp.outer(Ph, Ph) + (0.5 * (cp * cp)) * fnp.outer(ph / s, ph / s)
        d = (mp * mp + s * s) * Ph + (mp * s) * ph - m * m
        C = C + fnp.diag(d - fnp.diagonal(C))
        if mode == "tag_each":
            # відновлюємо тег ПІСЛЯ розриву — платимо валідацією
            C = flops.as_symmetric(C, symmetry=(0, 1))
    return m


def cost(fn, name):
    with flops.BudgetContext(flop_budget=10**13) as ctx:
        r = fn()
    print(f"{name:44s} {ctx.flops_used:>14,d}")
    return np.asarray(r), ctx.flops_used


print("=" * 78)
print("A) ЗАМИКАННЯ 32 ШАРИ — ТРИ РЕЖИМИ")
print("=" * 78)
r0, f0 = cost(lambda: closure("base"), "base   (сабміт: matmul(w.T, matmul(C,w)))")
r1, f1 = cost(lambda: closure("tag_once"), "tag_once (тег лише на старті, einsum)")
r2, f2 = cost(lambda: closure("tag_each"), "tag_each (тег відновлюється щошару)")
print(f"\n🔬 ПЛОМБИ тотожності `m`: tag_once {np.abs(r1 - r0).max() / np.abs(r0).max():.2e}"
      f" · tag_each {np.abs(r2 - r0).max() / np.abs(r0).max():.2e}")
print(f"📊 tag_once ×{f1 / f0:.4f} (Δ={f1 - f0:+,d}) · tag_each ×{f2 / f0:.4f} (Δ={f2 - f0:+,d})")

print()
print("=" * 78)
print("B) КОВАРІАЦІЯ ВИБІРКИ `zᵀz` — ПРЯМОКУТНА, ЯК У САБМІТІ")
print("=" * 78)
for npair in (1024, 3525):
    z = fnp.asarray(rng.standard_normal((npair, n)).astype(np.float32))
    a, fa = cost(lambda z=z: fnp.matmul(z.T, z), f"  matmul(z.T, z)   n_pairs={npair}")
    b, fb = cost(lambda z=z: fnp.einsum("ia,ib->ab", z, z), f"  einsum('ia,ib->ab')  n_pairs={npair}")
    print(f"     ×{fb / fa:.4f}  пломба {np.abs(b - a).max() / np.abs(a).max():.2e}\n")

print("=" * 78)
print("C) ЩЕ ДВА МІСЦЯ САБМІТА, ДЕ МОЖЕ БУТИ СИМЕТРІЯ")
print("=" * 78)
Wc = ws[0]
cost(lambda: fnp.matmul(Wc * 1.0, Wc), "  matmul(w, w)      (опора: несиметр.)")
cost(lambda: fnp.sum(Wc * Wc, axis=0), "  sum(w*w, axis=0)  (sg0 — уже дешево)")
Cnp = np.eye(n, dtype=np.float32)
Ct = flops.as_symmetric(fnp.asarray(Cnp), symmetry=(0, 1))
cost(lambda: fnp.diagonal(Ct), "  diagonal(Cs)")
cost(lambda: flops.as_symmetric(fnp.asarray(Cnp), symmetry=(0, 1)), "  ЦІНА САМОГО as_symmetric(256x256)")

print()
print("=" * 78)
print("D) ПІДСУМОК У БЮДЖЕТІ")
print("=" * 78)
B = 2.72e11
floor = 0.1 * B
best = min(f1, f2)
save_cl = f0 - best
save_cov = 0.5 * 2.0 * n * n * 3525          # оцінка для cov при n_pairs≈3525
mc = 0.1002 * B - f0
print(f"   замикання base = {f0:,d} ({100 * f0 / floor:.2f}% підлоги)")
print(f"   економія замикання = {save_cl:,.0f} ({100 * save_cl / floor:.3f}% підлоги)")
print(f"   економія cov (оцінка) = {save_cov:,.0f} ({100 * save_cov / floor:.3f}% підлоги)")
tot = save_cl + save_cov
g = tot / mc
print(f"   РАЗОМ {tot:,.0f} = {100 * tot / floor:.3f}% підлоги ⟹ +{100 * g:.2f}% семплів МК")
print(f"   ⟹ V×{1 / (1 + g):.4f} ⟹ бал ≈ {1.873e-07 * (0.9117 / (1 + g) + 0.0883):.4e}"
      f"  (×{1 / (0.9117 / (1 + g) + 0.0883):.4f})")
