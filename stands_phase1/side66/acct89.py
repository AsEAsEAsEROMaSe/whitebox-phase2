"""acct89 — ТОЧНА БУХГАЛТЕРІЯ ФЛОПІВ `sub97`: куди РЕАЛЬНО йде бюджет.

Питання, без якого хід із симетрії не можна перевести в бал:
  `_TARGET_UTIL = 0.0875` — це частка бюджету на СЕМПЛИ, а `left` НЕ віднімає
  вартість замикання. Отже `F = 0.0875·B + замикання + cov + eigh + whiten`?
  Тоді F/B ≈ 0.0978, і на residual лишається лише 0.0024, а шапка sub97 каже
  0.0125. Одне з двох невірне.

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (шанс 50/50):
  або (а) `budget`, що приходить у `predict`, НЕ дорівнює `B = 2.72e11`,
  або (б) замикання вже якось враховане. Ставлю на (а).

Це чиста бухгалтерія: ЖОДНОЇ математики, лише лічильник.
"""
import sys

import numpy as np

sys.path.insert(0, "/mnt/win_d/llm/White-Box/whest-starterkit")
import flopscope as flops                                    # noqa: E402
import flopscope.numpy as fnp                                # noqa: E402

B = 2.72e11
n, depth = 256, 32
rng = np.random.default_rng(7)


class MLP:
    def __init__(self, ws):
        self.weights = ws
        self.width = ws[0].shape[0]


ws = [fnp.asarray((rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32))
      for _ in range(depth)]
mlp = MLP(ws)

import estimator_sub97 as E                                  # noqa: E402

est = E.Estimator.__new__(E.Estimator)

print("=" * 74)
print("A) ПОВНИЙ ПРОГІН `sub97.predict` ПІД ЛІЧИЛЬНИКОМ")
print("=" * 74)
with flops.BudgetContext(flop_budget=10**13) as ctx:
    out = est.predict(mlp, int(B))
F = ctx.flops_used
print(f"   F = {F:,d}   F/B = {F / B:.5f}   ({100 * F / B:.3f}%)")
print(f"   вихід: shape={np.asarray(out).shape} · перші 3 = {np.asarray(out)[-1][:3]}")

print()
print("=" * 74)
print("B) РОЗКЛАД: ЗАМИКАННЯ ОКРЕМО")
print("=" * 74)
w_cut = ws
with flops.BudgetContext(flop_budget=10**13) as ctx:
    E._closure_parts(w_cut, E._KSIG, n, kdeep=depth)
Fcl = ctx.flops_used
print(f"   замикання = {Fcl:,d}   = {100 * Fcl / B:.4f}% B   = {100 * Fcl / F:.2f}% F")

print()
print("=" * 74)
print("C) СКІЛЬКИ СЕМПЛІВ І ЩО КОШТУЄ cov/eigh/whiten")
print("=" * 74)
per_sample_full = 2.0 * depth * n * n
used = 2.0 * E._PILOT_PAIRS * per_sample_full
per_sample_cut = 2.0 * sum(float(w.shape[0]) * float(w.shape[1]) for w in w_cut)
left = E._TARGET_UTIL * B - used
n_pairs = max(E._MIN_PAIRS, int(left / (2.0 * max(per_sample_cut, 1.0))))
print(f"   per_sample_full = {per_sample_full:,.0f} · пілот used = {used:,.0f}")
print(f"   per_sample_cut  = {per_sample_cut:,.0f}  (тут БЕЗ обрізання ⟹ верхня межа)")
print(f"   n_pairs = {n_pairs:,d}")
z = fnp.asarray(rng.standard_normal((n_pairs, n)).astype(np.float32))
for nm, fn in [("cov = matmul(z.T, z)", lambda: fnp.matmul(z.T, z)),
               ("cov = einsum('ia,ib->ab')", lambda: fnp.einsum("ia,ib->ab", z, z)),
               ("eigh(256)", lambda: fnp.linalg.eigh(fnp.asarray(
                   (np.eye(n) + 0.01 * rng.standard_normal((n, n))).astype(np.float32)
                   @ np.eye(n, dtype=np.float32)))),
               ]:
    with flops.BudgetContext(flop_budget=10**13) as ctx:
        fn()
    print(f"   {nm:32s} {ctx.flops_used:>14,d}  ({100 * ctx.flops_used / B:.4f}% B)")

print()
print("=" * 74)
print("D) ВИСНОВОК ДЛЯ ХОДУ ІЗ СИМЕТРІЄЮ")
print("=" * 74)
save = 521_723_937 + 0.5 * 2.0 * n * n * n_pairs
print(f"   економія симетрії = {save:,.0f} = {100 * save / B:.4f}% B")
print(f"   вона перекладається у семпли: +{100 * save / (F - Fcl):.2f}% ⟹ бал ×?")
print(f"   (потрібно F, а не B, бо семпли = {100 * (F - Fcl) / F:.1f}% від F)")
