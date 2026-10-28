"""Другий бік гейта: КІЛЬКІСТЬ ВИКЛИКІВ (це residual, податок на виклик ~3.3 мкс).
Реальна форма нашого циклу: x(2n x cut) @ W(cut x cut), 2n~10000, cut~200.
ПЕРЕДБАЧЕННЯ: L=2 дасть ~49 матмулів замість 1 ⟹ +48 викликів на шар,
  x31 шар = +1488 викликів = +5 мс = +0.18% B, проти економії ~2.0% B ⟹ ЧИСТИЙ ПЛЮС.
"""
import flopscope as flops
import flopscope.numpy as fnp

rng = fnp.random.default_rng(0)
ROWS, CUT = 10000, 200


def strassen(a, b, level):
    if level == 0 or a.shape[0] % 2 or a.shape[1] % 2 or b.shape[1] % 2:
        return fnp.matmul(a, b)
    m, k, n = a.shape[0] // 2, a.shape[1] // 2, b.shape[1] // 2
    A11, A12, A21, A22 = a[:m, :k], a[:m, k:], a[m:, :k], a[m:, k:]
    B11, B12, B21, B22 = b[:k, :n], b[:k, n:], b[k:, :n], b[k:, n:]
    r = lambda x, y: strassen(x, y, level - 1)
    M1 = r(A11 + A22, B11 + B22); M2 = r(A21 + A22, B11)
    M3 = r(A11, B12 - B22);       M4 = r(A22, B21 - B11)
    M5 = r(A11 + A12, B22);       M6 = r(A21 - A11, B11 + B12)
    M7 = r(A12 - A22, B21 + B22)
    C11 = M1 + M4 - M5 + M7; C12 = M3 + M5
    C21 = M2 + M4;           C22 = M1 - M2 + M3 + M6
    return fnp.concatenate([fnp.concatenate([C11, C12], axis=1),
                            fnp.concatenate([C21, C22], axis=1)], axis=0)


a = fnp.array(rng.standard_normal((ROWS, CUT)).astype(fnp.float32))
b = fnp.array(rng.standard_normal((CUT, CUT)).astype(fnp.float32))

ref = None
base = None
for L in (0, 1, 2, 3):
    flops.budget_reset()
    with flops.budget(10**14):
        out = strassen(a, b, L)
        d = flops.budget_summary_dict()
    f = d["flops_used"]
    calls = sum(v["calls"] for v in d["operations"].values())
    if L == 0:
        ref, base = out, f
        print(f"L=0 (щільний)  flops {f:>12,}  викликів {calls:>5}")
    else:
        err = float(fnp.max(fnp.abs(out - ref)) / fnp.max(fnp.abs(ref)))
        print(f"L={L}           flops {f:>12,}  викликів {calls:>5}"
              f"  ratio {f/base:.4f}  rel err {err:.2e}")
