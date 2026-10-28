# С111: ЦІНА СЕМПЛА. Штрассен-Віноград на НАШІЙ формі (256 = 2^8, ділиться глибше
# за форму 200 з strassen_gate2). Міряємо ФЛОПИ і КІЛЬКІСТЬ ВИКЛИКІВ окремо:
# флопи -> F, виклики -> residual (податок ~3.3 мкс/виклик), і вони входять у C по-різному.
# ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ: тариф на 256 буде НЕ гіршим за зміряний на 200
# (L=1 x0.886, L=2 x0.796, L=3 x0.735); оптимум за ПОВНИМ C лишиться L=2.
import numpy as np, flopscope, flopscope.numpy as fnp

def cost_and_calls(fn):
    flopscope.budget_reset()
    with flopscope.budget(10**16, quiet=True):
        fn(); d = flopscope.budget_summary_dict()
    return d['flops_used'], sum(v['calls'] for v in d['operations'].values())

def mm(a, b, level):
    """Штрассен-Віноград, рекурсія до level; інакше щільний матмул."""
    if level <= 0:
        return a @ b
    m, k = a.shape; k2, n = b.shape
    if m % 2 or k % 2 or n % 2 or min(m, k, n) < 64:
        return a @ b
    m2, kk, n2 = m // 2, k // 2, n // 2
    a11, a12 = a[:m2, :kk], a[:m2, kk:]
    a21, a22 = a[m2:, :kk], a[m2:, kk:]
    b11, b12 = b[:kk, :n2], b[:kk, n2:]
    b21, b22 = b[kk:, :n2], b[kk:, n2:]
    s1 = a21 + a22; s2 = s1 - a11; s3 = a11 - a21; s4 = a12 - s2
    t1 = b12 - b11; t2 = b22 - t1; t3 = b22 - b12; t4 = t2 - b21
    p1 = mm(a11, b11, level - 1); p2 = mm(a12, b21, level - 1)
    p3 = mm(s4, b22, level - 1);  p4 = mm(a22, t4, level - 1)
    p5 = mm(s1, t1, level - 1);   p6 = mm(s2, t2, level - 1)
    p7 = mm(s3, t3, level - 1)
    u1 = p1 + p2; u2 = p1 + p6; u3 = u2 + p7; u4 = u2 + p5
    u5 = u4 + p3; u6 = u3 - p4; u7 = u3 + p5
    top = fnp.concatenate([u1, u5], axis=1)
    bot = fnp.concatenate([u6, u7], axis=1)
    return fnp.concatenate([top, bot], axis=0)

rng = np.random.default_rng(0)
N = 6144
A = rng.standard_normal((N, 256)).astype(np.float32)
B = (rng.standard_normal((256, 256)) * 0.0884).astype(np.float32)

print(f"=== ШТРАССЕН НА НАШІЙ ФОРМІ {N}x256 @ 256x256, fp32 ===")
base = None
for L in range(0, 5):
    f, ops = cost_and_calls(lambda L=L: mm(fnp.asarray(A), fnp.asarray(B), L))
    if base is None: base = f
    # точність
    r = np.asarray(mm(fnp.asarray(A), fnp.asarray(B), L))
    ref = A @ B
    err = np.abs(r - ref).max() / np.abs(ref).max()
    print(f"  L={L}  флопів {f:>14,}  ×{f/base:.4f}  операцій {ops:>7,}  макс.відн.похибка {err:.2e}")

print()
print("=== ПОВНИЙ ПРОХІД 32 ШАРИ ===")
W = [(rng.standard_normal((256, 256)) * 0.0884).astype(np.float32) for _ in range(32)]
res = {}
for L in [0, 1, 2, 3]:
    def run(L=L):
        h = fnp.asarray(A); z = np.array(0, dtype=np.float32)
        for l in range(32):
            h = fnp.maximum(mm(h, fnp.asarray(W[l]), L), fnp.asarray(z))
    f, ops = cost_and_calls(run)
    res[L] = (f, ops)
    print(f"  L={L}  F = {f:>14,}  ({f/N:>10,.0f}/семпл)  операцій {ops:>8,}")

print()
print("=== ПОВНА ВАРТІСТЬ C = F + 1e11·R, ПОДАТОК 3.3 мкс/операцію ===")
B_BUDGET = 2.72e11; FLOOR = 0.1 * B_BUDGET; TAX = 3.3e-6
F0_CLOSURE = 1.66e9          # замикання, з прайс-листа
R_BASE = 18e-3               # наш нинішній residual, з реєстру per-MLP
f0, ops0 = res[0]
for L in [0, 1, 2, 3]:
    f, ops = res[L]
    p = f / N                                        # ціна семпла (флопи)
    extra_ops = ops - ops0
    F0 = F0_CLOSURE * (f / f0) + 1e11 * (R_BASE + extra_ops * TAX)
    Nmax = (FLOOR - F0) / p
    score = 0.1 * (1.136e-2) * p / (FLOOR - F0) * 1e0 if Nmax > 0 else float('nan')
    print(f"  L={L}  p={p:>10,.0f}  +операцій {extra_ops:>7,}  податок {extra_ops*TAX*1e3:>7.1f} мс"
          f"  F0={F0:.3e}  N_max={Nmax:>8,.0f}  бал≈{0.1*1.136e-2*p/(FLOOR-F0):.4e}")
