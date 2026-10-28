"""ТАРИФ ШТРАССЕНА НА flopscope 0.12.0, ФОРМА ФАЗИ 2 (1024).

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ (щоб не тлумачити заднім числом):
  У фазі 1 (0.10.0, форма 200) було L=1 ×0.8860, L=2 ×0.7962, L=3 ×0.7350.
  Тариф 0.12.0 рахує ЕЛЕМЕНТНІ операції по 0.5/елемент, а Штрассен додає
  18 додавань розміру (m×k)/(k×n)/(m×n) на рівень + concatenate (рух даних).
  H1: тариф погіршиться, але залишиться <1 — оптимум зсунеться до L=2..3 (55%)
  H2: тариф стане ≈1 або >1 на 1024 — елементні з'їдять усе             (30%)
  H3: тариф не зміниться зовсім (елементні не тарифікуються)            (15%)
  Окремо: residual у фазі 2 НЕ тарифікується ⟹ податок на виклик зник,
  лишається лише кап 0.4 с wall.
"""
import time
import numpy as np
import flopscope
import flopscope.numpy as fnp

_L = 0


def mm(a, b, level):
    if level <= 0 or a.shape[0] % 2 or a.shape[1] % 2 or b.shape[1] % 2:
        return fnp.matmul(a, b)
    m, k, n = a.shape[0] // 2, a.shape[1] // 2, b.shape[1] // 2
    A11, A12, A21, A22 = a[:m, :k], a[:m, k:], a[m:, :k], a[m:, k:]
    B11, B12, B21, B22 = b[:k, :n], b[:k, n:], b[k:, :n], b[k:, n:]
    M1 = mm(A11 + A22, B11 + B22, level - 1)
    M2 = mm(A21 + A22, B11, level - 1)
    M3 = mm(A11, B12 - B22, level - 1)
    M4 = mm(A22, B21 - B11, level - 1)
    M5 = mm(A11 + A12, B22, level - 1)
    M6 = mm(A21 - A11, B11 + B12, level - 1)
    M7 = mm(A12 - A22, B21 + B22, level - 1)
    top = fnp.concatenate([M1 + M4 - M5 + M7, M3 + M5], axis=1)
    bot = fnp.concatenate([M2 + M4, M1 - M2 + M3 + M6], axis=1)
    return fnp.concatenate([top, bot], axis=0)


def mm_block(a, b, level):
    """Той самий Штрассен, але злиття через fnp.block (пам'ять: +18%)."""
    if level <= 0 or a.shape[0] % 2 or a.shape[1] % 2 or b.shape[1] % 2:
        return fnp.matmul(a, b)
    m, k, n = a.shape[0] // 2, a.shape[1] // 2, b.shape[1] // 2
    A11, A12, A21, A22 = a[:m, :k], a[:m, k:], a[m:, :k], a[m:, k:]
    B11, B12, B21, B22 = b[:k, :n], b[:k, n:], b[k:, :n], b[k:, n:]
    M1 = mm_block(A11 + A22, B11 + B22, level - 1)
    M2 = mm_block(A21 + A22, B11, level - 1)
    M3 = mm_block(A11, B12 - B22, level - 1)
    M4 = mm_block(A22, B21 - B11, level - 1)
    M5 = mm_block(A11 + A12, B22, level - 1)
    M6 = mm_block(A21 - A11, B11 + B12, level - 1)
    M7 = mm_block(A12 - A22, B21 + B22, level - 1)
    return fnp.block([[M1 + M4 - M5 + M7, M3 + M5],
                      [M2 + M4, M1 - M2 + M3 + M6]])


def measure(fn, a, b, level):
    flopscope.budget_reset()
    t0 = time.perf_counter()
    out = fn(a, b, level)
    wall = time.perf_counter() - t0
    d = flopscope.budget_summary_dict()
    key = 'flops_used' if 'flops_used' in d else [k for k in d if 'flop' in k][0]
    return out, float(d[key]), wall


def run(N, K, M, dtype):
    rng = np.random.default_rng(0)
    a = fnp.asarray(rng.standard_normal((N, K)).astype(dtype))
    b = fnp.asarray(rng.standard_normal((K, M)).astype(dtype))
    ref, f0, w0 = measure(mm, a, b, 0)
    nrm = float(fnp.max(fnp.abs(ref)))
    print(f"\n### форма {N}x{K} @ {K}x{M}  dtype={dtype}")
    print(f"  L=0  {f0/1e9:10.4f} Гфл  wall {w0*1e3:8.2f} мс  ×1.0000")
    for L in (1, 2, 3, 4):
        if N % (1 << L) or K % (1 << L) or M % (1 << L):
            continue
        for name, fn in (("cat", mm), ("blk", mm_block)):
            o, f, w = measure(fn, a, b, L)
            err = float(fnp.max(fnp.abs(o - ref))) / nrm
            print(f"  L={L} {name}  {f/1e9:10.4f} Гфл  wall {w*1e3:8.2f} мс  "
                  f"×{f/f0:.4f}  похибка {err:.2e}")


if __name__ == "__main__":
    dt = 'float32'
    with flopscope.budget(int(2**48), quiet=True):
        run(4096, 1024, 1024, dt)
        run(8192, 1024, 1024, dt)
        run(1024, 1024, 1024, dt)
