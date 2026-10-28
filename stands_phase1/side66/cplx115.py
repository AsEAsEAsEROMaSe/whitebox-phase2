"""ТАРИФ КОМПЛЕКСНОГО МАТМУЛА: два семпли за ціну одного?

ПЕРЕДБАЧЕННЯ ДО ЗАПУСКУ:
  H1: complex64 тарифікується ×4 проти float32 (4 дійсні множення) — тоді
      ідея мертва, бо ми несемо 2 семпли за 4 ціни = ×2 ГІРШЕ.        (55%)
  H2: complex64 тарифікується ×2 (за байтами) — рівно нуль.           (30%)
  H3: complex64 тарифікується ×1 (за елементами) — ×2 у балі.         (15%)
  Окремо перевіряю float16 і float64, щоб мати опорну шкалу dtype.
"""
import numpy as np
import flopscope
import flopscope.numpy as fnp

N, K, M = 2048, 1024, 1024


def cost(dtype, cdtype=None):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((N, K))
    b = rng.standard_normal((K, M))
    if cdtype:
        a = (a + 1j * rng.standard_normal((N, K))).astype(cdtype)
        b = b.astype(dtype)
    else:
        a = a.astype(dtype)
        b = b.astype(dtype)
    try:
        A, B = fnp.asarray(a), fnp.asarray(b)
        flopscope.budget_reset()
        o = fnp.matmul(A, B)
        d = flopscope.budget_summary_dict()
        return float(d['flops_used']), str(o.dtype)
    except Exception as e:
        return None, 'ERR %s' % type(e).__name__ + ': ' + str(e)[:90]


def elem(dtype):
    rng = np.random.default_rng(1)
    a = fnp.asarray(rng.standard_normal((N, M)).astype(dtype))
    flopscope.budget_reset()
    _ = fnp.maximum(a, 0.0)
    return float(flopscope.budget_summary_dict()['flops_used'])


with flopscope.budget(int(2**48), quiet=True):
    base = None
    print("=== МАТМУЛ %dx%d @ %dx%d ===" % (N, K, K, M))
    for dt in ('float32', 'float64', 'float16'):
        f, od = cost(dt)
        if base is None:
            base = f
        print("  %-10s %14.0f  ×%.4f  out=%s" % (dt, f, f / base, od))
    for cd in ('complex64', 'complex128'):
        f, od = cost('float32', cd)
        if f is None:
            print("  %-10s %s" % (cd, od))
        else:
            print("  %-10s %14.0f  ×%.4f  out=%s  ⟹ на семпл ×%.4f"
                  % (cd, f, f / base, od, f / base / 2.0))
    print("\n=== ЕЛЕМЕНТНА maximum %dx%d ===" % (N, M))
    eb = elem('float32')
    for dt in ('float32', 'float64', 'float16'):
        print("  %-10s %14.0f  ×%.4f" % (dt, elem(dt), elem(dt) / eb))
