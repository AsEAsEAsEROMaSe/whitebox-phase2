import time, flopscope as flops, flopscope.numpy as fnp
from flopscope._budget import get_active_budget
with flops.BudgetContext(flop_budget=10**15):
    b = get_active_budget()
    for shape in [(8, 8), (256, 256), (1024, 1024)]:
        a = fnp.ones(shape, dtype=fnp.float32); c = fnp.ones(shape, dtype=fnp.float32); o = fnp.empty(shape, dtype=fnp.float32)
        big = fnp.ones((2, 1, 1024, 1024), dtype=fnp.float32); ob = fnp.empty((2, 1, 512, 512), dtype=fnp.float32)
        for name, fn in [("add out=", lambda: fnp.add(a, c, out=o)), ("add new", lambda: fnp.add(a, c)),
                         ("add на зрізах-чвертях", lambda: fnp.add(big[..., :512, :512], big[..., 512:, 512:], out=ob)),
                         ("copyto", lambda: fnp.copyto(o, a))]:
            K = 3000; t0 = time.perf_counter(); bk0 = b._total_flopscope_backend_time; ov0 = b._total_flopscope_overhead_time
            for _ in range(K): fn()
            wall = time.perf_counter() - t0; bk = b._total_flopscope_backend_time - bk0; ov = b._total_flopscope_overhead_time - ov0
            print("%-12s %-24s на виклик: wall %6.1f мкс  backend %6.1f  overhead %5.1f  RESIDUAL %5.1f мкс" % (shape, name, 1e6*wall/K, 1e6*bk/K, 1e6*ov/K, 1e6*(wall-bk-ov)/K), flush=True)
