import time, flopscope as flops, flopscope.numpy as fnp
from flopscope._budget import get_active_budget
with flops.BudgetContext(flop_budget=10**15):
    b = get_active_budget()
    big = fnp.ones((2, 1, 1024, 1024), dtype=fnp.float32); ob = fnp.empty((2, 1, 512, 512), dtype=fnp.float32)
    v1 = big[..., :512, :512]; v2 = big[..., 512:, 512:]
    def m(name, fn, K=3000):
        t0 = time.perf_counter(); bk0 = b._total_flopscope_backend_time; ov0 = b._total_flopscope_overhead_time
        for _ in range(K): fn()
        wall = time.perf_counter() - t0; bk = b._total_flopscope_backend_time - bk0; ov = b._total_flopscope_overhead_time - ov0
        print("%-34s residual %5.1f мкс/виклик (backend %6.1f)" % (name, 1e6*(wall-bk-ov)/K, 1e6*bk/K), flush=True)
    m("add, свіжі зрізи", lambda: fnp.add(big[..., :512, :512], big[..., 512:, 512:], out=ob))
    m("add, закешовані view", lambda: fnp.add(v1, v2, out=ob))
    m("лише зріз (без операції)", lambda: big[..., :512, :512])
    m("зріз ob[:, 0]", lambda: ob[:, 0])
