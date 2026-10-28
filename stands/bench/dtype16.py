import time, numpy as np, flopscope as flops, flopscope.numpy as fnp
n=1024; rng=np.random.default_rng(0)
def bench(name, fn, reps=2):
    fn(); flops.budget_reset(); t=time.perf_counter()
    for _ in range(reps): fn()
    dt=(time.perf_counter()-t)/reps; f=flops.budget_summary_dict()["flops_used"]/reps
    print(f"{name:34s} billed {f:.3e}  wall {dt*1e3:8.1f} ms  ms/GF {dt/f*1e12:7.1f}", flush=True)
for dt_ in ("float16","float32","float64"):
    A=fnp.asarray(rng.standard_normal((n,n)).astype(dt_)); B=fnp.asarray(rng.standard_normal((n,n)).astype(dt_))
    try: bench(f"matmul 1024 {dt_}", lambda: A@B)
    except Exception as e: print(dt_, "matmul ERR", type(e).__name__, str(e)[:120])
    try: bench(f"A*B {dt_}", lambda: A*B, 10)
    except Exception as e: print(dt_, "mul ERR", type(e).__name__, str(e)[:120])
    try: bench(f"einsum ij,jk {dt_}", lambda: fnp.einsum("ij,jk->ik", A, B))
    except Exception as e: print(dt_, "einsum ERR", type(e).__name__, str(e)[:120])
    try: bench(f"exp {dt_}", lambda: fnp.exp(A*0.1), 5)
    except Exception as e: print(dt_, "exp ERR", type(e).__name__, str(e)[:120])
for dt_ in ("int8","int16","int32"):
    try:
        A=fnp.asarray(rng.integers(-3,3,(n,n)).astype(dt_)); B=fnp.asarray(rng.integers(-3,3,(n,n)).astype(dt_))
        bench(f"matmul 1024 {dt_}", lambda: A@B)
    except Exception as e: print(dt_, "ERR", type(e).__name__, str(e)[:120])
