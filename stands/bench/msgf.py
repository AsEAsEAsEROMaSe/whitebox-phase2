"""ms per 1e9 billed flops for op classes under flopscope (fs16), 1 thread.
Prediction (before run): fp32 dense 1024 matmul ~10-20 ms/GF; batched 32x32 leaf ~100-200;
elementwise 1024x1024 ~300-1000 (memory-bound, 1 flop/elem); exp ~30-60 (16 flops/elem)."""
import os, time, numpy as np
import flopscope as flops, flopscope.numpy as fnp
n=1024; rng=np.random.default_rng(0)
def bench(name, fn, reps=3):
    fn()  # warm
    flops.budget_reset(); t=time.perf_counter()
    for _ in range(reps): fn()
    dt=time.perf_counter()-t; f=flops.budget_summary_dict()["flops_used"]
    print(f"{name:42s} flops {f/reps:.3e}  wall {dt/reps*1e3:8.1f} ms  => {dt/f*1e9*1e3:8.1f} ms/GF", flush=True)
for dt_ in ("float32","float64"):
    A=fnp.asarray(rng.standard_normal((n,n)).astype(dt_)); B=fnp.asarray(rng.standard_normal((n,n)).astype(dt_))
    bench(f"matmul 1024 {dt_}", lambda: A@B)
    a=A[:256,:256]; b=B[:256,:256]; bench(f"matmul 256 {dt_}", lambda: a@b, 20)
    a=A[:64,:64]; b=B[:64,:64]; bench(f"matmul 64 {dt_}", lambda: a@b, 200)
    a=A[:32,:32]; b=B[:32,:32]; bench(f"matmul 32 {dt_}", lambda: a@b, 500)
    X=fnp.asarray(rng.standard_normal((343,32,32)).astype(dt_)); Y=fnp.asarray(rng.standard_normal((343,32,32)).astype(dt_))
    bench(f"batched matmul 343x(32x32) {dt_}", lambda: X@Y, 20)
    X=fnp.asarray(rng.standard_normal((49,128,128)).astype(dt_)); Y=fnp.asarray(rng.standard_normal((49,128,128)).astype(dt_))
    bench(f"batched matmul 49x(128x128) {dt_}", lambda: X@Y, 10)
    bench(f"elementwise A*B 1024 {dt_}", lambda: A*B, 20)
    bench(f"elementwise A+B 1024 {dt_}", lambda: A+B, 20)
    bench(f"exp 1024 {dt_}", lambda: fnp.exp(A*0.1), 10)
    v=fnp.asarray(rng.standard_normal(n).astype(dt_))
    bench(f"matvec 1024 {dt_}", lambda: A@v, 200)
    bench(f"outer 1024 {dt_}", lambda: fnp.outer(v,v), 50)
    bench(f"einsum ij,ia,jb->ab {dt_} (256)", lambda: fnp.einsum("ij,ia,jb->ab", A[:256,:256], B[:256,:256], B[:256,:256]), 5)
    bench(f"einsum ij,ij->i (row dots) {dt_}", lambda: fnp.einsum("ij,ij->i", A, B), 20)
    bench(f"sum axis {dt_}", lambda: A.sum(axis=1), 50)
    bench(f"(W*W).T@K (n^3+n^2) {dt_}", lambda: (A*A).T@B, 3)
    W3=fnp.asarray(rng.standard_normal((64,64,64)).astype(dt_))
    bench(f"einsum 3-tensor abc,ai,bj->ijc (64) {dt_}", lambda: fnp.einsum("abc,ai,bj->ijc", W3, A[:64,:64], A[:64,:64]), 5)
