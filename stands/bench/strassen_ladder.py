"""Batched (pooled) Strassen ladder under flopscope: billed flops and wall vs level, n=1024 fp32.
Prediction: ms/GF grows with L (additions memory-bound ~290 ms/GF, leaf products ~10-25);
L5 ~ 100-150 (v29 measured 123 on grader), L7 ~ 300-400 (J2W 385?)."""
import time, numpy as np, flopscope as flops, flopscope.numpy as fnp
def smm(X, Y, lev):
    # X (P,m,k) Y (P,k,n) -> (P,m,n); batched Strassen-Winograd-free classic 7-product recursion
    if lev == 0: return X @ Y
    P,m,k = X.shape; n = Y.shape[2]; h,q,v = m//2,k//2,n//2
    A11,A12,A21,A22 = X[:,:h,:q],X[:,:h,q:],X[:,h:,:q],X[:,h:,q:]
    B11,B12,B21,B22 = Y[:,:q,:v],Y[:,:q,v:],Y[:,q:,:v],Y[:,q:,v:]
    L = fnp.concatenate([A11+A22, A21+A22, A11, A22, A11+A12, A21-A11, A12-A22], axis=0)
    R = fnp.concatenate([B11+B22, B11, B12-B22, B21-B11, B22, B11+B12, B21+B22], axis=0)
    M = smm(L, R, lev-1)
    M1,M2,M3,M4,M5,M6,M7 = (M[i*P:(i+1)*P] for i in range(7))
    top = fnp.concatenate([M1+M4-M5+M7, M3+M5], axis=2); bot = fnp.concatenate([M2+M4, M1-M2+M3+M6], axis=2)
    return fnp.concatenate([top, bot], axis=1)
rng=np.random.default_rng(0); n=1024
A=fnp.asarray(rng.standard_normal((1,n,n)).astype("float32")); B=fnp.asarray(rng.standard_normal((1,n,n)).astype("float32"))
ref=None
for lev in range(0,8):
    smm(A,B,lev)
    flops.budget_reset(); t=time.perf_counter(); reps=2
    for _ in range(reps): C=smm(A,B,lev)
    dt=(time.perf_counter()-t)/reps; f=flops.budget_summary_dict()["flops_used"]/reps
    if ref is None: ref=np.asarray(C)
    err=float(np.abs(np.asarray(C)-ref).max()/np.abs(ref).max())
    print(f"L{lev} leaf {n>>lev:4d}  billed {f:.3e} ({f/2.147e9:.3f} of dense)  wall {dt*1e3:7.1f} ms  ms/GF {dt/f*1e12:6.1f}  err {err:.1e}", flush=True)
