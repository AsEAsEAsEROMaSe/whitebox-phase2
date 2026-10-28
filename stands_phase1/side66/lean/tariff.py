import flopscope as flops, flopscope.numpy as fnp, time
n, r, k = 1024, 384, 4
f32 = fnp.float32
A = fnp.zeros((n, n), dtype=f32) + 1.0
B = fnp.zeros((n, n), dtype=f32) + 1.0
Q = fnp.zeros((n, r), dtype=f32) + 1.0
F = fnp.zeros((r, n), dtype=f32) + 1.0
S4 = fnp.zeros((k, n, n), dtype=f32) + 1.0
S4b = fnp.zeros((k, n, n), dtype=f32) + 1.0

def bench(name, fn):
    flops.budget_reset()
    t = time.time()
    fn()
    d = flops.budget_summary_dict()["flops_used"]
    w = time.time() - t
    print("%-34s %12.4e  %7.1f ms  %8.1f ms/GF" % (name, d, 1000*w, 1000*w/max(d/1e9, 1e-12)))

bench("matmul n^3",                 lambda: fnp.matmul(A, B))
bench("einsum ij,jk->ik",           lambda: fnp.einsum("ij,jk->ik", A, B))
bench("matmul n x n x r",           lambda: fnp.matmul(A, Q))
bench("matmul (n,r)@(r,n)",         lambda: fnp.matmul(Q, F))
bench("batched k x (n,n)@(n,r)",    lambda: fnp.matmul(S4, Q))
bench("hadamard n^2",               lambda: fnp.multiply(A, B))
bench("hub einsum kij,kcj->ic",     lambda: fnp.einsum("kij,kcj->ic", S4, S4b))
bench("sym: A_sym @ B",             lambda: fnp.matmul(flops.as_symmetric(A, symmetry=(0,1)), B))
bench("sym einsum ij,ia,jb->ab",    lambda: fnp.einsum("ij,ia,jb->ab", flops.as_symmetric(A, symmetry=(0,1)), B, B))
bench("gram einsum ia,ib->ab",      lambda: fnp.einsum("ia,ib->ab", A, A))
bench("qr (n,r)",                   lambda: fnp.linalg.qr(Q))
bench("sum axis=0 of (k,n,n)",      lambda: fnp.sum(S4, axis=0))
