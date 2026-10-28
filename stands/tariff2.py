import flopscope as flops, flopscope.numpy as fnp
n, k = 1024, 15
f32 = fnp.float32
AP = fnp.zeros((k, n, n), dtype=f32) + 1.0
C1 = fnp.zeros((k, n), dtype=f32) + 1.0
R = fnp.zeros((k, n), dtype=f32) + 1.0
Y = fnp.zeros((k, n), dtype=f32) + 1.0
MP = fnp.zeros((k, n, n), dtype=f32)
def bench(name, fn, ref=None):
    flops.budget_reset(); fn(); d = flops.budget_summary_dict()["flops_used"]
    print("%-38s %12.4e %s" % (name, d, "" if ref is None else "  x%.1f від наївного k·n²" % (d / ref)))
kn2 = k * n * n
bench("einsum kij,kj->ki  (R-згортка)", lambda: fnp.einsum("kij,kj->ki", AP, C1), kn2)
bench("matmul AP @ C1[...,None]", lambda: fnp.matmul(AP, C1[:, :, None]), kn2)
bench("einsum ki,ki->i", lambda: fnp.einsum("ki,ki->i", R, Y))
bench("multiply (k,n,n)*(k,1,n) бродкаст", lambda: fnp.multiply(AP, C1[:, None, :], out=MP), kn2)
bench("multiply (k,n,n)*(k,n,1) бродкаст", lambda: fnp.multiply(AP, (Y*0.5)[:, :, None], out=MP), kn2)
bench("add (k,n,n)", lambda: fnp.add(AP, MP, out=MP), kn2)
bench("einsum kij,kij,kij->i (D3)", lambda: fnp.einsum("kij,kij,kij->i", AP, AP, MP), kn2)
bench("einsum kij,kij->i (D3)", lambda: fnp.einsum("kij,kij->i", AP, MP), kn2)
