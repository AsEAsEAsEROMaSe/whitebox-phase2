"""С115-з: ДЕ НА КРИВІЙ ВІДГУКУ `m` СТОЇТЬ НАШ `_shape_corr`?

Крива [neuro4]: mu=0.11 → ×1.435 у похибці ВИХОДУ (всі шари).
Канал `m` МОНОТОННИЙ (на відміну від `C`/`s`, С115-ж) ⟹ часткове наближення
до істини тут ПРАЦЮЄ. Питання: скільки похибки `m` знімає нинішній _shape_corr
і скільки лишилось.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 _shape_corr знімає 3-10% похибки `m` (еквівалент mu 0.03-0.10)   55%
  P2 знімає >15%                                                      20%
  P3 знімає <3% (тобто виграш С114 був не через точність `m`)         25%
  P4 частка знятого РОСТЕ з глибиною (там і фітувалось)               60%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def chain(W, lf):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); MS = []
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if lf != 0.0: m = m - lf*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        MS.append(m.copy())
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def cer(m, tr):
    a = (m@tr)/max((m*m).sum(), 1e-300); e = a*m - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 24
A = {i: [] for i in range(16)}; B = {i: [] for i in range(16)}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16,1024,1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16,1024)
    M0 = chain(W, 0.0); M1 = chain(W, LAM_FEAT)
    for i in range(16):
        A[i].append(cer(M0[i], T[i])); B[i].append(cer(M1[i], T[i]))
    log("%2d" % j)
print("\n%-4s %10s %10s %8s %10s" % ("ш", "БЕЗ corr%", "З corr%", "×краще", "екв. mu"))
for i in range(16):
    a, b = np.median(A[i]), np.median(B[i])
    print("%-4d %10.4f %10.4f %8.4f %10.4f" % (i, a*100, b*100, a/b, 1.0 - b/a))
json.dump({"a": {k: list(map(float,v)) for k,v in A.items()},
           "b": {k: list(map(float,v)) for k,v in B.items()}},
          open("/home/trim/wblogs/sigm1.json","w"))
log("ГОТОВО")
