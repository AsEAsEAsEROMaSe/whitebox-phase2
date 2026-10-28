"""С115-м: СПЕКТР ДИСПЕРСІЇ ПО СТЕПЕНЯХ. Скільки дає 5-дизайн?

Кубатура, точна до степеня d, знімає ВСІ компоненти степеня <= d ⟹ її виграш
визначається часткою дисперсії у степенях > d. Для 5-дизайну — степені >= 6.

МЕТОД: однорідність out(z) = ‖z‖·out(ẑ) ⟹ задача живе на СФЕРІ. Двоточкова
функція F(u,v) = Σ_a out_a(u)·out_a(v) розкладається по сферичних гармоніках:
   F = Σ_k ‖g_k‖²·R_k(t),  t = u·v,  R_k(1)=1
У високій розмірності s = t√d ~ N(0,1) і R_k ∝ he_k(s) (ортонормовані Ерміта):
   ‖g_k‖² = E[F·he_k(s)] · d^{k/2}/√(k!)
🔒 ПЛОМБА: Σ_k ‖g_k‖² мусить дати E‖out‖² (незалежний шлях до того самого числа).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 непарні 50-60% (антитетика бере)                          70%
  P2 степінь 2 = 15-20% (вибілювання бере)                     65%
  P3 степені >=6 < 3%  ⟹ 5-дизайн дає VR >= ×30                45%
  P4 степені >=6 > 10% ⟹ кубатура дає лише ×5-10               35%
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NPAIR = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 17
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
KMAX = 9
D = 1024
CH = 1 << 13
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)

def he(s, k):
    """ортонормовані поліноми Ерміта he_k = He_k/sqrt(k!)"""
    H = [cp.ones_like(s), s]
    for j in range(1, k): H.append((s*H[j] - j*H[j-1]))
    from math import factorial
    return H[k]/np.sqrt(float(factorial(k)))

res = []
for j in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][j].as_py()
    W = [cp.asarray(x) for x in np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16,1024,1024)]
    accF = cp.zeros(KMAX+1, dtype=cp.float64)
    accN = cp.zeros((), dtype=cp.float64); accP = cp.zeros((), dtype=cp.float64); n = 0
    for off in range(0, NPAIR, CH):
        b = min(CH, NPAIR-off)
        rs = cp.random.RandomState(seed=61000+1000*j+off//CH)
        u = rs.standard_normal((b, D), dtype=cp.float32); u /= cp.linalg.norm(u, axis=1, keepdims=True)
        v = rs.standard_normal((b, D), dtype=cp.float32); v /= cp.linalg.norm(v, axis=1, keepdims=True)
        t = (u*v).sum(1).astype(cp.float64)
        a, c = u, v
        for i in range(16):
            a = cp.maximum(a @ W[i], 0.0); c = cp.maximum(c @ W[i], 0.0)
        F = (a.astype(cp.float64)*c.astype(cp.float64)).sum(1)
        s = t*np.sqrt(float(D))
        for k in range(KMAX+1): accF[k] += (F*he(s, k)).sum()
        accN += (a.astype(cp.float64)**2).sum(); accP += F.sum(); n += b
        del u, v, a, c, F
    from math import factorial
    g = np.array([float(accF[k].get())/n * D**(k/2.0)/np.sqrt(float(factorial(k))) for k in range(KMAX+1)])
    tot = float(accN.get())/n                     # E‖out‖²
    res.append((nm, g, tot))
    log("%-20s  Σg_k %.6g   E‖out‖² %.6g   збіг %.4f" % (nm, g.sum(), tot, g.sum()/tot))
    for w in W: del w
    cp.get_default_memory_pool().free_all_blocks()
G = np.stack([r[1] for r in res]); TOT = np.array([r[2] for r in res])
var = G[:,1:].sum(1)                              # дисперсія = все крім k=0
frac = G[:,1:]/var[:,None]
print("\n%-6s %10s" % ("степінь", "частка%"))
for k in range(1, KMAX+1): print("%-6d %10.3f" % (k, np.median(frac[:,k-1])*100))
print("\n%-22s %10s %10s" % ("правило точне до", "лишок%", "VR"))
for d in (1, 2, 3, 5, 7):
    r = np.median(frac[:, d:].sum(1))
    print("%-22s %10.3f %10.2f" % ("степеня %d" % d, r*100, 1.0/max(r,1e-9)))
print("\n🔒 медіана Σg_k/E‖out‖² = %.4f (мусить бути ≈1)" % np.median(G.sum(1)/TOT))
log("ГОТОВО")
