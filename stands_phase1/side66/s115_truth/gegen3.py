"""С115-м: РОЗДІЛЕННЯ ВСЕРЕДИНІ «парні >= 4» — від цього залежить, чи варта кубатура.

v2 дала правильні АГРЕГАТИ (непарні 55.1% / степінь 2 17.7% / парні>=4 26.8%,
🔒 відтворює amalgonim 55.8/18.1/26.1 іншим методом), але кинула всю масу парних
на степені 8-9 і занулила 4-6 — артефакт колінеарності мономів.

ЛІКУВАННЯ: F(t) розпадається на ПАРНУ і НЕПАРНУ частини за t:
   F_even(t) = (F(t)+F(-t))/2 = Σ_{k парні} a_k t^k
   F_odd(t)  = (F(t)-F(-t))/2 = Σ_{k непарні} a_k t^k
⟹ удвічі менше параметрів на фіт, обумовленість різко краща.
🔒 ПЛОМБИ: Σa_k = F(1) · невід'ємність · нев'язка фіту.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 степінь 4 забирає більшість «парних >=4» ⟹ 5-дизайн дає VR >= x4   45%
  P2 маса РІВНОМІРНО розмазана по 4,6,8 ⟹ 5-дизайн дає лише x1.3-2      40%
  P3 маса справді на 8+ ⟹ кубатура НЕ допомагає взагалі                 20%
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NPER = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 15
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
D = 1024; CH = 1 << 12
TP = np.array([0.10,0.20,0.30,0.40,0.50,0.58,0.66,0.74,0.80,0.855,0.90,0.935,0.96,0.978,0.99,0.996,1.00])
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)

def nnls(V, y, it=60000):
    sc = np.sqrt((V*V).sum(0)); Vn = V/sc
    a = np.maximum(np.linalg.lstsq(Vn, y, rcond=None)[0], 0.0)
    L = np.linalg.norm(Vn, 2)**2
    for _ in range(it):
        a = np.maximum(a - (Vn.T @ (Vn @ a - y))/L, 0.0)
    return a/sc, np.sqrt(((Vn@a - y)**2).mean())

def Fval(W, t, seed):
    s = 0.0; n = 0
    for off in range(0, NPER, CH):
        b = min(CH, NPER-off)
        rs = cp.random.RandomState(seed=seed+off//CH)
        u = rs.standard_normal((b, D), dtype=cp.float32); u /= cp.linalg.norm(u,axis=1,keepdims=True)
        w = rs.standard_normal((b, D), dtype=cp.float32)
        w -= (w*u).sum(1, keepdims=True)*u; w /= cp.linalg.norm(w,axis=1,keepdims=True)
        v = cp.float32(t)*u + cp.float32(np.sqrt(max(1-t*t,0.0)))*w
        a, c = u, v
        for i in range(16):
            a = cp.maximum(a @ W[i], 0.0); c = cp.maximum(c @ W[i], 0.0)
        s += float((a.astype(cp.float64)*c.astype(cp.float64)).sum()); n += b
        del u, v, w, a, c
    return s/n

KE = [0,2,4,6,8,10,12,14]; KO = [1,3,5,7,9,11,13,15]
res = []
for j in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][j].as_py()
    W = [cp.asarray(x) for x in np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16,1024,1024)]
    Fp = np.array([Fval(W, t, 81000+1000*j+37*i) for i, t in enumerate(TP)])
    Fm = np.array([Fval(W, -t, 91000+1000*j+37*i) for i, t in enumerate(TP)])
    Fe = 0.5*(Fp+Fm); Fo = 0.5*(Fp-Fm)
    ae, re_ = nnls(np.stack([TP**k for k in KE], 1), Fe)
    ao, ro_ = nnls(np.stack([TP**k for k in KO], 1), Fo)
    a = np.zeros(16)
    for i, k in enumerate(KE): a[k] = ae[i]
    for i, k in enumerate(KO): a[k] = ao[i]
    res.append((nm, a, Fp[-1]))
    log("%-20s Σa %.5g  F(1) %.5g  збіг %.4f  нев'язки %.1e/%.1e" %
        (nm, a.sum(), Fp[-1], a.sum()/Fp[-1], re_/abs(Fp[-1]), ro_/abs(Fp[-1])))
    for x in W: del x
    cp.get_default_memory_pool().free_all_blocks()
A = np.stack([r[1] for r in res]); F1 = np.array([r[2] for r in res])
var = A[:,1:].sum(1); fr = A[:,1:]/var[:,None]
print("\n%-8s %10s" % ("степінь", "частка%"))
for k in range(1, 16): print("%-8d %10.3f" % (k, np.median(fr[:,k-1])*100))
print("\nнепарні %.1f%% · степ.2 %.1f%% · степ.4 %.1f%% · степ.6 %.1f%% · степ.8 %.1f%%" % (
    np.median(fr[:,0::2].sum(1))*100, np.median(fr[:,1])*100,
    np.median(fr[:,3])*100, np.median(fr[:,5])*100, np.median(fr[:,7::2].sum(1))*100))
print("\n%-22s %10s %10s" % ("правило точне до", "лишок%", "VR"))
for d in (1,2,3,5,7):
    r = np.median(fr[:, d:].sum(1)); print("%-22s %10.3f %10.2f" % ("степеня %d"%d, r*100, 1.0/max(r,1e-9)))
print("\n🔒 медіана Σa/F(1) = %.4f" % np.median(A.sum(1)/F1))
log("ГОТОВО")
