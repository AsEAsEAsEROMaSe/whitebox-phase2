"""С115-л: ДВОХ МОМЕНТІВ МАЛО — ЧИ ВИННА ПОВТОРНА ГАУСИЗАЦІЯ?

Усі попередні виміри гаусували стан НА КОЖНОМУ кроці. Тут:
  S0@k  гаусів стан з ІСТИННИМИ (m,C) на шарі k  →  далі ТОЧНЕ СЕМПЛЮВАННЯ
  порівняння з mC@k (гаусів стан + гаусове замикання далі) [gacc2]
різниця = ціна ПОВТОРНОЇ ГАУСИЗАЦІЇ на шарах k+1..15.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 S0@14 ≪ mC@14 (0.052%) ⟹ винна гаусизація останнього кроку   60%
  P2 S0@k мале для ВСІХ k ⟹ двох моментів досить, якщо далі семплити 40%
  P3 S0@8 велике ⟹ двох моментів справді мало вже на глибині 8      35%
🔒 ПЛОМБА: істинний стан (не гаусів) + точне семплювання ⟹ мусить дати ~шум.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
LOGN = int(sys.argv[1]) if len(sys.argv) > 1 else 22
CHUNK = 1 << 16
N = 1 << LOGN
KS = [8, 10, 12, 14]
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))
acc = {k: [] for k in KS}
for f in sorted([x for x in os.listdir("/home/trim/wblogs") if x.startswith("truthmc_")]):
    d = np.load("/home/trim/wblogs/"+f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(16,1024,1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    mfA, mfB = d["mf"][0], d["mf"][1]
    for k in KS:
        mt = cp.asarray(d["m%d" % k][0].astype(np.float64))
        Ct = cp.asarray(d["C%d" % k][0].astype(np.float64))
        Ct = 0.5*(Ct + Ct.T)
        ev = cp.linalg.eigvalsh(Ct)
        jit = max(0.0, -float(ev[0])) + 1e-9*float(cp.diagonal(Ct).mean())
        L = cp.linalg.cholesky(Ct + cp.eye(1024, dtype=cp.float64)*jit).astype(cp.float32)
        m32 = mt.astype(cp.float32)
        am = cp.zeros(1024, dtype=cp.float64); done = 0
        while done < N:
            b = min(CHUNK, N-done)
            g = cp.random.RandomState(seed=4242+k*97+(done//CHUNK)).standard_normal((b,1024), dtype=cp.float32)
            x = m32 + g @ L.T                      # 🔑 гаусів стан із ІСТИННИМИ (m,C)
            for i in range(k+1, 16):               # 🔑 далі ТОЧНЕ семплювання
                x = cp.maximum(x @ Wg[i], 0.0)
            am += x.sum(0, dtype=cp.float64); done += b
            del g, x
        pred = (am/N).get()
        acc[k].append(0.5*(cerr(pred, mfA) + cerr(pred, mfB)))
    log("%-20s S0@8 %.4f%%  S0@14 %.4f%%" % (nm, acc[8][-1]*100, acc[14][-1]*100))
    for g in Wg: del g
    cp.get_default_memory_pool().free_all_blocks()
MC = {8: 0.1175, 10: 0.1072, 12: 0.0859, 14: 0.0519}   # [gacc2] гаусове замикання далі
print("\n%-5s %12s %12s %10s" % ("шар k", "S0 (семпли)%", "mC (замик)%", "×виграш"))
for k in KS:
    v = np.median(acc[k])
    print("%-5d %12.4f %12.4f %10.3f" % (k, v*100, MC[k], MC[k]/(v*100)))
log("ГОТОВО")
