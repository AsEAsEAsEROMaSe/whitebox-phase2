"""С115-л: ЩО САМЕ ЗІПСОВАНО НА ГЛИБИНІ — наше `m` чи наша `C`?

Точне двомоментне замикання; на шарі k підміняємо стан із ПОЛОВИНИ A банку:
  m@k   лише середнє (як gacc)
  C@k   лише коваріація
  mC@k  весь стан ⟹ далі лишається РІВНО (15−k) точних двомоментних кроків
міряємо проти половини B. 🔒 незалежність за побудовою.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 mC@14 ≈ m@14 (C не важить)                          60%  [С115-е: full C = diag]
  P2 mC@14 помітно краще за m@14 (C важить на глибині)    30%
  P3 C@14 сам майже нічого не дає                         70%
  P4 mC@14 лишає 0.03…0.06% — ціна ОДНОГО кроку негаусовості
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
LOGN = int(sys.argv[1]) if len(sys.argv) > 1 else 21
SEED = 2718
CHUNK = 1 << 16
N = 1 << LOGN
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}

def step(Wg, m, C, i, sd):
    W64 = Wg[i].astype(cp.float64)
    mp32 = (m @ W64).astype(cp.float32)
    Cp = W64.T @ C @ W64; Cp = 0.5*(Cp + Cp.T)
    ev = cp.linalg.eigvalsh(Cp)
    jit = max(0.0, -float(ev[0])) + 1e-9*float(cp.diagonal(Cp).mean())
    L = cp.linalg.cholesky(Cp + cp.eye(1024, dtype=cp.float64)*jit).astype(cp.float32)
    am = cp.zeros(1024, dtype=cp.float64); aS = cp.zeros((1024,1024), dtype=cp.float64)
    done = 0
    while done < N:
        b = min(CHUNK, N-done)
        g = cp.random.RandomState(seed=sd+7919*i+(done//CHUNK)).standard_normal((b,1024), dtype=cp.float32)
        h = cp.maximum(mp32 + g @ L.T, 0.0)
        am += h.sum(0, dtype=cp.float64); aS += (h.T @ h).astype(cp.float64)
        done += b; del g, h
    m2 = am/N
    return m2, aS/N - cp.outer(m2, m2)

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

KS = [8, 10, 12, 14]
keys = ["база"] + ["%s@%d" % (t, k) for k in KS for t in ("m", "C", "mC")]
acc = {k: [] for k in keys}
for f in sorted([x for x in os.listdir("/home/trim/wblogs") if x.startswith("truthmc_")]):
    d = np.load("/home/trim/wblogs/"+f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(16,1024,1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    mfA, mfB = d["mf"][0], d["mf"][1]
    st = []
    m = cp.zeros(1024, dtype=cp.float64); C = cp.eye(1024, dtype=cp.float64)
    for i in range(16):
        st.append((m.copy(), C.copy()))
        m, C = step(Wg, m, C, i, SEED)
    acc["база"].append(0.5*(cerr(m.get(), mfB) + cerr(m.get(), mfA)))
    for k in KS:
        mt = cp.asarray(d["m%d" % k][0].astype(np.float64))
        Ct = cp.asarray(d["C%d" % k][0].astype(np.float64))
        for tag in ("m", "C", "mC"):
            m2, C2 = st[k+1]
            m2 = mt.copy() if tag in ("m","mC") else m2.copy()
            C2 = Ct.copy() if tag in ("C","mC") else C2.copy()
            for i in range(k+1, 16): m2, C2 = step(Wg, m2, C2, i, SEED)
            acc["%s@%d" % (tag, k)].append(0.5*(cerr(m2.get(), mfB) + cerr(m2.get(), mfA)))
    log("%s  база %.4f%%  mC@14 %.4f%%" % (nm, acc["база"][-1]*100, acc["mC@14"][-1]*100))
    for g in Wg: del g
    cp.get_default_memory_pool().free_all_blocks()
b = np.median(acc["база"])
print("\n%-7s %10s %8s" % ("якір", "похибка%", "×база"))
for k in keys:
    v = np.median(acc[k]); print("%-7s %10.4f %8.3f" % (k, v*100, b/v))
log("ГОТОВО")
