"""С115-к: ТОЧНЕ ДВОМОМЕНТНЕ ЗАМИКАННЯ ЧЕРЕЗ СЕМПЛЮВАННЯ.

Питання: скільки з нашої похибки 0.1202% — ЦІНА ДВОХ МОМЕНТІВ, а скільки — вада
формули (Мелер K=2, κ, _shape_corr, α)?

СХЕМА (потокова, хмару не зберігаємо — гаус повністю описується (m,C)):
   m_p = m@W ; C_p = WᵀCW ; L = chol(C_p)
   по чанках:  p = m_p + g@Lᵀ ;  h = relu(p) ;  акумулюємо sum(h), hᵀh
   m, C = з акумуляторів  ⟹  наступний шар
Це і є двомоментне замикання з ТОЧНОЮ мапою моментів ReLU.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 ≈ наша рекурсія (0.10-0.15%) ⟹ формула вже на межі класу        55%
  P2 помітно КРАЩЕ (<0.08%) ⟹ наша формула щось втрачає              25%
  P3 помітно ГІРШЕ (>0.16%) ⟹ наші фітовані поправки реально працюють 30%
🔒 ПЛОМБА: шар 0 мусить збігтися з ‖w‖/√(2π) — там гаус ТОЧНИЙ за побудовою.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq

PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LOGN = int(sys.argv[2]) if len(sys.argv) > 2 else 22
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 5150
CHUNK = 1 << 16
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)

tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
N = 1 << LOGN
res = []
for f in sorted([x for x in os.listdir("/home/trim/wblogs") if x.startswith("truthhi_")]):
    d = np.load("/home/trim/wblogs/"+f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(16,1024,1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    m = cp.zeros(1024, dtype=cp.float64)
    C = cp.eye(1024, dtype=cp.float64)
    lay0 = None
    for i in range(16):
        # 🚨 P104-100 має fp64 у 1/32 від fp32 ⟹ ВСЯ важка арифметика у fp32,
        #    акумулятори у fp64 (похибка чанка ~sqrt(65536)*1e-7 = 2.6e-5 — досить)
        # 🚨 Cp і Холецький — у fp64 (дешеві: 2*1024^3 = 10 мс навіть при fp64
        #    1/32); у fp32 коваріація втрачає додатну визначеність і cholesky
        #    МОВЧКИ віддає nan. Важкий цикл лишається fp32.
        Wi64 = Wg[i].astype(cp.float64)
        mp32 = (m @ Wi64).astype(cp.float32)
        Cp = Wi64.T @ C @ Wi64
        Cp = 0.5*(Cp + Cp.T)
        ev = cp.linalg.eigvalsh(Cp)
        jit = max(0.0, -float(ev[0])) + 1e-9*float(cp.diagonal(Cp).mean())
        L = cp.linalg.cholesky(Cp + cp.eye(1024, dtype=cp.float64)*jit).astype(cp.float32)
        if bool(cp.isnan(L).any()):
            log("  nan у Холецькому на шарі %d (lmin=%.3e)" % (i, float(ev[0]))); break
        am = cp.zeros(1024, dtype=cp.float64); aS = cp.zeros((1024,1024), dtype=cp.float64)
        done = 0
        while done < N:
            b = min(CHUNK, N - done)
            rs = cp.random.RandomState(seed=SEED + 7919*i + (done//CHUNK))
            g = rs.standard_normal((b, 1024), dtype=cp.float32)
            h = cp.maximum(mp32 + g @ L.T, 0.0)
            am += h.sum(0, dtype=cp.float64); aS += (h.T @ h).astype(cp.float64)
            done += b; del g, h
        m = am/N; C = aS/N - cp.outer(m, m)
        if i == 0: lay0 = m.get()
    pred = m.get()
    T = d["m"]; A, B = d["m_half"][:,0,:], d["m_half"][:,1,:]
    def cerr(x, tr):
        a = (x@tr)/max((x*x).sum(), 1e-300); e = a*x - tr
        return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))
    sg0 = np.sqrt((W[0]*W[0]).sum(0))*0.3989422804014327
    seal0 = float(np.sqrt(((lay0 - sg0)**2).sum())/np.sqrt((sg0*sg0).sum()))
    e = cerr(pred, T[15]); eh = 0.5*(cerr(pred, A[15]) + cerr(pred, B[15]))
    res.append((nm, e, eh, seal0))
    log("%-20s похибка %.4f%%  (пів-банк %.4f%%)  🔒шар0 %.2e" % (nm, e*100, eh*100, seal0))
    for gg in Wg: del gg
    cp.get_default_memory_pool().free_all_blocks()
print("\nмедіана похибки точного двомоментного замикання: %.4f%%" % (np.median([r[1] for r in res])*100))
print("🔒 медіана пломби шару 0 (мусить бути ~1e-4): %.2e" % np.median([r[3] for r in res]))
log("ГОТОВО")
