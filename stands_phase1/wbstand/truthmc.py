"""С115-л: БАНК ІСТИННОГО СТАНУ (m І C активацій) на глибоких шарах, половинами.

Мета: відділити «наше m зіпсоване» від «наша C зіпсована». Крива якоря [gacc]
підміняла лише `m`; тут підміняємо ВЕСЬ стан.
🔒 половини вибірки окремо ⟹ якір із A, ціль із B, незалежність за побудовою.
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
LOGN = int(sys.argv[2]) if len(sys.argv) > 2 else 22
KEEP = [8, 10, 12, 14]
CHUNK = 1 << 16
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
NPAIR = (1 << LOGN) // 2
files = sorted([x for x in os.listdir("/home/trim/wblogs") if x.startswith("truthhi_")])[:NNET]
for f in files:
    nm = str(np.load("/home/trim/wblogs/"+f, allow_pickle=True)["name"])
    out = "/home/trim/wblogs/truthmc_%s.npz" % nm
    if os.path.exists(out): log("%s — вже є" % nm); continue
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(16,1024,1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    am = {k: cp.zeros((2,1024), dtype=cp.float64) for k in KEEP}
    aS = {k: cp.zeros((2,1024,1024), dtype=cp.float64) for k in KEEP}
    amf = cp.zeros((2,1024), dtype=cp.float64)
    cnt = np.zeros(2, dtype=np.int64); done = 0; half = 0
    while done < NPAIR:
        b = min(CHUNK, NPAIR-done)
        rs = cp.random.RandomState(seed=880000+7*done//CHUNK)
        z = rs.standard_normal((b,1024), dtype=cp.float32)
        x = cp.concatenate([z, -z], 0)
        for i in range(16):
            x = cp.maximum(x @ Wg[i], 0.0)
            if i in KEEP:
                am[i][half] += x.sum(0, dtype=cp.float64)
                aS[i][half] += (x.T @ x).astype(cp.float64)
            if i == 15: amf[half] += x.sum(0, dtype=cp.float64)
        cnt[half] += 2*b; done += b; half = 1-half
        del x, z
    cg = cp.asarray(cnt)
    M = {k: (am[k]/cg[:,None]).get() for k in KEEP}
    Cv = {}
    for k in KEEP:
        c2 = (aS[k]/cg[:,None,None]).get()
        Cv[k] = np.stack([c2[h] - np.outer(M[k][h], M[k][h]) for h in (0,1)])
    mf = (amf/cg[:,None]).get()
    np.savez("/home/trim/wblogs/tmp_%s.npz" % nm, name=nm, keep=np.array(KEEP), mf=mf,
             **{("m%d" % k): M[k] for k in KEEP}, **{("C%d" % k): Cv[k].astype(np.float32) for k in KEEP})
    os.replace("/home/trim/wblogs/tmp_%s.npz" % nm, out)
    log("%s готово" % nm)
    for g in Wg: del g
    cp.get_default_memory_pool().free_all_blocks()
log("ГОТОВО")
