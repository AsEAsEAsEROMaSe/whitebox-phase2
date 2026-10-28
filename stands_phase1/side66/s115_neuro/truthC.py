"""БАНК ПОВНОЇ `C`: істинна КОВАРІАЦІЯ передактивацій по шарах.
Закриває застереження до вироку `sm_all`: досі вкидалась лише ДІАГОНАЛЬ,
позадіагональ лишалась нашою ⟹ справжня двомоментна стеля НЕ зміряна.
Стійкість: чекпоінт по мережі, атомарний запис, підхоплення.
"""
import sys, os, time, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
OUT = "/home/trim/wblogs/truthC_%02d.npz"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 6
N, B = 1 << 20, 1 << 13          # 1.05M семплів: для C цього досить (n=1024)
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
log("cupy %s · пристроїв %d · N=%d" % (cp.__version__, cp.cuda.runtime.getDeviceCount(), N))
assert cp.cuda.runtime.getDeviceCount() == 1
tb = pq.read_table(PQ)
for j in range(min(NNET, tb.num_rows)):
    f = OUT % j
    if os.path.exists(f): log("%2d вже є" % j); continue
    nm = tb["mlp_name"][j].as_py()
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16, 1024, 1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    d, n = 16, 1024
    SP = cp.zeros((d, n), cp.float64)
    CC = cp.zeros((d, n, n), cp.float32)      # fp32 акумулятор: 64 МБ
    rng = cp.random.default_rng(1777 + j); done = 0
    while done < N:
        b = min(B, N - done)
        a = rng.standard_normal((b, n), dtype=cp.float32)
        for i in range(d):
            p = a @ Wg[i]
            SP[i] += p.sum(0, dtype=cp.float64)
            CC[i] += (p.T @ p) / N
            a = cp.maximum(p, 0.0)
        done += b
    mp = cp.asnumpy(SP/N); C = cp.asnumpy(CC).astype(np.float64)
    for i in range(d): C[i] -= np.outer(mp[i], mp[i])
    np.savez(f + ".tmp.npz", name=nm, n=N, mp=mp, C=C.astype(np.float32))
    os.replace(f + ".tmp.npz", f)
    del Wg, CC; cp.get_default_memory_pool().free_all_blocks()
    log("%2d %-22s готово" % (j, nm))
log("ГОТОВО")
