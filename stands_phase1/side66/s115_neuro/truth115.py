"""БАНК ІСТИНИ ФАЗИ 2 на GPU: m_true, s_true (передактивація) по 16 шарах.
Датасет дає ЛИШЕ m (400k). Тут N=2^22 (шум ×3.24 нижче) І s, якої нема ніде.
Шаблон стійкості: чекпоінт після КОЖНОЇ мережі, атомарний запис, підхоплення.
"""
import sys, os, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq

PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
OUT = "/home/trim/wblogs/truth115_%02d.npz"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LOGN = int(sys.argv[2]) if len(sys.argv) > 2 else 22
N, B = 1 << LOGN, 1 << 14
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
log("cupy %s · пристроїв %d · N=2^%d=%d батч %d" % (cp.__version__, cp.cuda.runtime.getDeviceCount(), LOGN, N, B))
assert cp.cuda.runtime.getDeviceCount() == 1, "пінувати по UUID!"

tb = pq.read_table(PQ)
for j in range(min(NNET, tb.num_rows)):
    f = OUT % j
    if os.path.exists(f): log("%2d вже є, пропуск" % j); continue
    nm = tb["mlp_name"][j].as_py()
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16, 1024, 1024)
    Wg = [cp.asarray(W[i]) for i in range(16)]
    d, n = 16, 1024
    sp = cp.zeros((d, n), cp.float64); sp2 = cp.zeros((d, n), cp.float64)
    sr = cp.zeros((d, n), cp.float64)
    rng = cp.random.default_rng(777 + j)
    done = 0
    while done < N:
        b = min(B, N - done)
        a = rng.standard_normal((b, n), dtype=cp.float32)
        for i in range(d):
            p = a @ Wg[i]
            sp[i] += p.sum(0, dtype=cp.float64); sp2[i] += (p.astype(cp.float64)**2).sum(0)
            a = cp.maximum(p, 0.0)
            sr[i] += a.sum(0, dtype=cp.float64)
        done += b
    mp = (sp/N); vp = cp.maximum(sp2/N - mp*mp, 0.0)
    np.savez(f + ".tmp.npz", name=nm, n=N,
             mp=cp.asnumpy(mp), s=cp.asnumpy(cp.sqrt(vp)), m=cp.asnumpy(sr/N))
    os.replace(f + ".tmp.npz", f)
    del Wg; cp.get_default_memory_pool().free_all_blocks()
    log("%2d %-22s готово" % (j, nm))
log("ГОТОВО")
