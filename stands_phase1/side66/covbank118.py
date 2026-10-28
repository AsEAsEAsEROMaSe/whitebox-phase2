"""С118 — БАНК ПОВНИХ (μ_l, Cov(h_l)) НА ВСІХ ШАРАХ, 8 мереж × 2²⁰, fp64.

h_0 = z (відомо: μ=0, Σ=I), h_{l+1} = ReLU(h_l W_l), l = 0..15; зберігаємо l = 1..16
(h_16 = вихід). Потрібен для inj118: точний залишок кроку ОБОХ каналів
   Δm_l = μ_{l+1} − step_m(μ_l, Σ_l) ,  ΔC_l = Σ_{l+1} − step_C(μ_l, Σ_l)
Чекпоінт на мережу: /home/trim/wblogs/covbank118_<net>.npz (mu_l, S_l).
"""
import sys, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 15
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def run(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    mu = [cp.zeros(NW, cp.float64) for _ in range(D+1)]
    S = [cp.zeros((NW, NW), cp.float64) for _ in range(D+1)]
    done, b = 0, 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        x = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
        for l in range(D):
            x = cp.maximum(x @ Wg[l], 0.0)
            xd = x.astype(cp.float64)
            mu[l+1] += xd.sum(0); S[l+1] += xd.T @ xd
            del xd
        del x
        done += n; b += 1
    out = {}
    for l in range(1, D+1):
        m = mu[l]/NTRUTH; out["mu_%d" % l] = cp.asnumpy(m)
        out["S_%d" % l] = cp.asnumpy(S[l]/NTRUTH - cp.outer(m, m))
    del Wg, mu, S; cp.get_default_memory_pool().free_all_blocks()
    return out

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
NNET = int(os.environ.get("NNET", "8"))
lg("мереж %d · N=%d" % (NNET, NTRUTH))
for idx in range(NNET):
    nm = NAMES[idx]; dst = "%s/covbank118_%s.npz" % (OUT, nm)
    if os.path.exists(dst):
        lg("%-20s уже є" % nm); continue
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    try: r = run(W, 920000 + idx*7919)
    except Exception as e: lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
    np.savez(dst + ".tmp", **r); os.replace(dst + ".tmp.npz", dst)
    lg("%-20s готово · tr S_16 %.4f · ‖μ_16‖ %.4f" % (nm, np.trace(r["S_16"]), np.linalg.norm(r["mu_16"])))
lg("ГОТОВО")
print("DONE_COVBANK")
