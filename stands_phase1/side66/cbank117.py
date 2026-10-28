"""С117 — БАНК ІСТИННИХ ПОВНИХ КОВАРІАЦІЙ ПЕРЕДАКТИВАЦІЙ.

НАВІЩО. `lad116` вкидав істинні (mp,sd) на шарі K і діставав ШКОДУ (K=12 ×0.365).
Я перевірив код: рядок 41-42 масштабує ВЕСЬ `cp_` через outer(g,g) ⟹ діагональ
узгоджена, ламається НЕ вона. Лишається одне: вкидалась істинна ДІАГОНАЛЬ при
НАШИХ КОРЕЛЯЦІЯХ. На шарі 15 кореляцій уже нема (вихід — середнє) ⟹ там ×9.395.

⟹ НЕЗМІРЯНА КЛІТИНКА: істинна ПОВНА `cp` + істинна `mp` разом на шарі K.
[С115] істинна C сама дає лише ×1.03…1.07; істинна діагональ сама дає ×0.365;
разом — не міряв ніхто, а на шарі 15 така пара дала СИНЕРГІЮ ×8.4.

Цей файл лише СКЛАДАЄ банк. Вирок — у `cinj117.py`.

Двопрохідна схема (проти катастрофічного скорочення при |t|≈2.2):
  прохід 1 — оцінка середніх m̂ по одному батчу
  прохід 2 — Gram(P − m̂) у fp32, накопичення у fp64
🔒 ПЛОМБИ: (1) шар 0 має CP = WᵀW ТОЧНО ⟹ порівняти з аналітикою
           (2) sd з повної CP мусить збігтися зі старим банком gamall_*
"""
import sys, glob, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 21
BATCH = 1 << 15
OUT = "/home/trim/wbcov"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
os.makedirs(OUT, exist_ok=True)

def build(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    # ── прохід 1: грубі середні ──
    rs = cp.random.RandomState(seed=seed)
    x = rs.standard_normal((BATCH, NW), dtype=cp.float32)
    mhat = []
    for k in range(D):
        p = x @ Wg[k]
        mhat.append(p.mean(0))
        if k < D-1: x = cp.maximum(p, 0.0)
    del x, p; cp.get_default_memory_pool().free_all_blocks()
    # ── прохід 2 ──
    S1 = [cp.zeros(NW, cp.float64) for _ in range(D)]
    G = [cp.zeros((NW, NW), cp.float64) for _ in range(D)]
    RL = [cp.zeros(NW, cp.float64) for _ in range(D)]
    done, b = 0, 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        rs = cp.random.RandomState(seed=seed + 1000 + b)
        x = rs.standard_normal((n, NW), dtype=cp.float32)
        for k in range(D):
            p = x @ Wg[k]
            d = p - mhat[k]
            S1[k] += d.astype(cp.float64).sum(0)
            G[k] += (d.T @ d).astype(cp.float64)
            RL[k] += cp.maximum(p, 0.0).astype(cp.float64).sum(0)
            if k < D-1: x = cp.maximum(p, 0.0)
        del x, p, d; cp.get_default_memory_pool().free_all_blocks()
        done += n; b += 1
    out = {}
    for k in range(D):
        mu = S1[k]/NTRUTH
        CPk = G[k]/NTRUTH - cp.outer(mu, mu)
        out["cp_%d" % k] = cp.asnumpy(CPk).astype(np.float32)
        out["mp_%d" % k] = cp.asnumpy(mu + mhat[k].astype(cp.float64))
        out["relu_%d" % k] = cp.asnumpy(RL[k]/NTRUTH)
    del Wg, S1, G, RL; cp.get_default_memory_pool().free_all_blocks()
    return out

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
NNET = int(os.environ.get("NNET", "8"))
lg("мереж %d · N=%d · вихід %s" % (NNET, NTRUTH, OUT))
for idx in range(NNET):
    nm = NAMES[idx]
    dst = "%s/cov_%s.npz" % (OUT, nm)
    if os.path.exists(dst): lg("%-20s уже є" % nm); continue
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    try:
        r = build(W, 900000 + idx*7919)
    except Exception as e:
        lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
    # 🔒 пломба 1: шар 0 CP = WᵀW
    an = W[0].T @ W[0]
    rel = np.abs(r["cp_0"] - an).max()/np.abs(an).max()
    r["name"] = nm; r["n"] = NTRUTH
    tmp = dst + ".tmp"
    np.savez(tmp, **r); os.replace(tmp + ".npz", dst)
    lg("%-20s готово · 🔒ш0 vs WᵀW відн.макс %.2e" % (nm, rel))
lg("ГОТОВО")
print("DONE_CBANK")
