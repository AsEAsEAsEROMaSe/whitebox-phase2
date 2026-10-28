"""С117 — ВТРУЧАННЯ В ЗАЛЕЖНІСТЬ МІЖ НЕЙРОНАМИ (метод ЮЗЕРА, наведений на горло).

[mstep117b] похибка кроку C — ЗАЛЕЖНІСТЬ між нейронами, не асиметрія маргіналів.
Питання: СКІЛЬКИ похибки ВИХОДУ створює втрата залежності на шарі K?
Це стеля ВСЬОГО класу понейронних схем одразу (гаус, γ₃, будь-яка майбутня).

На істинних семплах, з ОДНОГО префіксу до шару K (шум МК скорочується):
   ref     h_K як є → вихід
   shuf    кожен нейрон h_K своєю перестановкою (маргінали ті самі, залежність — 0)
   gauss   h_K ← N(μ_K, Σ_K) із тих самих семплів (2-й порядок є, вище — 0)
           ← РІВНО припущення нашого замикання
Похибка = косинусна відносна (%) між виходом ref і виходом втручання.
🔒 K=0 на z (iid гаус): shuf мусить дати підлогу МК; gauss — те саме.

🔮 S1 (60%) shuf@14 ≥ 0.3% (≫ наші 0.13%) ⟹ залежність винна, стеля класу реальна
🔮 S2 (55%) gauss@K < shuf@K, але gauss@14 ≥ 0.1%
🔮 S3 (85%) K=0 = підлога МК (інакше стенд зламаний)
🔮 S4 (50%) похибка РОСТЕ з K (глибше — менше шарів на відновлення)
"""
import sys, glob, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 15
KS = [0, 1, 4, 8, 12, 14]
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def fwd(x, Wg, k0, k1):
    for k in range(k0, k1):
        x = cp.maximum(x @ Wg[k], 0.0)
    return x

def run(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    acc = {("ref", K): cp.zeros(NW, cp.float64) for K in KS}
    acc.update({("shuf", K): cp.zeros(NW, cp.float64) for K in KS})
    acc.update({("gauss", K): cp.zeros(NW, cp.float64) for K in KS})
    done, b = 0, 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        rs = cp.random.RandomState(seed=seed + b)
        z = rs.standard_normal((n, NW), dtype=cp.float32)
        for K in KS:
            # префікс: вхід шару K (K=0 ⟹ сам z; K≥1 ⟹ h_{K-1} після ReLU)
            h = z if K == 0 else fwd(z, Wg, 0, K)
            # ref
            o = fwd(h, Wg, K, D-1); acc[("ref", K)] += cp.maximum(o @ Wg[D-1], 0.0).astype(cp.float64).sum(0)
            # shuf: кожен стовпець своєю перестановкою
            keys = rs.random_sample((n, NW), dtype=cp.float32)
            idx = cp.argsort(keys, axis=0)
            hs = cp.take_along_axis(h, idx, axis=0)
            o = fwd(hs, Wg, K, D-1); acc[("shuf", K)] += cp.maximum(o @ Wg[D-1], 0.0).astype(cp.float64).sum(0)
            # gauss: N(μ, Σ) з тих самих семплів
            hd = h.astype(cp.float64); mu = hd.mean(0); dc = hd - mu
            S = (dc.T @ dc)/float(n)
            S[cp.arange(NW), cp.arange(NW)] += 1e-6*float(cp.diag(S).max())
            Lc = cp.linalg.cholesky(S)
            g = (mu + rs.standard_normal((n, NW), dtype=cp.float64) @ Lc.T).astype(cp.float32)
            o = fwd(g, Wg, K, D-1); acc[("gauss", K)] += cp.maximum(o @ Wg[D-1], 0.0).astype(cp.float64).sum(0)
            del h, hs, g, o, hd, dc, S, Lc, idx, keys
        del z; cp.get_default_memory_pool().free_all_blocks()
        done += n; b += 1
    out = {"%s_%d" % (k[0], k[1]): cp.asnumpy(v)/NTRUTH for k, v in acc.items()}
    del Wg, acc; cp.get_default_memory_pool().free_all_blocks()
    return out

def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T
    return 100*np.sqrt(float(r @ r)/float(T @ T))

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
NNET = int(os.environ.get("NNET", "8"))
lg("мереж %d · N=%d · K=%s" % (NNET, NTRUTH, KS))
RES = []
for idx in range(NNET):
    nm = NAMES[idx]; dst = "%s/dep117_%s.npz" % (OUT, nm)
    if os.path.exists(dst):
        d = np.load(dst); r = {k: d[k] for k in d.files}; lg("%-20s уже є" % nm)
    else:
        W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
        try: r = run(W, 700000 + idx*7919)
        except Exception as e: lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
        np.savez(dst + ".tmp", **r); os.replace(dst + ".tmp.npz", dst)
    RES.append(r)
    lg("%-20s готово · shuf@14 %.4f%% · gauss@14 %.4f%% · 🔒K=0 shuf %.4f%%"
       % (nm, cerr(r["shuf_14"], r["ref_14"]), cerr(r["gauss_14"], r["ref_14"]),
          cerr(r["shuf_0"], r["ref_0"])))

print("\n── похибка ВИХОДУ від втручання на шарі K (%%, косинусна, %d мереж) ──" % len(RES))
print("%4s %10s %10s" % ("K", "shuf", "gauss"))
for K in KS:
    a = np.mean([cerr(r["shuf_%d" % K], r["ref_%d" % K]) for r in RES])
    g = np.mean([cerr(r["gauss_%d" % K], r["ref_%d" % K]) for r in RES])
    print("%4d %10.4f %10.4f%s" % (K, a, g, "   ← 🔒 підлога МК" if K == 0 else ""))
print("\nдля порівняння: похибка НАШОГО замикання на виході ≈ 0.13%")
lg("ГОТОВО")
print("DONE_DEP117")
