"""С119/05.09 — БАНК κ_aab(p_l) ІЗ N СЕМПЛІВ (CPU, 1 ядро): носій дрібної високорангової частини κ₃, якому байдужий ранг.

Гіпотеза: дефіцит κ₃, що важить, ізотропний і високоранговий (kdir: поза топ-64 ×1.50) — жоден детермінований носій його не несе,
але семпли несуть будь-яку структуру; питання лише в ШУМІ. Даунстрім (крок Cov, контракція з W_ai W_aj) усереднює ~10⁶ елементів,
тож ефективний SNR вищий за поелементний.
Банк: P3f_l[a,b] = E[p̃_a² p̃_b], p̃ = p − mean (вибіркове центрування), одним проходом через сирі моменти. Сід 555000 (≠ sub118 777000).
Далі: rec118h KBANK=... KTRUE=c (c=0.5, 1.0) STATE=1 на 8 мережах.
ПЕРЕДБАЧЕННЯ (до прогону): база 0.0605% · оракул 2²⁰ 0.0337% · N=6144 c=0.5 → ~0.050% (шанс 40% що <0.050) ·
N=98304 c=1.0 → ~0.040%; надлишок над оракулом ∝ 1/√N. Якщо N=6144 дає ≤0.046% — окупність при util 0.23 нуль, але шлях відкритий.
"""
import os, sys, time
import numpy as np, pyarrow.parquet as pq
D, NW = 16, 1024; OUT = "/home/trim/wblogs"; T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
NS = int(os.environ.get("NS", "6144")); BATCH = min(NS, 1 << 14); SEED = int(os.environ.get("SEED", "555000"))
NETS = os.environ.get("NETS", "regina-johnston").split(",")
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
for nm in NETS:
    out = "%s/ksamp119_%s_N%d.npz" % (OUT, nm, NS)
    if os.path.exists(out): lg("є %s" % out); continue
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float32).reshape(D, NW, NW)
    M1 = np.zeros((D, NW)); M2 = np.zeros((D, NW, NW)); M3 = np.zeros((D, NW, NW))
    done, b = 0, 0
    while done < NS:
        n = min(BATCH, NS - done)
        x = np.random.RandomState(SEED + b).standard_normal((n, NW)).astype(np.float32)
        for l in range(1, D):
            x = np.maximum(x @ W[l-1], 0.0)
            p = (x @ W[l]).astype(np.float64)
            M1[l] += p.sum(0); M2[l] += p.T @ p; M3[l] += (p*p).T @ p
        done += n; b += 1
    bank = {}
    for l in range(1, D):
        mu = M1[l]/NS; E2 = M2[l]/NS; E3 = M3[l]/NS
        bank["P3f_%d" % l] = E3 - mu[None, :]*np.diag(E2)[:, None] - 2*mu[:, None]*E2 + 2*(mu*mu)[:, None]*mu[None, :]
        bank["mu_%d" % l] = mu; bank["s_%d" % l] = np.sqrt(np.diag(E2) - mu*mu)
    np.savez(out + ".tmp.npz", **bank); os.replace(out + ".tmp.npz", out)
    sb = "%s/sub118_%s.npz" % (OUT, nm)
    if os.path.exists(sb):
        T = np.load(sb)["P3f_15"]; K = bank["P3f_15"]
        lg("%-18s N=%d: ‖K−T‖/‖T‖ %.3f · діаг %.3f · cos %.4f" % (nm, NS, np.linalg.norm(K-T)/np.linalg.norm(T),
           np.linalg.norm(np.diag(K)-np.diag(T))/np.linalg.norm(np.diag(T)), (K*T).sum()/np.linalg.norm(K)/np.linalg.norm(T)))
    else: lg("%s N=%d готово" % (nm, NS))
print("DONE_KSAMP")
