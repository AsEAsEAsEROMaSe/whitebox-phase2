"""С118 — ТРЕТІ МОМЕНТИ ПЕРЕДАКТИВАЦІЙ НА ВСІХ ШАРАХ: ПОВНІ ТА У ПІДПРОСТОРІ r

Для inj118b: побудувати Δ̂m_l (Еджворт по γ₃) і Δ̂C_l (парний Еджворт по κ_aab) з
ІСТИННОГО стану, але з кумулянтами, обмеженими топ-r власних напрямків Σ(h_l), і вкинути
в просту рекурсію. Дає СТЕЛЮ «3-й порядок у підпросторі рангу r» для обох каналів разом.

На кожному шарі l=1..15 (l=0: h=z гаусів ТОЧНО, κ₃=0):  p̃ = (h_l − μ_l)W_l
   повні:  P3_l = E[p̃² ⊗ p̃]  (1024×1024, E[p̃_a² p̃_b]); діагональ = E[p̃_a³]
   підпростір: q = u[:, :r]·G[:r], u = (h−μ)V, G = VᵀW ⟹ P3^{(r)} = E[q² ⊗ q]
Σ_l і V_l — з першого проходу (той самий сід ⟹ ті самі семпли).
Чекпоінт: /home/trim/wblogs/sub118_<net>.npz
"""
import sys, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 14
RS = [74, 128, 256]
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def run(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    mu = [cp.zeros(NW, cp.float64) for _ in range(D)]
    S = [cp.zeros((NW, NW), cp.float64) for _ in range(D)]
    done, b = 0, 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        x = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
        for l in range(1, D):
            x = cp.maximum(x @ Wg[l-1], 0.0)
            xd = x.astype(cp.float64); mu[l] += xd.sum(0); S[l] += xd.T @ xd; del xd
        del x; done += n; b += 1
    V, G = [None]*D, [None]*D
    for l in range(1, D):
        mu[l] /= NTRUTH; S[l] = S[l]/NTRUTH - cp.outer(mu[l], mu[l])
        lam, v = cp.linalg.eigh(S[l]); V[l] = v[:, ::-1].copy()
        G[l] = V[l].T @ Wg[l].astype(cp.float64)
        del lam, v
    P3 = {(l, "f"): cp.zeros((NW, NW), cp.float64) for l in range(1, D)}
    for l in range(1, D):
        for r in RS: P3[(l, r)] = cp.zeros((NW, NW), cp.float64)
    done, b = 0, 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        x = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
        for l in range(1, D):
            x = cp.maximum(x @ Wg[l-1], 0.0)
            h = x.astype(cp.float64) - mu[l]
            p = h @ Wg[l].astype(cp.float64)
            P3[(l, "f")] += (p*p).T @ p
            u = h @ V[l]
            for r in RS:
                q = u[:, :r] @ G[l][:r]
                P3[(l, r)] += (q*q).T @ q
                del q
            del h, p, u
        del x; done += n; b += 1
    out = {}
    for l in range(1, D):
        out["mu_%d" % l] = cp.asnumpy(mu[l]); out["S_%d" % l] = cp.asnumpy(S[l])
        out["P3f_%d" % l] = cp.asnumpy(P3[(l, "f")])/NTRUTH
        for r in RS: out["P3r%d_%d" % (r, l)] = cp.asnumpy(P3[(l, r)])/NTRUTH
    del Wg, mu, S, V, G, P3; cp.get_default_memory_pool().free_all_blocks()
    return out

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
NNET = int(os.environ.get("NNET", "8"))
lg("мереж %d · N=%d · r=%s" % (NNET, NTRUTH, RS))
for idx in range(NNET):
    nm = NAMES[idx]; dst = "%s/sub118_%s.npz" % (OUT, nm)
    if os.path.exists(dst):
        lg("%-20s уже є" % nm); continue
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    try: r = run(W, 920000 + idx*7919)          # інший потік семплів, ніж covbank118 (BATCH інший)
    except Exception as e: lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
    np.savez(dst + ".tmp", **r); os.replace(dst + ".tmp.npz", dst)
    s = np.sqrt(np.diag(W[15].T @ r["S_15"] @ W[15]))
    lg("%-20s готово · γ₃ шар15: full sd %.4f · r74 частка %.3f · r256 %.3f" % (
        nm, (np.diag(r["P3f_15"])/s**3).std(),
        np.diag(r["P3r74_15"]).std()/np.diag(r["P3f_15"]).std(), np.diag(r["P3r256_15"]).std()/np.diag(r["P3f_15"]).std()))
lg("ГОТОВО")
print("DONE_SUB118")
