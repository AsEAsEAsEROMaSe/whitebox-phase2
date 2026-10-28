"""С117 — СТЕЛЯ «ВИЩА ЗАЛЕЖНІСТЬ У ПІДПРОСТОРІ РАНГУ r» (втручання, метод ЮЗЕРА).

[dep117] знищити ВСЮ залежність на шарі 14 = 0.8% виходу; гаусизувати з
істинними (μ,Σ) = 0.06% ⟹ другий порядок несе 93%, вища залежність — 0.06%/шар,
і саме вона накопичується у дрейф (μ,Σ). Питання: чи живе ця вища залежність
у НИЗЬКОРАНГОВОМУ підпросторі (PR(Σ₁₅)≈74)? Якщо так — тензор κ₃ у підпросторі
має стелю; якщо вона розмазана по 1024 вимірах — мертвий механізмом.

Часткова гаусизація h_{K-1} (входу шару K):
   u = (h−μ)·V_r           (топ-r власних напрямків Σ) — лишаємо ЯК Є (усі порядки)
   w_⊥ ← N(0, Σ_⊥)         (доповнення) — гаусизуємо, незалежно від u
   h' = μ + u·V_rᵀ + w_⊥
Далі — ІСТИННА мережа до виходу. Похибка: косинусна (%) проти ref з тих самих семплів.
🔒 r=0 мусить відтворити gauss@K з dep117 · 🔒 r=1024 ⟹ h'=h ⟹ похибка 0.

🔮 K1 (40%) r=74 знімає ≥50% від gauss@K (над підлогою) ⟹ підпростір несе структуру
🔮 K2 (45%) r=74 знімає <30% ⟹ вища залежність розмазана, тензор у підпросторі мертвий
🔮 K3 (80%) монотонно по r; r=256 знімає ≥70%
"""
import sys, glob, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 15
KS = [8, 14]
RS = [0, 8, 32, 74, 128, 256, 1024]
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def fwd(x, Wg, k0, k1):
    for k in range(k0, k1):
        x = cp.maximum(x @ Wg[k], 0.0)
    return x

def run(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    acc = {}
    for K in KS:
        acc[("ref", K)] = cp.zeros(NW, cp.float64)
        for r in RS: acc[(r, K)] = cp.zeros(NW, cp.float64)
    done, b = 0, 0
    while done < NTRUTH:
        n = min(BATCH, NTRUTH - done)
        rs = cp.random.RandomState(seed=seed + b)
        z = rs.standard_normal((n, NW), dtype=cp.float32)
        for K in KS:
            h = fwd(z, Wg, 0, K)
            o = fwd(h, Wg, K, D-1); acc[("ref", K)] += cp.maximum(o @ Wg[D-1], 0.0).astype(cp.float64).sum(0)
            hd = h.astype(cp.float64); mu = hd.mean(0); dc = hd - mu
            S = (dc.T @ dc)/float(n)
            lam, V = cp.linalg.eigh(S)                 # зростання
            lam = cp.maximum(lam, 0.0); V = V[:, ::-1]; lam = lam[::-1]
            Zf = rs.standard_normal((n, NW), dtype=cp.float64)
            for r in RS:
                if r >= NW:
                    hp = h
                else:
                    Vr, Vp = V[:, :r], V[:, r:]
                    u = dc @ Vr                                          # (n, r) як є
                    wg = (Zf[:, r:] * cp.sqrt(lam[r:])) @ Vp.T           # N(0, Σ_⊥)
                    hp = (mu + u @ Vr.T + wg).astype(cp.float32)
                o = fwd(hp, Wg, K, D-1); acc[(r, K)] += cp.maximum(o @ Wg[D-1], 0.0).astype(cp.float64).sum(0)
                del hp, o
            del h, hd, dc, S, lam, V, Zf
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
lg("мереж %d · N=%d · K=%s · r=%s" % (NNET, NTRUTH, KS, RS))
RES = []
for idx in range(NNET):
    nm = NAMES[idx]; dst = "%s/k3sub117_%s.npz" % (OUT, nm)
    if os.path.exists(dst):
        d = np.load(dst); r = {k: d[k] for k in d.files}; lg("%-20s уже є" % nm)
    else:
        W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
        try: r = run(W, 800000 + idx*7919)
        except Exception as e: lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
        np.savez(dst + ".tmp", **r); os.replace(dst + ".tmp.npz", dst)
    RES.append(r)
    lg("%-20s готово · K=14: r=0 %.4f%% · r=74 %.4f%% · r=256 %.4f%% · 🔒r=1024 %.2e"
       % (nm, cerr(r["0_14"], r["ref_14"]), cerr(r["74_14"], r["ref_14"]),
          cerr(r["256_14"], r["ref_14"]), cerr(r["1024_14"], r["ref_14"])))

print("\n── похибка виходу (%%, косинусна, %d мереж): вища структура збережена у топ-r ──" % len(RES))
print("%6s | %s" % ("r", " ".join("K=%-8d" % K for K in KS)))
for r in RS:
    print("%6d | %s" % (r, " ".join("%8.4f  " % np.mean([cerr(x["%d_%d" % (r, K)], x["ref_%d" % K]) for x in RES]) for K in KS)))
print("(r=0 ≡ gauss@K із dep117 · r=1024 ≡ ref ⟹ 0)")
lg("ГОТОВО")
print("DONE_K3SUB")
