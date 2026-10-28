"""С118 — ЧИ ЖИВЕ ПОНЕЙРОННА γ₃ (і γ₄) ПЕРЕДАКТИВАЦІЙ У ТОП-r ПІДПРОСТОРІ ВХОДУ?

Будівничий крок після k3sub117 (топ-74 напрямків Σ(h) тримають 68% вищої залежності
на ВИХОДІ при K=14, 47% при K=8). Тут — те саме питання для ОДНОГО КРОКУ, тобто для
величини, що є ВСІМ залишком кроку (res117: R² 0.999 від γ₃+γ₄):
   p = h·W,  h = μ + u·V_rᵀ + c,   u = (h−μ)·V_r (топ-r власних Σ(h)), c ⟂ u — гаус
   ⟹ кумулянти адитивні: κ₃(p_a) = κ₃(u·g_a), κ₄(p_a) = κ₄(u·g_a),  g_a = V_rᵀ w_a
Ніякого тензора: q_a = u·g_a рахується прямо з семплів ⟹ це ОРАКУЛЬНА стеля
представлення «κ₃ у підпросторі рангу r» для кроку K−1 → K.

Метрики по r (усереднення по 8 мережах):
   E3 = 1 − ‖δ₃_true − δ₃_r‖²/‖δ₃_true‖²,   δ₃ = −(γ₃/6)·s·t·φ(t)  (вага у кроці m)
   E4 — те саме для δ₄ = (γ₄/24)·s·(t²−1)·φ(t);  corr — по нейронах.
🔒 r=1024 ⟹ E=1.000 (тавтологія) · 🔒 r=0 ⟹ 0 · 🔒 γ₃ з семплів проти банку gamall
🔒 КОНТРОЛЬ: ВИПАДКОВИЙ ортонормований підпростір r=74 — має дати ≈ r/1024 ≈ 0.07.

🔮 P1 (50%): K=14, r=74: E3 ≥ 0.60 (узгоджено з k3sub 68%)
🔮 P2 (70%): випадковий r=74 дає E3 < 0.15
🔮 P3 (80%): E3 монотонна по r; r=256 ≥ 0.85 при K=14
🔮 P4 (60%): E4 < E3 при тому ж r (четвертий порядок «ширший»)
"""
import sys, glob, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 15
KS = [8, 14]
RS = [8, 32, 74, 128, 256, 512, 1024]
RC = 74
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def fwd(x, Wg, k0, k1):
    for k in range(k0, k1):
        x = cp.maximum(x @ Wg[k], 0.0)
    return x

def phi(t): return np.exp(-0.5*t*t)/np.sqrt(2*np.pi)

def run(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    out = {}
    for K in KS:
        # ── прохід 1: μ, Σ входу шару K і (mp, s) передактивації p = h·W_K ──
        mu = cp.zeros(NW, cp.float64); S = cp.zeros((NW, NW), cp.float64)
        done, b = 0, 0
        while done < NTRUTH:
            n = min(BATCH, NTRUTH - done)
            z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
            h = fwd(z, Wg, 0, K).astype(cp.float64)
            mu += h.sum(0); S += h.T @ h
            done += n; b += 1
        mu /= NTRUTH; S = S/NTRUTH - cp.outer(mu, mu)
        lam, V = cp.linalg.eigh(S); V = V[:, ::-1]; lam = cp.maximum(lam[::-1], 0.0)
        pr = float(lam.sum()**2/(lam**2).sum())
        WK = Wg[K].astype(cp.float64)
        G = V.T @ WK                                    # (1024 напрямків × 1024 нейронів)
        rs = cp.random.RandomState(seed=seed + 424242)
        Q, _ = cp.linalg.qr(rs.standard_normal((NW, RC), dtype=cp.float64))
        Gc = Q.T @ WK
        # ── прохід 2: моменти q_r = u[:, :r]·G[:r] та контроль ──
        keys = [("r", r) for r in RS] + [("c", RC)]
        m2 = {k: cp.zeros(NW, cp.float64) for k in keys}
        m3 = {k: cp.zeros(NW, cp.float64) for k in keys}
        m4 = {k: cp.zeros(NW, cp.float64) for k in keys}
        done, b = 0, 0
        while done < NTRUTH:
            n = min(BATCH, NTRUTH - done)
            z = cp.random.RandomState(seed=seed + b).standard_normal((n, NW), dtype=cp.float32)
            h = fwd(z, Wg, 0, K).astype(cp.float64) - mu
            u = h @ V
            for r in RS:
                q = u[:, :r] @ G[:r]
                m2[("r", r)] += (q*q).sum(0); m3[("r", r)] += (q**3).sum(0); m4[("r", r)] += (q**4).sum(0)
            q = (h @ Q) @ Gc
            m2[("c", RC)] += (q*q).sum(0); m3[("c", RC)] += (q**3).sum(0); m4[("c", RC)] += (q**4).sum(0)
            del z, h, u, q
            done += n; b += 1
        mp = cp.asnumpy(mu @ WK)
        res = {"mp": mp, "pr": pr, "lam": cp.asnumpy(lam)}
        for k in keys:
            v2 = cp.asnumpy(m2[k])/NTRUTH; v3 = cp.asnumpy(m3[k])/NTRUTH; v4 = cp.asnumpy(m4[k])/NTRUTH
            tag = "%s%d" % k
            res["k2_%s" % tag] = v2; res["k3_%s" % tag] = v3; res["k4_%s" % tag] = v4 - 3*v2*v2
        out[K] = res
        del mu, S, lam, V, WK, G, Q, Gc, m2, m3, m4
        cp.get_default_memory_pool().free_all_blocks()
    del Wg; cp.get_default_memory_pool().free_all_blocks()
    return out

def energy(pred, true):
    return 1.0 - float(((true - pred)**2).sum()/(true**2).sum())

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
NNET = int(os.environ.get("NNET", "8"))
lg("мереж %d · N=%d · K=%s · r=%s · контроль r=%d" % (NNET, NTRUTH, KS, RS, RC))
RES = []
for idx in range(NNET):
    nm = NAMES[idx]; dst = "%s/g3sub118_%s.npz" % (OUT, nm)
    if os.path.exists(dst):
        d = np.load(dst); r = {k: d[k] for k in d.files}; lg("%-20s уже є" % nm)
    else:
        W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
        try: o = run(W, 900000 + idx*7919)
        except Exception as e: lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
        r = {}
        for K, res in o.items():
            for k, v in res.items(): r["%s_K%d" % (k, K)] = v
        np.savez(dst + ".tmp", **r); os.replace(dst + ".tmp.npz", dst)
    RES.append((nm, r))
    # 🔒 пломба проти банку gamall (γ₃ передактивації шару K)
    g = "%s/gamall_%s.npz" % (OUT, nm)
    seal = ""
    if os.path.exists(g):
        d = np.load(g)
        for K in KS:
            s = np.sqrt(r["k2_r1024_K%d" % K]); g3 = r["k3_r1024_K%d" % K]/s**3
            gb = d["g3_%d" % K]
            seal += " 🔒K%d corr(γ₃, банк)=%.4f" % (K, np.corrcoef(g3, gb)[0, 1])
    lg("%-20s готово · PR(Σ) K8 %.1f K14 %.1f%s" % (nm, r["pr_K8"], r["pr_K14"], seal))

def table(order):
    print("\n── частка енергії δ%d (вага кроку m), усереднено по %d мережах ──" % (order, len(RES)))
    print("%8s | %s" % ("r", " ".join("K=%-2d E    corr   " % K for K in KS)))
    rows = [("r", r) for r in RS] + [("c", RC)]
    for kind, r in rows:
        cells = []
        for K in KS:
            Es, Cs = [], []
            for nm, d in RES:
                s = np.sqrt(d["k2_r1024_K%d" % K]); mp = d["mp_K%d" % K]; t = mp/s
                tag = "%s%d_K%d" % (kind, r, K)
                if order == 3:
                    true = -(d["k3_r1024_K%d" % K]/s**3)/6*s*t*phi(t)
                    pred = -(d["k3_%s" % tag]/s**3)/6*s*t*phi(t)
                else:
                    true = (d["k4_r1024_K%d" % K]/s**4)/24*s*(t*t-1)*phi(t)
                    pred = (d["k4_%s" % tag]/s**4)/24*s*(t*t-1)*phi(t)
                Es.append(energy(pred, true)); Cs.append(np.corrcoef(pred, true)[0, 1])
            cells.append("%7.4f %7.4f  " % (np.mean(Es), np.mean(Cs)))
        print("%8s | %s" % (("r=%d" % r) if kind == "r" else ("ВИПАДК %d" % r), " ".join(cells)))
    print("(r=1024 ⟹ 1.0000 тавтологія · випадковий підпростір — контроль)")

table(3); table(4)
print("PR(Σ входу): K8 %.1f · K14 %.1f" % (np.mean([d["pr_K8"] for _, d in RES]), np.mean([d["pr_K14"] for _, d in RES])))
lg("ГОТОВО")
print("DONE_G3SUB")
