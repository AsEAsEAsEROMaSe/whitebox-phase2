"""С118 — ЧИ ПОЯСНЮЮТЬ МІШАНІ ТРЕТІ/ЧЕТВЕРТІ КУМУЛЯНТИ ЗАЛИШОК КРОКУ КОВАРІАЦІЇ?

Мотив. res117: залишок кроку СЕРЕДНЬОГО = γ₃+γ₄ (R² 0.999). mstep117: крок Мелера
для C від ІСТИННОЇ cp має 5.3% похибки кореляцій на шар, і це негаусовість, не
обрізання ряду. Стеля «виправити лише середнє» в нашій рекурсії = ×1.26 (С117), бо
sd хибна на 2.9% і через φ(t)·δs годує mp. Питання: чи є залишок кроку C функцією
МІШАНИХ кумулянтів (κ_aab, κ_abb, κ_aabb, κ_aaab) — тобто того, що несе тензор κ₃/κ₄
у підпросторі? Якщо так — тристороння рекурсія лагодить ОБИДВА канали узгоджено.

Парний Еджворт (виведено з кумулянтної твірної, He — пробабілістичні):
   ΔCov₃(a,b) = c2ₐc1ᵦ κ_aab + c1ₐc2ᵦ κ_abb
              + ρ·[6c3ₐc2ᵦ κ_aab + 6c2ₐc3ᵦ κ_abb + 4c4ₐc1ᵦ κ_aaa + 4c1ₐc4ᵦ κ_bbb]   (+O(ρ²))
   ΔCov₄(a,b) = c2ₐc2ᵦ κ_aabb + c3ₐc1ᵦ κ_aaab + c1ₐc3ᵦ κ_abbb                       (+O(ρ))
   c_n = σ(−1)ⁿHe_{n−2}(t)φ(t)/n! (n≥2), c1 = σΦ, c0 = σψ ;  кумулянти СТАНДАРТИЗОВАНІ
   діагональ — точна 1-D: ΔE[f²] = s²φ(γ₃/3 − γ₄t/12), ΔE[f] = s(−tφγ₃/6 + (t²−1)φγ₄/24)
ΔC_true = Cov_true(ReLU p) − Мелер(mp,s,ρ; n≤8) з ІСТИННИХ (mp,Σ_p) тих самих семплів.
Підпростір: κ^{(r)} з q = u_r·G_r (кумулянти адитивні, гаусове доповнення не дає внеску).
🔒 ГАУСІВ КОНТРОЛЬ: p_g ~ N(mp, Σ_p) тими ж акумуляторами ⟹ ΔC_g ≈ підлога, κ_g ≈ 0.

Метрики (позадіагональ, Фробеніус; і функціонал кроку v_c = diag(WᵀΔC W) — похибка s'²):
   R²₃, R²₃₊₄ повні · R²₃(r=74), R²₃(r=256) · підлога з контролю
🔮 P1 (55%): R²₃₊₄ позадіагоналі ≥ 0.7 при K=14 (як res117 для середнього)
🔮 P2 (50%): третій порядок сам ≥ 0.5 (κ₄ важить більше, ніж для середнього)
🔮 P3 (60%): r=256 тримає ≥ 0.7 від повного R²₃ · r=74 ≥ 0.45
🔮 P4 (70%): функціонал v_c (діагональ наступного шару) пояснюється КРАЩЕ за Фробеніус
"""
import sys, glob, time, os
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq

D, NW = 16, 1024
NTRUTH = 1 << 20
BATCH = 1 << 15
KS = [8, 14]
RS = [74, 256]
OUT = "/home/trim/wblogs"
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def fwd(x, Wg, k0, k1):
    for k in range(k0, k1):
        x = cp.maximum(x @ Wg[k], 0.0)
    return x

def run(W, seed):
    Wg = [cp.asarray(W[k].astype(np.float32)) for k in range(D)]
    out = {}
    for K in KS:
        WK = Wg[K].astype(cp.float64)
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
        mp = mu @ WK; Sp = WK.T @ S @ WK
        Sp = 0.5*(Sp + Sp.T)
        Lp = cp.linalg.cholesky(Sp + 1e-9*cp.trace(Sp)/NW*cp.eye(NW))
        G = V.T @ WK
        # акумулятори
        def zero(): return cp.zeros((NW, NW), cp.float64)
        A = {}
        for tag in ["t", "g"] + ["r%d" % r for r in RS]:
            A[tag] = {"P2": zero(), "P3": zero(), "P4": zero(), "P31": zero()}
        for tag in ["t", "g"]:
            A[tag]["R1"] = cp.zeros(NW, cp.float64); A[tag]["R2"] = zero()
            A[tag]["M1"] = cp.zeros(NW, cp.float64)
        def acc(tag, pc, relu=None):
            p2 = pc*pc; p3 = p2*pc
            A[tag]["P2"] += pc.T @ pc; A[tag]["P3"] += p2.T @ pc
            A[tag]["P4"] += p2.T @ p2; A[tag]["P31"] += p3.T @ pc
            if relu is not None:
                A[tag]["R1"] += relu.sum(0); A[tag]["R2"] += relu.T @ relu; A[tag]["M1"] += pc.sum(0)
            del p2, p3
        done, b = 0, 0
        while done < NTRUTH:
            n = min(BATCH, NTRUTH - done)
            rs = cp.random.RandomState(seed=seed + b)
            z = rs.standard_normal((n, NW), dtype=cp.float32)
            h = fwd(z, Wg, 0, K).astype(cp.float64) - mu
            p = h @ WK                                   # центрована передактивація
            acc("t", p, cp.maximum(p + mp, 0.0))
            u = h @ V
            for r in RS:
                acc("r%d" % r, u[:, :r] @ G[:r])
            del h, u, p
            zg = rs.standard_normal((n, NW), dtype=cp.float64)
            pg = zg @ Lp.T
            acc("g", pg, cp.maximum(pg + mp, 0.0))
            del z, zg, pg
            done += n; b += 1
        res = {"mp": cp.asnumpy(mp), "Sp": cp.asnumpy(Sp), "WK": cp.asnumpy(WK), "lam": cp.asnumpy(lam)}
        for tag, dd in A.items():
            for k, v in dd.items(): res["%s_%s" % (tag, k)] = cp.asnumpy(v)/NTRUTH
        out[K] = res
        del mu, S, lam, V, WK, G, Sp, Lp, A
        cp.get_default_memory_pool().free_all_blocks()
    del Wg; cp.get_default_memory_pool().free_all_blocks()
    return out

# ─────────── аналіз (numpy) ───────────
SQ2PI = np.sqrt(2*np.pi)
def phi(t): return np.exp(-0.5*t*t)/SQ2PI
from math import erf
def Phi(t): return 0.5*(1.0 + np.vectorize(erf)(t/np.sqrt(2)))
def He(n, t):
    a, b = np.ones_like(t), t
    if n == 0: return a
    for k in range(1, n):
        a, b = b, t*b - k*a
    return b
def herm_coef(s, t, N=8):
    c = np.zeros((N+1, len(t)))
    ph = phi(t); c[0] = s*(ph + t*Phi(t)); c[1] = s*Phi(t)
    f = 1.0
    for n in range(2, N+1):
        f *= n
        c[n] = s*((-1)**n)*He(n-2, t)*ph/f
    return c

def analyse(d, K, r=None):
    """повертає словник R² по частинах для ключа tag ('t', або 'r74')"""
    mp = d["mp"]; Sp = d["Sp"]; W = d["WK"]
    s = np.sqrt(np.diag(Sp)); t = mp/s; rho = Sp/np.outer(s, s)
    c = herm_coef(s, t, 8)
    # Мелер n≤8 із ІСТИННИХ (mp, Σ_p)
    Cg = np.zeros_like(Sp); f = 1.0
    for n in range(1, 9):
        f *= n; Cg += f*np.outer(c[n], c[n])*rho**n
    Ct = d["t_R2"] - np.outer(d["t_R1"], d["t_R1"])
    dC = Ct - Cg
    Cgc = d["g_R2"] - np.outer(d["g_R1"], d["g_R1"])
    dCg = Cgc - Cg                                      # 🔒 контроль: підлога
    tag = "t" if r is None else "r%d" % r
    P3 = d[tag+"_P3"]; P4 = d[tag+"_P4"]; P31 = d[tag+"_P31"]; P2 = d[tag+"_P2"]
    ss = np.outer(s, s)
    k_aab = P3/(np.outer(s*s, s))                       # E[a²b]/(s_a² s_b)
    k_abb = k_aab.T
    # стандартизовані кумулянти 4-го порядку: κ_aabb = E[a²b²] − E[a²]E[b²] − 2E[ab]²  (сирі моменти P)
    K4 = (P4 - np.outer(np.diag(P2), np.diag(P2)) - 2*P2*P2)/(ss*ss)
    K31 = (P31 - 3*np.outer(np.diag(P2), np.ones(NW))*P2)/(np.outer(s**3, s))
    g3 = np.diag(P3)/s**3; g4 = np.diag(K4)
    # парний Еджворт
    d3 = np.outer(c[2], c[1])*k_aab + np.outer(c[1], c[2])*k_abb \
       + rho*(6*np.outer(c[3], c[2])*k_aab + 6*np.outer(c[2], c[3])*k_abb
              + 4*np.outer(c[4]*g3, c[1]) + 4*np.outer(c[1], c[4]*g3))
    d4 = np.outer(c[2], c[2])*K4 + np.outer(c[3], c[1])*K31 + np.outer(c[1], c[3])*K31.T
    # діагональ — точна 1-D
    ph = phi(t); m0 = c[0]
    dE2 = s*s*ph*(g3/3 - g4*t/12); dE1 = s*(-t*ph*g3/6 + (t*t-1)*ph*g4/24)
    dVar = dE2 - 2*m0*dE1
    dE2_3 = s*s*ph*g3/3; dE1_3 = s*(-t*ph*g3/6); dVar3 = dE2_3 - 2*m0*dE1_3
    off = ~np.eye(NW, dtype=bool)
    def R2(pred, true, mask): return 1.0 - ((true - pred)[mask]**2).sum()/(true[mask]**2).sum()
    out = {}
    out["R2off_3"] = R2(d3, dC, off); out["R2off_34"] = R2(d3 + d4, dC, off)
    out["floor_off"] = (dCg[off]**2).sum()/(dC[off]**2).sum()
    dd = np.diag(dC); one = np.ones(NW, dtype=bool)
    out["R2diag_3"] = R2(dVar3, dd, one)
    out["R2diag_34"] = R2(dVar, dd, one)
    out["floor_diag"] = (np.diag(dCg)**2).sum()/(np.diag(dC)**2).sum()
    # функціонал кроку: похибка s'² наступного шару
    Wn = d.get("Wnext")
    if Wn is not None:
        def vfun(M): return np.einsum("ac,ab,bc->c", Wn, M, Wn)
        vt = vfun(dC); vg = vfun(dCg)
        full3 = d3 + np.diag(dVar3 - np.diag(d3)); full34 = d3 + d4 + np.diag(dVar - np.diag(d3 + d4))
        out["R2v_3"] = 1 - ((vt - vfun(full3))**2).sum()/(vt**2).sum()
        out["R2v_34"] = 1 - ((vt - vfun(full34))**2).sum()/(vt**2).sum()
        out["floor_v"] = (vg**2).sum()/(vt**2).sum()
        sn2 = np.einsum("ac,ab,bc->c", Wn, Ct, Wn)
        out["rel_sd_err"] = 100*np.sqrt(np.mean((0.5*vt/sn2)**2))       # % похибки sd' від ΔC
    out["rel_off"] = 100*np.sqrt((dC[off]**2).sum()/(Ct[off]**2).sum())
    out["rel_off_g"] = 100*np.sqrt((dCg[off]**2).sum()/(Ct[off]**2).sum())
    return out

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
NNET = int(os.environ.get("NNET", "8"))
lg("мереж %d · N=%d · K=%s · r=%s" % (NNET, NTRUTH, KS, RS))
RES = []
for idx in range(NNET):
    nm = NAMES[idx]; dst = "%s/cstep118_%s.npz" % (OUT, nm)
    W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(D, NW, NW)
    if os.path.exists(dst):
        d = np.load(dst); r = {k: d[k] for k in d.files}; lg("%-20s уже є" % nm)
    else:
        try: o = run(W, 910000 + idx*7919)
        except Exception as e: lg("%-20s ПОМИЛКА %s" % (nm, e)); continue
        r = {}
        for K, res in o.items():
            for k, v in res.items(): r["%s_K%d" % (k, K)] = v
        np.savez(dst + ".tmp", **r); os.replace(dst + ".tmp.npz", dst)
    per = {}
    for K in KS:
        d = {k[:-len("_K%d" % K)]: v for k, v in r.items() if k.endswith("_K%d" % K)}
        d["Wnext"] = W[K+1] if K+1 < D else None
        per[(K, "full")] = analyse(d, K)
        for rr in RS: per[(K, rr)] = analyse(d, K, rr)
    RES.append((nm, per))
    a = per[(14, "full")]
    lg("%-20s готово · K14: ‖ΔC‖/‖C‖ off %.2f%% (🔒гаус %.2f%%) · R²off 3: %.3f 3+4: %.3f (підлога %.3f) · R²diag 3+4 %.3f · R²v 3+4 %.3f"
       % (nm, a["rel_off"], a["rel_off_g"], a["R2off_3"], a["R2off_34"], a["floor_off"], a["R2diag_34"], a.get("R2v_34", np.nan)))

def mean(key, K, sub):
    v = [p[(K, sub)].get(key, np.nan) for _, p in RES]; return np.nanmean(v)
print("\n── залишок кроку C, усереднено по %d мережах ──" % len(RES))
print("%-22s | %s" % ("", "   ".join("K=%-2d" % K for K in KS)))
for key, lab in [("rel_off", "‖ΔC‖/‖C‖ позадіаг, %"), ("rel_off_g", "  🔒 гаусів контроль, %"), ("rel_sd_err", "похибка sd' від ΔC, %"),
                 ("R2off_3", "R² off, 3-й порядок"), ("R2off_34", "R² off, 3+4"), ("floor_off", "  підлога (гаус) off"),
                 ("R2diag_3", "R² diag, 3"), ("R2diag_34", "R² diag, 3+4"), ("floor_diag", "  підлога diag"),
                 ("R2v_3", "R² функціонал v, 3"), ("R2v_34", "R² функціонал v, 3+4"), ("floor_v", "  підлога v")]:
    print("%-22s | %s" % (lab, "   ".join("%6.3f" % mean(key, K, "full") for K in KS)))
print("\n── те саме з кумулянтами У ПІДПРОСТОРІ (сирі моменти q, істинне s) ──")
for sub in RS:
    for key, lab in [("R2off_3", "R² off 3"), ("R2off_34", "R² off 3+4"), ("R2v_34", "R² v 3+4")]:
        print("r=%-4d %-16s | %s" % (sub, lab, "   ".join("%6.3f" % mean(key, K, sub) for K in KS)))
lg("ГОТОВО")
print("DONE_CSTEP")
