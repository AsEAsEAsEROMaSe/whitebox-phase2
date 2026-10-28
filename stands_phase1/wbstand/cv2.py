"""С115-и КОНТРОЛЬ: чи виживає ×2.33 без ВИТОКУ, і чи б'є воно НАШ перенос.

Три схеми на ТИХ САМИХ семплах (закон: нове проти тривіальної бази в одній таблиці):
  plain   mean(a_fin)                                   — без нічого
  transp  fm − v,  v₀ = mean(x₀) − σ/√(2π) (ТОЧНА істина), v ← (v@W)·gate
          — це ЯДРО нашого λ-блоку (аналітичний перенос)
  kwonIS  B фітується НА ТИХ САМИХ семплах  (з витоком, як cv1)
  kwonCF  2-фолдовий КРОС-ФІТИНГ: B з фолду −k застосовується до фолду k
          — леак-фрі, як заявляє kwon

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 kwonCF помітно слабший за kwonIS (витік був)                    70%
  P2 kwonCF усе одно б'є transp                                     55%
  P3 kwonCF ≈ 1.0 (увесь ефект був витоком)                         35%
  P4 оптимальний ridge для CF НАБАГАТО більший за 0.3               75%
"""
import numpy as np, pyarrow.parquet as pq, sys, json
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

NPAIR = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
NREP  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LAYS  = [4, 8, 11, 14]
RIDGES = [0.3, 3.0, 30.0]
FIN = 15
INV_S2PI = 0.3989422804014327

def closure_means(W):
    d, n = W.shape[0], W.shape[1]
    m = np.zeros(n); C = np.eye(n); MS = []
    for i in range(d):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(d))
        MS.append(m.copy())
        if i == d-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def forward(W, z):
    d = W.shape[0]
    cov = (z.T @ z)/z.shape[0]
    wz = np.linalg.inv(np.linalg.cholesky(cov)).T
    y = (z @ wz) @ W[0]
    x = np.concatenate([np.maximum(y, 0.0), np.maximum(-y, 0.0)], 0).astype(np.float32)
    A = {}
    # 🔑 наш аналітичний перенос: старт із ТОЧНО відомого середнього шару 0
    sg0 = np.sqrt(np.maximum((W[0]*W[0]).sum(0), 1e-30))
    v = x.mean(0).astype(np.float64) - sg0*INV_S2PI
    for i in range(1, d):
        p = x @ W[i].astype(np.float32)
        gate = (p > 0.0).mean(0).astype(np.float64)
        v = (v @ W[i]) * gate
        x = np.maximum(p, 0.0)
        if i in LAYS or i == d-1: A[i] = x
    return A, v

def fitB(D, F, ridge):
    G = (D.T @ D)/D.shape[0]
    G = G + ridge*np.trace(G)/G.shape[0]*np.eye(G.shape[0])
    return np.linalg.solve(G, (D.T @ F)/D.shape[0])

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 8
keys = [("plain", 0, 0.0), ("transp", 0, 0.0)]
for L in LAYS:
    for R in RIDGES: keys += [("kwonIS", L, R), ("kwonCF", L, R)]
acc = {k: [] for k in keys}
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16,1024,1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16,1024)
    MU = closure_means(W); tr = T[FIN]
    for r in range(NREP):
        rng = np.random.default_rng(90000 + 1000*j + r)
        z = rng.standard_normal((NPAIR, 1024)).astype(np.float32)
        A, v = forward(W, z)
        f = A[FIN].astype(np.float64); fm = f.mean(0); nr = f.shape[0]
        acc[("plain", 0, 0.0)].append(float(((fm - tr)**2).sum()))
        acc[("transp", 0, 0.0)].append(float(((fm - v - tr)**2).sum()))
        h = nr//2; idx = rng.permutation(nr); I0, I1 = idx[:h], idx[h:]
        for L in LAYS:
            a = A[L].astype(np.float64); am = a.mean(0); dev = am - MU[L]
            D = a - am; F = f - fm
            for R in RIDGES:
                acc[("kwonIS", L, R)].append(
                    float(((fm - fitB(D, F, R).T @ dev - tr)**2).sum()))
                # 🔒 крос-фітинг: B з ІНШОГО фолду
                est = np.zeros_like(fm)
                for Ia, Ib in ((I0, I1), (I1, I0)):
                    Ba = fitB(a[Ia] - a[Ia].mean(0), f[Ia] - f[Ia].mean(0), R)
                    est += f[Ib].mean(0) - Ba.T @ (a[Ib].mean(0) - MU[L])
                acc[("kwonCF", L, R)].append(float(((0.5*est - tr)**2).sum()))
    log("%2d" % j)
b = np.mean(acc[("plain", 0, 0.0)])
print("\nN=%d пар · реплік=%d" % (NPAIR, NREP))
print("%-8s %-4s %-6s %14s %8s" % ("схема", "шар", "ridge", "MSE", "VR"))
for k in keys:
    v_ = np.mean(acc[k])
    print("%-8s %-4s %-6s %14.6e %8.3f" % (k[0], k[1] or "-", k[2] or "-", v_, b/v_))
json.dump({str(k): list(map(float,v)) for k,v in acc.items()}, open("/home/trim/wblogs/cv2.json","w"))
log("ГОТОВО")
