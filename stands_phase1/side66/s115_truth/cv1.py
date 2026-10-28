"""С115-и: РЕГРЕСІЙНА КОНТРОЛЬНА ЗМІННА (`kwon` §9) ПРОТИ НАШОГО АНАЛІТИЧНОГО ПЕРЕНОСУ.

Розрив, який міряємо: у нього ×2.19-2.39 на цьому каналі, у нас ×1.14.
Різниця схем:
   kwon  μ̂ = mean(a_fin) − Bᵀ(mean(a_ℓ) − μ_an^ℓ),  B = ridge З ВИБІРКИ, по мережі
   наш   v пропагується АНАЛІТИЧНО крізь (W · gate), ваги ОФЛАЙН
⟹ ми апроксимуємо B лінеаризацією мережі, він МІРЯЄ B.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 VR має максимум у СЕРЕДНІХ шарах (у нас 16 глибина ⟹ ℓ≈3-6)      70%
  P2 реалізовний VR на піку >= ×1.5                                   50%
  P3 реалізовний VR на піку >= ×2.0                                   30%
  P4 оракульний VR на передостанньому >= ×50                          65%
  P5 крива «оракул ГЛИБОКО добре, замикальний центр ГЛИБОКО погано»    75%
     (підпис `mliston`/`kwon`: неточний центр ⟹ глибше гірше)
"""
import numpy as np, pyarrow.parquet as pq, sys, json, time
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

NPAIR = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
NREP  = int(sys.argv[3]) if len(sys.argv) > 3 else 6
LAYS  = [1, 2, 3, 4, 6, 8, 11, 14]
RIDGE = 0.3

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
    """антитетика + вибілювання, як у оцінювачі. Повертає активації потрібних шарів."""
    d = W.shape[0]
    cov = (z.T @ z)/z.shape[0]
    L = np.linalg.cholesky(cov)
    wz = np.linalg.inv(L).T
    y = (z @ wz) @ W[0]
    x = np.concatenate([np.maximum(y, 0.0), np.maximum(-y, 0.0)], 0).astype(np.float32)
    A = {0: x} if 0 in LAYS else {}
    for i in range(1, d):
        x = np.maximum(x @ W[i].astype(np.float32), 0.0)
        if i in LAYS or i == d-1: A[i] = x
    return A

tb = pq.read_table(PQ); NN = int(sys.argv[1]) if len(sys.argv) > 1 else 8
FIN = 15
acc = {("plain", 0): []}
for L in LAYS:
    for md in ("an", "or"): acc[(md, L)] = []
for j in range(min(NN, tb.num_rows)):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    MU = closure_means(W); tr = T[FIN]
    for r in range(NREP):
        rng = np.random.default_rng(90000 + 1000*j + r)
        z = rng.standard_normal((NPAIR, 1024)).astype(np.float32)
        A = forward(W, z)
        f = A[FIN].astype(np.float64); fm = f.mean(0)
        acc[("plain", 0)].append(float(((fm - tr)**2).sum()))
        for L in LAYS:
            a = A[L].astype(np.float64); am = a.mean(0)
            D = a - am; F = f - fm
            G = (D.T @ D)/D.shape[0]
            G = G + RIDGE*np.trace(G)/1024.0*np.eye(1024)
            B = np.linalg.solve(G, (D.T @ F)/D.shape[0])
            for md, ctr in (("an", MU[L]), ("or", T[L])):
                est = fm - B.T @ (am - ctr)
                acc[(md, L)].append(float(((est - tr)**2).sum()))
    log("%2d %s" % (j, tb["mlp_name"][j].as_py()))
b = np.mean(acc[("plain", 0)])
print("\nN=%d пар · реплік=%d · ridge=%.2f" % (NPAIR, NREP, RIDGE))
print("%-5s %14s %8s | %14s %8s" % ("шар", "ЗАМИК. центр", "VR", "ТОЧНИЙ центр", "VR"))
for L in LAYS:
    va, vo = np.mean(acc[("an", L)]), np.mean(acc[("or", L)])
    print("%-5d %14.6e %8.3f | %14.6e %8.3f" % (L, va, b/va, vo, b/vo))
print("%-5s %14.6e %8.3f" % ("нема", b, 1.0))
json.dump({str(k): list(map(float,v)) for k,v in acc.items()}, open("/home/trim/wblogs/cv1.json","w"))
log("ГОТОВО")
