"""Чи є похибка народження κ₂₁ (Rb, pieces_useall) першим порядком по ІСТИННИХ парних κ₅-зрізах шару l−1?
Ознаки: F_ab = κ(a,b)[x,y] · E[∂^a relu²(p_x)]/a! · E[∂^b relu(p_y)]/b!, (a,b) ∈ {(4,1),(3,2),(2,3),(1,4)}; перенос як у народженні:
T[i,j] = Σ_xy (W_xi² W_yj + 2 W_xi W_yi W_xj) F[x,y]. Для порівняння той самий шаблон з κ(2,1)/κ(3,1)/κ(2,2) (4-й порядок, формула їх має).
🔒 контроль: F з перемішаними мітками нейронів. Критерій ДО: R²(Rb_13 | κ₅-ознаки) ≥ 0.3 при контролі ≈ 0; шанс 20%."""
import sys, math, numpy as np, torch
dev = "cuda"; D = [0, 1, 2]
def He(m, x):
    h0, h1 = torch.ones_like(x), x
    if m == 0: return h0
    for k in range(1, m): h0, h1 = h1, x*h1 - k*h0
    return h1
def wick_relu(k, mu, s):   # E[∂^k relu(p)], p~N(mu,s²)
    t = mu/s; phi = torch.exp(-0.5*t*t)/math.sqrt(2*math.pi); Phi = 0.5*(1+torch.erf(t/math.sqrt(2)))
    if k == 0: return s*(phi + t*Phi)
    if k == 1: return Phi
    return He(k-2, -t)*phi/s**(k-1)
def wick_relu2(k, mu, s):  # E[∂^k relu²(p)]
    t = mu/s; phi = torch.exp(-0.5*t*t)/math.sqrt(2*math.pi); Phi = 0.5*(1+torch.erf(t/math.sqrt(2)))
    if k == 0: return s*s*((1+t*t)*Phi + t*phi)
    if k == 1: return 2*s*(phi + t*Phi)
    if k == 2: return 2*Phi
    return 2*He(k-3, -t)*phi/s**(k-2)
res = {}
for i in D:
    K = np.load("/home/trim/wb2data/kmix_%d.npz" % i); P = np.load("/home/trim/wb2data/pieces_useall_%d.npz" % i)
    W = torch.tensor(np.load("/home/trim/wb2data/dumps_p2/mlp_%04d.npz" % i)["weights"], device=dev, dtype=torch.float64)
    kp = np.load("/home/trim/wb2data/kp_p2/mlp_%05d.npz" % i)
    for ls in [9, 12]:
        lt = ls + 1
        if "Rb_%d" % lt not in P.files: continue
        g = lambda k: torch.tensor(0.5*(K["h0_l%d_%s" % (ls, k)].astype(np.float64) + K["h1_l%d_%s" % (ls, k)].astype(np.float64)), device=dev)
        v = 0.5*(K["h0_l%d_v" % ls] + K["h1_l%d_v" % ls]); m = [torch.tensor(v[k]/v[0], device=dev) for k in range(6)]
        E11, E21, E12, E31, E22, E41, E32 = (g(k) for k in ("11", "21", "12", "31", "22", "41", "32"))
        X2 = m[2][:, None]; Y2 = m[2][None, :]; X3 = m[3][:, None]
        k21 = E21; k31 = E31 - 3*X2*E11; k22 = E22 - X2*Y2 - 2*E11**2
        k41 = E41 - 6*X2*E21 - 4*E11*X3; k32 = E32 - 3*X2*E12 - 6*E11*E21 - Y2*X3
        mu = torch.tensor(kp["pre_mean"][ls], device=dev, dtype=torch.float64) + m[1]; s = (m[2]-m[1]**2).sqrt()
        cum = {(4, 1): k41, (1, 4): k41.T, (3, 2): k32, (2, 3): k32.T, (2, 1): k21, (1, 2): k21.T, (3, 1): k31, (1, 3): k31.T, (2, 2): k22}
        Wn = W[lt]; W2 = Wn*Wn
        def tr(F): G = F @ Wn; return W2.T @ G + 2*((Wn*G).T @ Wn)
        perm = torch.randperm(Wn.shape[0], generator=torch.Generator().manual_seed(5)).to(dev)
        feats = {}
        for (a, b), kab in cum.items():
            F = kab * (wick_relu2(a, mu, s)/math.factorial(a))[:, None] * (wick_relu(b, mu, s)/math.factorial(b))[None, :]
            F.fill_diagonal_(0.0); feats["%d%d" % (a, b)] = tr(F)
            if a + b == 5: Fp = F[perm][:, perm]; feats["🔒%d%d" % (a, b)] = tr(Fp)
        Rb = torch.tensor(P["Rb_%d" % lt], device=dev, dtype=torch.float64)
        res[(i, lt)] = (Rb, feats)
off = None
def r2(y, X): sol = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]; return 1 - float(((y - X@sol)**2).sum()/(y*y).sum()), sol
for lt in [10, 13]:
    keys = [k for k in res if k[1] == lt]
    if not keys: continue
    print("== Rb_%d (джерело шар %d), мережі %s" % (lt, lt-1, [k[0] for k in keys]))
    n = res[keys[0]][0].shape[0]; off = ~torch.eye(n, dtype=torch.bool, device=dev)
    groups = {"κ5 усі 4": ["41", "14", "32", "23"], "🔒κ5 перемішані": ["🔒41", "🔒14", "🔒32", "🔒23"], "κ4-зрізи (є у формулі)": ["21", "12", "31", "13", "22"],
              "41": ["41"], "14": ["14"], "32": ["32"], "23": ["23"], "κ4+κ5": ["21", "12", "31", "13", "22", "41", "14", "32", "23"]}
    for gname, cols in groups.items():
        ins = []; loo = []
        for k in keys:
            y = res[k][0][off]; X = torch.stack([res[k][1][c][off] for c in cols], 1); ins.append(r2(y, X)[0])
            o = [q for q in keys if q != k]
            Xo = torch.cat([torch.stack([res[q][1][c][off] for c in cols], 1) for q in o]); yo = torch.cat([res[q][0][off] for q in o])
            _, sol = r2(yo, Xo); loo.append(1 - float(((y - X@sol)**2).sum()/(y*y).sum()))
        print("  %-24s R² в мережі %s | LOO %s" % (gname, " ".join("%.3f" % x for x in ins), " ".join("%.3f" % x for x in loo)), flush=True)
