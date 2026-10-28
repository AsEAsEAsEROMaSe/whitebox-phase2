"""01.10 («глянь туди, де стіна»): чи істинний зріз κ₄(p)_{aabc} на 32 нейронах є функцією ПАРНИХ величин трійки (a;b,c)?
Ціль стандартизована: t = κ₄_{aabc}/(σ_a²σ_bσ_c). Ознаки: стандартизовані m, C, D21 (усі орієнтації), K22, K31, D3, K4 трійки.
Моделі: рідж на добутках до 3-го степеня; GBM. Крос: навчання на двох мережах, тест на третій. Передбачення: крос-R² ≤0.3 (шанс 65%)."""
import sys, itertools, numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
m = 32
def load(i, ls):
    D = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls)); Nh = float(D["Nh"]); a = D["a"]
    g = lambda k: (D["0_" + k] + D["1_" + k]) / (2 * Nh)
    e = g("q1"); R2 = g("q2"); R3 = g("q3").reshape(m, m, m); R4 = g("q4").reshape(m, m, m, m)
    C2 = R2 - np.outer(e, e)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    k4 = R4 - sum(np.einsum("ab,cd->abcd", C2, C2).transpose(np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    mean = a + e; s = np.sqrt(np.diag(C2))
    C = C2 / np.outer(s, s); D21 = np.einsum("aac->ac", C3) / (s[:, None]**2 * s[None, :]); D3 = np.einsum("aaa->a", C3) / s**3
    K22 = np.einsum("aabb->ab", k4) / np.outer(s**2, s**2); K31 = np.einsum("aaab->ab", k4) / (s[:, None]**3 * s[None, :]); K4 = np.einsum("aaaa->a", k4) / s**4
    mu = mean / s
    rows, y = [], []
    for a_ in range(m):
        for b in range(m):
            for c in range(b + 1, m):
                if a_ in (b, c): continue
                rows.append([mu[a_], mu[b] + mu[c], mu[b] * mu[c], C[a_, b] + C[a_, c], C[a_, b] * C[a_, c], C[b, c],
                             D21[a_, b] + D21[a_, c], D21[b, a_] + D21[c, a_], D21[b, c] + D21[c, b], D21[a_, b] * D21[a_, c],
                             K22[a_, b] + K22[a_, c], K22[b, c], K31[a_, b] + K31[a_, c], K31[b, a_] + K31[c, a_], K31[b, c] + K31[c, b],
                             D3[a_], D3[b] + D3[c], K4[a_], K4[b] + K4[c],
                             C[a_, b] * C[b, c] + C[a_, c] * C[b, c], C[a_, b] * C[a_, c] * C[b, c]])
                y.append(k4[a_, a_, b, c] / (s[a_]**2 * s[b] * s[c]))
    return np.array(rows), np.array(y)
nets = {(0, 12): load(0, 12), (1, 12): load(1, 12), (2, 14): load(2, 14)}
def r2(y, p): return 1 - ((y - p)**2).sum() / ((y - y.mean())**2).sum()
for test in nets:
    Xtr = np.vstack([nets[k][0] for k in nets if k != test]); ytr = np.concatenate([nets[k][1] for k in nets if k != test]); Xte, yte = nets[test]
    sc = StandardScaler().fit(Xtr)
    out = []
    for deg in (1, 2, 3):
        pf = PolynomialFeatures(deg, include_bias=False); A = pf.fit_transform(sc.transform(Xtr)); B = pf.transform(sc.transform(Xte))
        rg = Ridge(alpha=1e-2).fit(A, ytr); out.append("рідж ст.%d: train %.3f test %.3f" % (deg, r2(ytr, rg.predict(A)), r2(yte, rg.predict(B))))
    gb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=31).fit(Xtr, ytr)
    out.append("GBM: train %.3f test %.3f" % (r2(ytr, gb.predict(Xtr)), r2(yte, gb.predict(Xte))))
    # контроль: перемішана ціль
    rng = np.random.default_rng(0); ysh = rng.permutation(ytr); gbs = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05).fit(Xtr, ysh)
    out.append("🔒перемішана: test %.3f" % r2(yte, gbs.predict(Xte)))
    print("тест на мережі %s (n=%d): " % (test, len(yte)) + " | ".join(out), flush=True)
print("DONE_WALL211", flush=True)
