"""01.10: зріз κ₄(p)_{aabc} як функція парних величин + носії (sea3, n²-море) на dumps 0 (ш12) і 2 (ш14): крос між мережами,
стеля шуму, і референсна похибка народження моря κ₃ з передбаченим зрізом (лише діаграма ((1,1,2),))."""
import sys, numpy as np, torch
sys.path[:0] = ["/home/trim/leanpkg", "/mnt/win_d/llm/White-Box/whest-starterkit/kprop_reference", "/home/trim/wb2lean"]
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import mlp_kprop.kprop_ds as kds
from mlp_kprop.kprop_ds import relu_kprop
from mlp_kprop.diagslice import DSTower
from mlp_kprop.kprop_harmonic import get_all_terms_iso as GATI
torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
T3 = GATI(3); T4 = GATI(4); m = 32
def cums(D, h):
    Nh = float(D["Nh"]); a = D["a"]
    g = (lambda k: (D["0_" + k] + D["1_" + k]) / (2 * Nh)) if h is None else (lambda k: D["%d_%s" % (h, k)] / Nh)
    e = g("q1"); R2 = g("q2"); R3 = g("q3").reshape(m, m, m); R4 = g("q4").reshape(m, m, m, m)
    C2 = R2 - np.outer(e, e)
    C3 = R3 - (R2[:, :, None]*e[None, None, :] + R2[:, None, :]*e[None, :, None] + R2[None, :, :]*e[:, None, None]) + 2*e[:, None, None]*e[None, :, None]*e[None, None, :]
    k4 = R4 - sum(np.einsum("ab,cd->abcd", C2, C2).transpose(np.argsort(p)) for p in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)])
    return a + e, C2, C3, k4
IDX = [(a_, b, c) for a_ in range(m) for b in range(m) for c in range(b + 1, m) if a_ not in (b, c)]
FEAT = sys.argv[1] if len(sys.argv) > 1 else "true"   # true | chain | chain_nok4
def load(i, ls):
    D = np.load("/home/trim/wb2data/brc_%d_%d.npz" % (i, ls))
    mean, C2, C3, k4 = cums(D, None); _, _, _, k4a = cums(D, 0); _, _, _, k4b = cums(D, 1)
    s = np.sqrt(np.diag(C2))
    if FEAT == "true":
        C = C2 / np.outer(s, s); D21 = np.einsum("aac->ac", C3) / (s[:, None]**2 * s[None, :]); D3 = np.einsum("aaa->a", C3) / s**3
        K22 = np.einsum("aabb->ab", k4) / np.outer(s**2, s**2); K31 = np.einsum("aaab->ab", k4) / (s[:, None]**3 * s[None, :]); K4 = np.einsum("aaaa->a", k4) / s**4; mu = mean / s
    else:
        Bc = np.load("/home/trim/wb2data/blocksS_%d_%d.npz" % (i, ls)); sc_ = np.sqrt(Bc["var"])
        C = Bc["C"] / np.outer(sc_, sc_); D21 = Bc["D21"] / (sc_[:, None]**2 * sc_[None, :]); D3 = Bc["D3"] / sc_**3; mu = Bc["mu"] / sc_
        if FEAT == "chain":
            K22 = Bc["K22"] / np.outer(sc_**2, sc_**2); K31 = Bc["K31"] / (sc_[:, None]**3 * sc_[None, :]); K4 = Bc["K4"] / sc_**4
        else:
            K22 = np.zeros((m, m)); K31 = np.zeros((m, m)); K4 = np.zeros(m)
    car = np.load("/home/trim/wb2data/seacarry_%d_%d.npz" % (i, ls))["carrier"]; c2 = torch.load("/home/trim/wb2data/carrier2_%d_%d.pt" % (i, ls))["sea"].numpy()
    rows, y, yn, cf = [], [], [], []
    for a_, b, c in IDX:
        rows.append([mu[a_], mu[b] + mu[c], mu[b] * mu[c], C[a_, b] + C[a_, c], C[a_, b] * C[a_, c], C[b, c],
                     D21[a_, b] + D21[a_, c], D21[b, a_] + D21[c, a_], D21[b, c] + D21[c, b], D21[a_, b] * D21[a_, c],
                     K22[a_, b] + K22[a_, c], K22[b, c], K31[a_, b] + K31[a_, c], K31[b, a_] + K31[c, a_], K31[b, c] + K31[c, b],
                     D3[a_], D3[b] + D3[c], K4[a_], K4[b] + K4[c], C[a_, b] * C[b, c] + C[a_, c] * C[b, c], C[a_, b] * C[a_, c] * C[b, c]])
        sc = s[a_]**2 * s[b] * s[c]
        y.append(k4[a_, a_, b, c] / sc); yn.append((k4a[a_, a_, b, c] - k4b[a_, a_, b, c]) / 2 / sc)
        cf.append([car[a_, a_, b, c] / sc, c2[a_, a_, b, c] / sc])
    return dict(X=np.array(rows), y=np.array(y), yn=np.array(yn), cf=np.array(cf), mean=mean, C2=C2, C3=C3, k4=k4, s=s, D=D)
nets = {(0, 12): load(0, 12), (1, 12): load(1, 12), (2, 14): load(2, 14)}
print("ознаки:", FEAT, flush=True)
def r2(y, p): return 1 - ((y - p)**2).sum() / ((y - y.mean())**2).sum()
def birth_err(net, K4t):
    mean, C2, C3 = [torch.tensor(net[k]) for k in ("mean", "C2", "C3")]; D = net["D"]; Nh = float(D["Nh"])
    h1 = torch.tensor((D["0_h1"] + D["1_h1"]) / (2 * Nh)); h2 = torch.tensor((D["0_h2"] + D["1_h2"]) / (2 * Nh)); h3 = torch.tensor((D["0_h3"] + D["1_h3"]).reshape(m, m, m) / (2 * Nh))
    tru3 = h3 - (h2[:, :, None]*h1[None, None, :] + h2[:, None, :]*h1[None, :, None] + h2[None, :, :]*h1[:, None, None]) + 2*h1[:, None, None]*h1[None, :, None]*h1[None, None, :]
    ar = torch.arange(m); A_, B_, C_ = torch.meshgrid(ar, ar, ar, indexing="ij"); sea = (A_ != B_) & (A_ != C_) & (B_ != C_); tru = tru3[sea]
    keep = lambda vp: vp in T3[(1, 1, 1)] or vp in {((1, 1, 2),), ((1, 2, 1),), ((2, 1, 1),)}
    def patched(k_max, d_max=None, use_mean_var=False):
        return {ip: ({vp: (c if keep(vp) else 0.0) for vp, c in vd.items()} if ip == (1, 1, 1) else dict(vd)) for ip, vd in T4.items()}
    kds.get_all_terms_iso = patched
    x = relu_kprop(DSTower.from_tower({1: mean, 2: C2, 3: C3, 4: torch.tensor(K4t)}), k_max=4)[3].to_tensor()[sea]
    return float((x - tru).norm() / tru.norm())
def tensor_from(net, pred):
    """κ₄ на S лише з (a,a,b,c)-патерном (усі 3 положення подвоєної осі), решта 0."""
    K = np.zeros((m,) * 4); s = net["s"]
    for (a_, b, c), v in zip(IDX, pred):
        val = v * s[a_]**2 * s[b] * s[c]
        for T in (K,):
            T[a_, a_, b, c] = T[a_, a_, c, b] = T[b, c, a_, a_] = T[c, b, a_, a_] = val
            T[a_, b, a_, c] = T[a_, c, a_, b] = T[b, a_, c, a_] = T[c, a_, b, a_] = val
            T[a_, b, c, a_] = T[a_, c, b, a_] = T[b, a_, a_, c] = T[c, a_, a_, b] = val
    return K
for test in nets:
    trs = [nets[k] for k in nets if k != test]; te = nets[test]; train = [k for k in nets if k != test]
    tr = dict(X=np.vstack([t["X"] for t in trs]), y=np.concatenate([t["y"] for t in trs]), cf=np.vstack([t["cf"] for t in trs]))
    noise = float(np.linalg.norm(te["yn"]) / np.linalg.norm(te["y"] - te["y"].mean()))
    print("тест %s (train %s): шум цілі %.3f ⟹ стеля R² ≈ %.3f" % (test, train, noise, 1 - noise**2), flush=True)
    sc = StandardScaler().fit(tr["X"]); pf = PolynomialFeatures(2, include_bias=False)
    A = pf.fit_transform(sc.transform(tr["X"])); B = pf.transform(sc.transform(te["X"]))
    preds = {}
    rg = Ridge(alpha=1e-2).fit(A, tr["y"]); preds["пари (рідж ст.2)"] = rg.predict(B)
    for j, nm in enumerate(("sea3", "sea3+n²-море")):
        cfe = te["cf"][:, :j + 1].sum(1); preds["носій " + nm + " (як є)"] = cfe
    Atr = np.hstack([A, tr["cf"]]); Bte = np.hstack([B, te["cf"]])
    rg2 = Ridge(alpha=1e-2).fit(Atr, tr["y"]); preds["пари + носії (рідж)"] = rg2.predict(Bte)
    gb = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05).fit(np.hstack([tr["X"], tr["cf"]]), tr["y"]); preds["пари + носії (GBM)"] = gb.predict(np.hstack([te["X"], te["cf"]]))
    preds["істина"] = te["y"]; preds["нуль"] = np.zeros_like(te["y"])
    for nm, p in preds.items():
        print("   %-28s R²_test %.3f  народження моря κ₃ (референс, ((1,1,2),)): %.4f" % (nm, r2(te["y"], p), birth_err(te, tensor_from(te, p))), flush=True)
print("DONE_WALL211C", flush=True)
