"""01.10: фіксовані коефіцієнти парної функції зрізу κ₄(p)_{aabc} (рідж ст.2 на 21 стандартизованій ознаці з блоків ЛАНЦЮГА:
mu, var, C, D21, D3; без K22/K31/K4) для кожного LOO-фолду; варіанти: pairs (без носія) і pairs+sea3 (носій як ознака).
Вихід: wb2data/pairfit_<fold>.npz (mean, scale, poly-степені, coef, intercept) + R²_test і народження на референсі."""
import sys, numpy as np, torch
sys.path[:0] = ["/home/trim/wb2lean"]
sys.argv = [sys.argv[0], "chain_nok4"]
exec(open("/home/trim/wb2lean/wall211c.py").read().split("for test in nets:")[0])   # завантаження nets, birth_err, tensor_from, r2
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
for test in nets:
    trs = [nets[k] for k in nets if k != test]; te = nets[test]
    Xtr = np.vstack([t["X"] for t in trs]); ytr = np.concatenate([t["y"] for t in trs]); ctr = np.vstack([t["cf"][:, :1] for t in trs])
    sc = StandardScaler().fit(Xtr); pf = PolynomialFeatures(2, include_bias=False); A = pf.fit_transform(sc.transform(Xtr)); B = pf.transform(sc.transform(te["X"]))
    for nm, Atr, Bte in (("pairs", A, B), ("pairs_sea3", np.hstack([A, ctr]), np.hstack([B, te["cf"][:, :1]]))):
        rg = Ridge(alpha=1e-2).fit(Atr, ytr); p = rg.predict(Bte)
        np.savez("/home/trim/wb2data/pairfit_%s_%d.npz" % (nm, test[0]), mean=sc.mean_, scale=sc.scale_, powers=pf.powers_, coef=rg.coef_, intercept=rg.intercept_)
        print("фолд тест %s %-10s R²_test %.3f  народження: %.4f" % (test, nm, r2(te["y"], p), birth_err(te, tensor_from(te, p))), flush=True)
print("DONE_PAIRFIT", flush=True)
