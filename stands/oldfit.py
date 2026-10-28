"""Пункт 2, вирок: GBM на зріз старого ярусу. Train dumps 0-5, test 6-7. Без витоку (Sf, D3f виключені)."""
import numpy as np, time, sys
from sklearn.ensemble import HistGradientBoostingRegressor
FN = ["C","mu_i","mu_j","var_i","var_j","Sy_ij","Sy_ji","D3y_i","D3y_j","CC_had","CC","CSy","SyC","CSyT","C*D3i/v","C*D3j/v","C*mu_i","C*mu_j","Sf_ij","Sf_ji","L1","L2","L3","Lsl","layer"]
NFN = ["mu","var","D3y","D3f","diagCC","rowC2","Sy_row","Sy_col","absSy_row","d3rule","D3old_prev","layer"]
D = {i: np.load("/home/trim/wb2data/oldfeat_%d.npz" % i) for i in range(8)}
tr, te = [0,1,2,3,4,5], [6,7]
def cat(ids, key): return np.concatenate([D[i][key] for i in ids])
Xtr, ytr, Xte, yte = cat(tr,"feats"), cat(tr,"targ"), cat(te,"feats"), cat(te,"targ")
def report(name, pred, y, layer):
    out = []
    for l in sorted(set(layer.astype(int))):
        m = layer == l; r = y[m] - pred[m]; out.append("l%d: R² %.3f ε %.3f" % (l, 1 - (r*r).sum()/(y[m]**2).sum(), np.linalg.norm(r)/np.linalg.norm(y[m])))
    r = y - pred; print("%-28s ALL R² %.4f ε %.3f | %s" % (name, 1 - (r*r).sum()/(y*y).sum(), np.linalg.norm(r)/np.linalg.norm(y), " ".join(out)), flush=True)
lay = Xte[:, 24]
report("rule Lsl only (test)", Xte[:, 23], yte, lay)
report("young Sy_ij only (test)", 0*yte, yte, lay)
sets = {"all−Sf": [k for k in range(25) if k not in (18, 19)], "no rule, no Sf": [k for k in range(25) if k not in (18, 19, 20, 21, 22, 23)], "rule terms only": [20, 21, 22, 23, 24]}
sub = np.random.default_rng(0).choice(len(Xtr), min(len(Xtr), 3000000), replace=False)
for nm, cols in sets.items():
    t0 = time.time(); gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.1, max_leaf_nodes=63, early_stopping=True, validation_fraction=0.1, random_state=0)
    gb.fit(Xtr[sub][:, cols], ytr[sub]); print("fit %s %.0fs iters %d" % (nm, time.time()-t0, gb.n_iter_), flush=True)
    report("GBM %s (test)" % nm, gb.predict(Xte[:, cols]), yte, lay)
    report("GBM %s (train sub)" % nm, gb.predict(Xtr[sub][:, cols]), ytr[sub], Xtr[sub][:, 24])
# D3 старого ярусу
Ntr, nytr, Nte, nyte = cat(tr,"nfeat"), cat(tr,"ntarg"), cat(te,"nfeat"), cat(te,"ntarg")
nlay = Nte[:, 11]
report("D3: rule only (test)", Nte[:, 9], nyte, nlay)
for nm, cols in {"D3 all−D3f": [k for k in range(12) if k != 3], "D3 no rule": [k for k in range(12) if k not in (3, 9, 10)]}.items():
    gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, random_state=0).fit(Ntr[:, cols], nytr)
    report("GBM %s (test)" % nm, gb.predict(Nte[:, cols]), nyte, nlay)
print("DONE_OLDFIT")
