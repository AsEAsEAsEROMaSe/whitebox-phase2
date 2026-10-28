"""Пілот кривої навчання (27.09): чи зростає передбачуваність похибки виходу ланцюга з кількістю мереж.
Дані: state_<name>.npz (dumpstate.py) з істиною (mini-100) або з gen_truth. Ціль на нейрон: r = gt15 − pred15.
Ознаки на нейрон: пре/пост-стан шарів 10..15 (16+5 на шар), ознаки W шарів 12..15, pred шарів 12..15, lam.
Моделі: HistGradientBoosting і Ridge; test = фіксовані 20 мереж; train n ∈ {10,20,40,80} (і далі gen).
Метрики: R²_test на r, і MSE_після/MSE_до = mean((r−r̂)²)/mean(r²) помережно (геом). Контролі: перемішана ціль
(r іншої мережі з тими самими ознаками) і ціль після зняття помережного масштабу.
Передбачення: R²_test ≤0.05 на всіх n і без зростання (шанс 65%) ⟹ масштаб даних не важіль."""
import sys, glob, numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0)
import os, re
files = sorted(glob.glob("/home/trim/wb2data/gen/state_*.npz"))
GEN = os.environ.get("GEN", "0") == "1"   # GEN=1: test = усі mini-100 (істина 1e9), train = gen-мережі з готовою істиною
if not GEN: files = [f for f in files if not re.match(r"state_g\d{5}\.npz$", os.path.basename(f))]   # GEN=0: лише mini-100
if GEN: files = [f for f in files if (not re.match(r"state_g\d{5}\.npz$", os.path.basename(f))) or os.path.exists("/home/trim/wb2data/gen/truth_%05d.npz" % int(os.path.basename(f)[7:12]))]
def feats(d):
    L15 = d["pred"].shape[0] - 1; n = d["pred"].shape[1]; cols = []
    for l in range(10, 16):
        cols.append(d["pre%d" % l]); cols.append(d["post%d" % l])
    for l in range(12, 16): cols.append(d["wfeat"][l]); cols.append(d["pred"][l][:, None])
    X = np.concatenate(cols, 1).astype(np.float64)
    X = np.concatenate([X, np.full((n, 1), d["lam"][-1])], 1)
    return X
def load(f):
    d = np.load(f, allow_pickle=True)
    if "gt" in d: gt = d["gt"]
    else:
        name = str(d["name"]); t = np.load("/home/trim/wb2data/gen/truth_%05d.npz" % int(name[1:])); gt = t["alm"]
    return feats(d), gt[-1] - d["pred"][-1], d["pred"][-1], gt[-1]
data = [load(f) for f in files]; names = [f.split("state_")[1][:-4] for f in files]
print("мереж", len(data), "ознак", data[0][0].shape[1], flush=True)
# базові числа
mse0 = np.array([np.mean(r ** 2) for _, r, _, _ in data]); print("MSE ланцюга сер %.3e  геом %.3e" % (mse0.mean(), np.exp(np.log(mse0).mean())))
if GEN:
    isg = [bool(re.match(r"g\d{5}$", nm)) for nm in names]; test = np.array([i for i, g in enumerate(isg) if not g]); pool = np.array([i for i, g in enumerate(isg) if g])
    print("GEN-режим: test = %d офіційних, pool = %d згенерованих" % (len(test), len(pool)), flush=True)
else:
    idx = rng.permutation(len(data)); test = idx[:20]; pool = idx[20:]
def evaluate(model, tr, te, target="r", ctrl=None):
    Xtr = np.concatenate([data[i][0] for i in tr]); ytr = np.concatenate([data[i][1] for i in tr])
    if ctrl == "shuffle":   # ціль від іншої мережі (тієї ж довжини)
        perm = rng.permutation(len(tr)); ytr = np.concatenate([data[tr[j]][1] for j in perm])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12; ysd = ytr.std()
    model.fit((Xtr - mu) / sd, ytr / ysd)
    ss_res = ss_tot = 0.0; ratios = []
    for i in te:
        X, r = data[i][0], data[i][1]; rh = model.predict((X - mu) / sd) * ysd
        ss_res += np.sum((r - rh) ** 2); ss_tot += np.sum(r ** 2); ratios.append(np.mean((r - rh) ** 2) / np.mean(r ** 2))
    return 1 - ss_res / ss_tot, float(np.exp(np.mean(np.log(ratios))))
sizes = [n for n in [10, 20, 40, 80, 160, 320, 640] if n <= len(pool)]
for n in sizes:
    tr = pool[:n]
    for lab, mk in [("HGB", lambda: HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)), ("Ridge", lambda: Ridge(alpha=10.0))]:
        t0 = time.time(); r2, ratio = evaluate(mk(), tr, test); r2s, ratios = evaluate(mk(), tr, test, ctrl="shuffle")
        print("n=%3d %-5s R²_test %+.4f  MSE_після/до %.4f | 🔒перемішана ціль R² %+.4f ratio %.4f  (%.0f с)" % (n, lab, r2, ratio, r2s, ratios, time.time() - t0), flush=True)
print("DONE_PILOT")
