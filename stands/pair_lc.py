"""Друга стадія пілоту (27.09): парні ознаки через C15 (пре-коваріація шару 15), Cpost15 і D21_15 (κ₃-зріз).
Один крок «передачі повідомлень»: для нейрона j агрегати Σ_k M_jk·f_k і Σ_k M_jk²·f_k для M ∈ {C15, Cpost15, D21, D21ᵀ}
і f ∈ {pred15, mu_pre, t, D3/σ³, var, 1} + понейронні 147 ознак pilot_lc. Та сама крива навчання і контроль.
Передбачення: приріст R²_test над понейронними ≤0.01 (шанс 60%): похибка живе у хвості спектра C, а згортка з C бачить голову."""
import os, sys, glob, time, numpy as np, torch
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
sys.argv = sys.argv[:1]; sys.path.insert(0, "/home/trim/wb2lean")
import pilot_lc as P   # завантажує data/names/test/pool за тими самими правилами (GEN=0/1)
rng = np.random.default_rng(0)
def pairfeats(f):
    d = np.load(f, allow_pickle=True); pre = d["pre15"]; n = pre.shape[0]
    C = torch.tensor(d["C15"].astype(np.float32)); Cp = torch.tensor(d["Cpost15"].astype(np.float32)); D = torch.tensor(d["D21_15"].astype(np.float32))
    for M in (C, Cp): M.fill_diagonal_(0.0)
    sd = torch.sqrt(torch.clamp(torch.tensor(pre[:, 1]), min=1e-12))
    fs = [torch.tensor(d["pred"][-1].astype(np.float32)), torch.tensor(pre[:, 0]), torch.tensor(pre[:, 2]), torch.tensor(pre[:, 4]), torch.tensor(pre[:, 1]), torch.ones(n)]
    F = torch.stack(fs, 1)
    out = []
    for M in (C, Cp, D, D.T):
        Mn = M / (sd[:, None] * sd[None, :])   # нормована (кореляційна) версія
        out += [M @ F, (M * M) @ F, Mn @ F, (Mn * Mn) @ F, (Mn ** 3) @ F]
    return torch.cat(out, 1).numpy()
t0 = time.time(); PF = [pairfeats(f) for f in P.files]; print("парні ознаки", PF[0].shape[1], "%.0f с" % (time.time() - t0), flush=True)
data2 = [(np.concatenate([X, pf], 1), r, p, g) for (X, r, p, g), pf in zip(P.data, PF)]
def evaluate(model, dat, tr, te, ctrl=None):
    Xtr = np.concatenate([dat[i][0] for i in tr]); ytr = np.concatenate([dat[i][1] for i in tr])
    if ctrl == "shuffle": perm = rng.permutation(len(tr)); ytr = np.concatenate([dat[tr[j]][1] for j in perm])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12; ysd = ytr.std(); model.fit((Xtr - mu) / sd, ytr / ysd)
    ss_res = ss_tot = 0.0; ratios = []
    for i in te:
        X, r = dat[i][0], dat[i][1]; rh = model.predict((X - mu) / sd) * ysd
        ss_res += np.sum((r - rh) ** 2); ss_tot += np.sum(r ** 2); ratios.append(np.mean((r - rh) ** 2) / np.mean(r ** 2))
    return 1 - ss_res / ss_tot, float(np.exp(np.mean(np.log(ratios))))
mk = lambda: HGB(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0)
sizes = [n for n in [20, 40, 80, 160, 320, 640] if n <= len(P.pool)]
for n in sizes:
    tr = P.pool[:n]
    r2a, ra = evaluate(mk(), P.data, tr, P.test); r2b, rb = evaluate(mk(), data2, tr, P.test); r2s, rs = evaluate(mk(), data2, tr, P.test, ctrl="shuffle")
    print("n=%3d понейронні R² %+.4f ratio %.4f | +парні R² %+.4f ratio %.4f | 🔒перемішана %+.4f" % (n, r2a, ra, r2b, rb, r2s), flush=True)
print("DONE_PAIR")
