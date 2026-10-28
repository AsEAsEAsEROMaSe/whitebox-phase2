import glob, numpy as np
A = {int(f.split("_")[-1][:-4]): np.load(f) for f in glob.glob("/home/trim/wb2data/extrap_*.npz")}
B = {int(f.split("_")[-1][:-4]): np.load(f) for f in glob.glob("/home/trim/wb2data/extrap2_*.npz")}
ids = sorted(set(A) & set(B)); print("мережі", ids)
P = {i: dict(**{k: A[i][k] for k in A[i].files}, **{k: B[i][k] for k in B[i].files}) for i in ids}
mse = lambda k: np.mean([np.mean((P[i][k]-P[i]["gt"])**2) for i in ids]); b0 = mse("reg"); bd = mse("dense")
print("reg %.4e dense %.4e" % (b0, bd))
def loo(base, dirs):
    r = []; g = []
    for i in ids:
        X = np.concatenate([np.stack([P[j][k]-P[j][base] for k in dirs], 1) for j in ids if j != i]); y = np.concatenate([P[j]["gt"]-P[j][base] for j in ids if j != i])
        s = np.linalg.lstsq(X, y, rcond=None)[0]; g.append(s)
        r.append(np.mean((P[i][base] + np.stack([P[i][k]-P[i][base] for k in dirs], 1)@s - P[i]["gt"])**2))
    return np.mean(r), np.mean(g, 0)
for k in ["reg", "k211", "regen", "B2", "wk22", "no31", "trunc"]:
    r, g = loo("dense", [k]); print("%-6s MSE сам %.4e | dense+γ(%s−dense) LOO %.4e ×%.3f над reg (×%.3f над dense) γ=%.2f" % (k, mse(k), k, r, b0/r, bd/r, g[0]))
r, g = loo("dense", ["k211", "regen", "B2", "wk22", "no31", "trunc"]); print("усі 6 разом LOO ×%.3f над reg" % (b0/r), np.round(g, 2))
