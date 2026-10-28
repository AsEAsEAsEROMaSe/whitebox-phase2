import glob, numpy as np
D = {int(f.split("_")[-1][:-4]): np.load(f) for f in glob.glob("/home/trim/wb2data/knt_*.npz")}; ids = sorted(D)
names = sorted(set.intersection(*[set(D[i]["names"]) for i in ids])); print("мережі", ids, "груп", len(names))
def J(i): nm = list(D[i]["names"]); return np.stack([D[i]["J_%d" % nm.index(n)] for n in names], 1)
E = {i: D[i]["gt"]-D[i]["p0"] for i in ids}; Js = {i: J(i) for i in ids}
b0 = np.mean([np.mean(E[i]**2) for i in ids]); print("база MSE %.4e" % b0)
lin = {n: np.mean([D[i]["lin_%d" % list(D[i]["names"]).index(n)] for i in ids]) for n in names}
for lam in [0, 1e-3, 1e-2, 1e-1]:
    r = []; rf = []; th = []
    for i in ids:
        o = [j for j in ids if j != i]; A = sum(Js[j].T@Js[j] for j in o); b = sum(Js[j].T@E[j] for j in o)
        t = np.linalg.solve(A + lam*np.trace(A)/len(names)*np.eye(len(names)), b); th.append(t)
        r.append(np.mean((E[i]-Js[i]@t)**2)); k = ids[(ids.index(i)+1) % len(ids)]; rf.append(np.mean((E[i]-Js[k]@t)**2))
    th = np.array(th); print("ridge %.0e: LOO ×%.3f | 🔒чужий J ×%.3f | max|θ| %.2f" % (lam, b0/np.mean(r), b0/np.mean(rf), np.abs(th.mean(0)).max()))
# одиночні
print("одиночні LOO:")
res = []
for c, n in enumerate(names):
    r = []
    for i in ids:
        o = [j for j in ids if j != i]; t = sum(Js[j][:, c]@E[j] for j in o)/sum(Js[j][:, c]@Js[j][:, c] for j in o); r.append(np.mean((E[i]-Js[i][:, c]*t)**2))
    res.append((b0/np.mean(r), n, t, lin[n]))
for x in sorted(res, reverse=True)[:12]: print("  ×%.3f %-16s θ-1=%+.2f нелін %.3f" % x)
