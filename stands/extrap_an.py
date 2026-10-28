import glob, numpy as np
D = {int(f.split("_")[-1][:-4]): np.load(f) for f in sorted(glob.glob("/home/trim/wb2data/extrap_*.npz"))}
ids = sorted(D); print("мережі", ids)
base = np.mean([np.mean((D[i]["reg"]-D[i]["gt"])**2) for i in ids]); print("reg MSE %.4e" % base)
for x in ["dense", "simple", "nok4", "nofb"]:
    E = {i: D[i]["gt"]-D[i]["reg"] for i in ids}; Dx = {i: D[i]["reg"]-D[i][x] for i in ids}
    cos = np.mean([E[i]@Dx[i]/np.linalg.norm(E[i])/np.linalg.norm(Dx[i]) for i in ids])
    orac = np.mean([np.mean((E[i]-(E[i]@Dx[i]/(Dx[i]@Dx[i]))*Dx[i])**2) for i in ids])   # β на мережу, з істини
    loo = []; fr = []
    for i in ids:
        o = [j for j in ids if j != i]; b = sum(E[j]@Dx[j] for j in o)/sum(Dx[j]@Dx[j] for j in o)
        loo.append(np.mean((E[i]-b*Dx[i])**2)); k = ids[(ids.index(i)+1) % len(ids)]; fr.append(np.mean((E[i]-b*Dx[k])**2))
    print("%-7s cos(e,d) %+.3f | оракул β/мережу ×%.3f | LOO спільний β ×%.3f | 🔒чужа d ×%.3f" % (x, cos, base/orac, base/np.mean(loo), base/np.mean(fr)))
