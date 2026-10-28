"""20.10.2026 white-box-16: безкоштовні постпоправки на виході sb2 на 100 mini (дампи + LT2 у torch):
(a) спільний множник на все передбачення; (b) окремі a·pred + b·u; (c) b лише на u. Крос: фіт на 50 парних, оцінка на 50 непарних і навпаки."""
import glob, sys, numpy as np, torch
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
from lt2torch import LT2
torch.set_num_threads(1)
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
m = LT2("/home/trim/wb2lean/lt2re/lt2_shipped.npz", "cuda")
def wseed(s):
    wss, _, _ = np.random.SeedSequence(int(s)).spawn(3); r = np.random.default_rng(wss)
    return [(r.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
P, U, T = [], [], []
for f in sorted(glob.glob("/home/trim/wb2data/lt2dump/mini/mini_*.npz")):
    z = np.load(f); k = int(z["idx"])
    W = [torch.from_numpy(w).cuda()[None] for w in wseed(z["seed"])]
    with torch.no_grad():
        _, us = m.run(torch.from_numpy(z["X"]).cuda()[None], torch.from_numpy(z["Phi"]).cuda()[None], W)
    P.append(z["pred"][-1].astype(np.float64)); U.append(us[-1][0].double().cpu().numpy()); T.append(t["all_layer_means"][k][-1].astype(np.float64))
P, U, T = map(np.array, (P, U, T))
np.savez("/home/trim/wb2data/lt2train/mini_PUT.npz", P=P, U=U, T=T)
base = np.mean((P + U - T) ** 2)
print("off %.4e on %.4e ratio %.4f" % (np.mean((P - T) ** 2), base, base / np.mean((P - T) ** 2)))
for name, idx_fit, idx_ev in (("A->B", slice(0, None, 2), slice(1, None, 2)), ("B->A", slice(1, None, 2), slice(0, None, 2))):
    p, u, tt = P[idx_fit].ravel(), U[idx_fit].ravel(), T[idx_fit].ravel()
    al = np.dot(p + u, tt) / np.dot(p + u, p + u)
    ab = np.linalg.lstsq(np.stack([p, u], 1), tt, rcond=None)[0]
    bu = np.dot(u, tt - p) / np.dot(u, u)
    pe, ue, te = P[idx_ev], U[idx_ev], T[idx_ev]
    b0 = np.mean((pe + ue - te) ** 2)
    print("%s alpha %.6f ×%.4f | a,b %.6f %.4f ×%.4f | b_u %.4f ×%.4f" % (name, al, b0 / np.mean((al * (pe + ue) - te) ** 2),
          ab[0], ab[1], b0 / np.mean((ab[0] * pe + ab[1] * ue - te) ** 2), bu, b0 / np.mean((pe + bu * ue - te) ** 2)))
