"""21.10.2026 white-box-f9: вихідна цінність накопиченого моря. МК (2^22): стан передактивацій (μ, C) на всіх шарах, справжній c4.
sea_L (L=2..15) = Σ_{s=L−6}^{L−1} дерева шару s, згорнуті з M_{s→L}. У ланцюгу: g4row_L += β_L·sea_L, β_L = МНК (c4_true − g4row) на sea
(по живих нейронах цієї ж мережі: оракул формули, верхня межа), і β з іншої мережі (крос). Також c4_true (стеля каналу).
ПЕРЕДБАЧЕННЯ: own ×1.05…1.12, крос ×1.03…1.08, стеля c4 ×1.14…1.25. Шанс крос ≥ ×1.05: 30%.
usage: seaorac.py <k> [beta_from.npy]"""
import os, sys, math, types, importlib.util
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False); torch.set_num_threads(1)
k = int(sys.argv[1]); BF = sys.argv[2] if len(sys.argv) > 2 else None; N = 2 ** 22; B = 8192; dev = "cuda"; DEP = 6
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
cache = "/home/trim/wblogs/sea_n%d.npz" % k
if os.path.exists(cache):
    zz = np.load(cache); SEA, C4 = zz["sea"], zz["c4"]
else:
    W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(45000 + k)
    S1 = torch.zeros(16, 1024, dtype=torch.float64, device=dev); Q = torch.zeros(16, 1024, 1024, dtype=torch.float32, device=dev)
    P = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); n = 0; sh = {}
    while n < N:
        h = torch.randn(B, 1024, generator=gg, device=dev)
        for l in range(16):
            p = h @ W[l]
            if l not in sh: sh[l] = p.mean(0)
            z = p - sh[l]; S1[l] += z.double().sum(0); Q[l] += z.T @ z
            zd = z.double()
            for i in range(4): P[l, i] += (zd ** (i + 1)).sum(0)
            h = torch.relu(p)
        n += B
    Wd = W.double(); st = {}
    for s in range(16):
        m1 = S1[s] / n; C = Q[s].double() / n - torch.outer(m1, m1); mu = m1 + sh[s].double(); sd = torch.sqrt(torch.diag(C)); t = mu / sd
        phi = torch.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + torch.erf(t / math.sqrt(2)))
        rho = C / torch.outer(sd, sd); rho.fill_diagonal_(0.0)
        st[s] = (sd * Phi, sd * phi, -sd * t * phi, rho, Phi)
    SEA = np.zeros((16, 1024))
    for L in range(2, 16):
        M = Wd[L].clone(); acc = torch.zeros(1024, dtype=torch.float64, device=dev)
        for s in range(L - 1, max(L - 1 - DEP, -1), -1):
            a1, a2, a3, rho, Phi = st[s]
            V = rho @ (M * a1[:, None]); U = M * a2[:, None] * V
            acc += 4.0 * torch.sum(M * a3[:, None] * V ** 3, 0) + 12.0 * torch.sum(U * (rho @ U), 0)
            M = Wd[s] @ (Phi[:, None] * M)
        SEA[L] = acc.cpu().numpy()
    q = (P / n).cpu().numpy(); q1, q2, q3, q4 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    var = q2 - q1 ** 2; C4 = q4 - 4 * q1 * q3 + 6 * q1 ** 2 * q2 - 3 * q1 ** 4 - 3 * var ** 2
    np.savez(cache, sea=SEA, c4=C4); del W, Q, Wd, st; torch.cuda.empty_cache()
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x.numpy()) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
LS = tuple(range(2, 16))
def run(c4t=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if c4t is not None: o.update(G4set=LS, c4t=[cp.asarray(x.astype(np.float32)) for x in c4t])
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in o["pre"]}
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2)), pre
b, pre = run()
G4 = np.stack([pre[l][3] if (l in pre and pre[l][3] is not None) else np.zeros(1024) for l in range(16)])
live = np.stack([np.abs(pre[l][2]) > 1e-12 if (l in pre and pre[l][2] is not None) else np.zeros(1024, bool) for l in range(16)])
beta = np.zeros(16)
for l in LS:
    m = live[l]; y = (C4[l] - G4[l])[m]; x = SEA[l][m]; beta[l] = x @ y / (x @ x)
    if l in (6, 10, 14):
        print("net %d L%d corr(c4−g4row, sea) %.3f  beta %.3f  corr(g4row,c4) %.3f corr(g4row+βsea,c4) %.3f" % (k, l, np.corrcoef(x, y)[0, 1], beta[l],
              np.corrcoef(G4[l][m], C4[l][m])[0, 1], np.corrcoef((G4[l] + beta[l] * SEA[l])[m], C4[l][m])[0, 1]), flush=True)
np.save("/home/trim/wblogs/sea_beta_n%d.npy" % k, beta)
res = {"own": run(G4 + beta[:, None] * SEA)[0], "own_half": run(G4 + 0.5 * beta[:, None] * SEA)[0], "c4true": run(np.where(live, C4, G4))[0]}
if BF:
    bx = np.load(BF); res["cross"] = run(G4 + bx[:, None] * SEA)[0]; res["cross_half"] = run(G4 + 0.5 * bx[:, None] * SEA)[0]
print("net %d base %.4e " % (k, b) + " ".join("%s ×%.4f" % (kk, b / v) for kk, v in res.items()), flush=True)
