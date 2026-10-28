"""21.10.2026 white-box-f9 (облік на прохання white-box-89): яку частку оракула (m,C)@14 (×9.2) пояснюють κ3/κ4 передактивацій.
Справжні c3 (D3), c4 (g4row) на шарах 1..15 і D21 (κ(a,a,b) передактивацій) на шарах 1..14, МК: c3/c4 2^24 (сід 22000+k),
D21 2^23 (сід 33000+k, точне центрування). Режими: base, c3, c4, d21, c3+c4, c3+d21, c3+c4+d21.
ПЕРЕДБАЧЕННЯ: c3+c4+d21 ≈ ×2.3…2.6 (канали нададитивні), тобто κ3/κ4 пояснюють ≲ третину log(9.2); решта на переносі (m,C).
usage: k34acc.py <k>"""
import os, sys, types, importlib.util, math, json
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k = int(sys.argv[1]); L21 = tuple(range(1, 15)); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"
c34f = "/home/trim/wblogs/c34_n%d.npz" % k
if not os.path.exists(c34f):
    W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(22000 + k)
    S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}; n = 0
    while n < 2 ** 24:
        h = torch.randn(B, 1024, generator=gg, device=dev)
        for l in range(16):
            p = h @ W[l]
            if l not in sh: sh[l] = p.mean(0)
            x = (p - sh[l]).double()
            S[l, 0] += x.sum(0); S[l, 1] += (x * x).sum(0); S[l, 2] += (x ** 3).sum(0); S[l, 3] += (x ** 4).sum(0)
            h = torch.relu(p)
        n += B
    a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    var = m2 - m1 ** 2
    np.savez(c34f, c3=m3 - 3 * m1 * m2 + 2 * m1 ** 3, c4=m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2)
    del W, S; torch.cuda.empty_cache()
z = np.load(c34f); c3, c4 = z["c3"], z["c4"]
d21f = "/home/trim/wblogs/d21t_all_n%d.npy" % k
if os.path.exists(d21f):
    KT = np.load(d21f, mmap_mode="r")
else:
    W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(33000 + k)
    KT = np.zeros((16, 1024, 1024), dtype=np.float32)
    for grp in (L21[:7], L21[7:]):   # two passes, 7 layers each (GPU memory)
        S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in grp}; S2 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in grp}
        G = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in grp}; Q = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in grp}
        sh = {}; n = 0; gg.manual_seed(33000 + k)
        while n < 2 ** 23:
            h = torch.randn(B, 1024, generator=gg, device=dev)
            for l in range(max(grp) + 1):
                p = h @ W[l]
                if l in G:
                    if l not in sh: sh[l] = p.mean(0)
                    x = p - sh[l]; S1[l] += x.double().sum(0); S2[l] += (x.double() ** 2).sum(0)
                    G[l] += ((x * x).T @ x).double(); Q[l] += (x.T @ x).double()
                h = torch.relu(p)
            n += B
        for l in grp:
            m1 = (S1[l] / n).cpu().numpy(); m2 = (S2[l] / n).cpu().numpy(); M = (G[l] / n).cpu().numpy(); Sm = (Q[l] / n).cpu().numpy()
            Kl = M - np.outer(m2, m1) - 2 * Sm * m1[:, None] + 2 * np.outer(m1 ** 2, m1); np.fill_diagonal(Kl, 0.0); KT[l] = Kl
        del G, Q, S1, S2; torch.cuda.empty_cache()
    np.save(d21f, KT); del W; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
A = tuple(range(1, 16))
c3g = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]; c4g = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
KTg = {l: cp.asarray(np.ascontiguousarray(KT[l])) for l in L21}
def run(m3=False, m4=False, m21=False):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if m3: o.update(D3set=A, c3t=c3g)
    if m4: o.update(G4set=A, c4t=c4g)
    if m21: o.update(D21set=L21, d21t=KTg, lam21=1.0)
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
b = run(); res = {"k": k, "base": b}
for nm, fl in (("c3", (1, 0, 0)), ("c4", (0, 1, 0)), ("d21", (0, 0, 1)), ("c3+c4", (1, 1, 0)), ("c3+d21", (1, 0, 1)), ("c3+c4+d21", (1, 1, 1))):
    res[nm] = run(*map(bool, fl))
print("net %d base %.4e " % (k, b) + " ".join("%s ×%.3f" % (nm, b / res[nm]) for nm in res if nm not in ("k", "base")), flush=True)
json.dump(res, open("/home/trim/wblogs/k34acc_n%d.json" % k, "w"))
