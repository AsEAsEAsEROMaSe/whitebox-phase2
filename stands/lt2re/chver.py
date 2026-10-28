"""chver (c2, 22.10.2026): перевірка LOO-β повним прогоном (нелінійно): δD3 = Σ β3_j sd3·X̃_j, δg4 = Σ β4_j sd4·X̃_j на 12..14.
usage: chver.py <k> <beta.npz> <out.json>"""
import os, sys, json, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, BETA, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; LS = (12, 13, 14)
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc_k5.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x.numpy()) for x in w]
zc = np.load("/home/trim/wblogs/chpair_n%d.npz" % k); zd = np.load("/home/trim/wblogs/pair/dir_n%d.npz" % k); zb = np.load(BETA)
beta = zb["beta_te%d" % k]; which = str(zb["which"])
b3 = beta[:12] if which in ("A3", "both") else np.zeros(12); b4 = (beta[12:] if which == "both" else (beta if which == "A4" else np.zeros(12)))
def feats(li):
    D3, g4, mu, var = zc["D3_%d" % li], zc["g4_%d" % li], zc["mu_%d" % li], zc["var_%d" % li]
    D21, C = zc["D21_%d" % li], zc["Coff_%d" % li]; t = mu / np.sqrt(var)
    X = np.stack([D3, g4, mu, var, t, D21.sum(1), np.abs(D21).sum(1), (C * C).sum(1), (D21 * C).sum(1), (D21 * D21).sum(1), D21.sum(0), C.sum(1)], 1).astype(np.float64)
    return (X - X.mean(0)) / (X.std(0) + 1e-12), D3.astype(np.float64), g4.astype(np.float64)
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(scale):
    c3t, c4t = {}, {}
    for i, li in enumerate(LS):
        X, D3, g4 = feats(li)
        c3t[li] = D3 + scale * float(zd["sd3"][i]) * (X @ b3); c4t[li] = g4 + scale * float(zd["sd4"][i]) * (X @ b4)
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": LS, "G4set": LS, "K5set": (),
                "c3t": [cp.asarray(c3t.get(l, np.zeros(1024)).astype(np.float32)) for l in range(16)], "c4t": [cp.asarray(c4t.get(l, np.zeros(1024)).astype(np.float32)) for l in range(16)]}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
base = run(0.0); res = {"k": k, "which": which, "base": base}
for s in (0.5, 1.0, 1.5):
    res["s%.1f" % s] = run(s); print("net %d %s scale %.1f ×%.3f" % (k, which, s, base / res["s%.1f" % s]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
