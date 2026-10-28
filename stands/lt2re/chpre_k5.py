"""chpre_k5 (c2, 22.10.2026): оракул справжнього понейронного κ5 передактивацій (МК 2^24) у кроці ReLU
(Еджворт 5-го порядку: dE[ReLU] = -κ5 (t^3-3t) φ /(120 σ^4), dE[ReLU^2] = κ5 (t^2-1) φ/(60 σ^3)).
ПЕРЕДБАЧЕННЯ: сам κ5 на ланцюгових c3/c4 дає ×1.0…1.1; поверх істинних c3,c4 (both_all ×1.9…2.1) додає ≤×1.2. Шанс >×1.3: 15%.
usage: chpre_k5.py <k> <out.json>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
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
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(22000 + k)
S = torch.zeros(16, 5, 1024, dtype=torch.float64, device=dev); sh = {}; n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = (p - sh[l]).double(); x2 = x * x
        S[l, 0] += x.sum(0); S[l, 1] += x2.sum(0); S[l, 2] += (x2 * x).sum(0); S[l, 3] += (x2 * x2).sum(0); S[l, 4] += (x2 * x2 * x).sum(0)
        h = torch.relu(p)
    n += B
a = (S / n).cpu().numpy(); m1, m2, m3, m4, m5 = a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4]
var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4; c4 = mu4 - 3 * var ** 2
mu5 = m5 - 5 * m1 * m4 + 10 * m1 ** 2 * m3 - 10 * m1 ** 3 * m2 + 4 * m1 ** 5; c5 = mu5 - 10 * c3 * var
g5 = c5 / var ** 2.5
print("γ5 по шарах (медіана |γ5|):", " ".join("%.3f" % np.median(np.abs(g5[l])) for l in range(16)), flush=True)
del W, S; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
c3t = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]; c4t = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
c5t = [cp.asarray(c5[l].astype(np.float32)) for l in range(16)]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(D3set=(), G4set=(), K5set=(), **kw):
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": tuple(D3set), "G4set": tuple(G4set), "K5set": tuple(K5set), "c3t": c3t, "c4t": c4t, "c5t": c5t, **kw}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "base": run(), "g5_med": [float(np.median(np.abs(g5[l]))) for l in range(16)]}
A = range(1, 16)
for nm, d3s, g4s, k5s, kw in (("k5_all", (), (), A, {}), ("k5_mean_only", (), (), A, {"k5var": False}), ("k5_8-15", (), (), range(8, 16), {}),
                              ("k5_half", (), (), A, {"k5s": 0.5}), ("both_all", A, A, (), {}), ("both_k5_all", A, A, A, {})):
    res[nm] = run(d3s, g4s, k5s, **kw)
    print("net %d %-14s ×%.3f" % (k, nm, res["base"] / res[nm]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
