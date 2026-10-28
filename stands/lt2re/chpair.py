"""chpair (c2, 22.10.2026): дамп для тесту «пари»: справжні c3/c4 передактивацій (МК 2^24) + стан ланцюга sb2 у природному
порядку на шарах 12..15: D3, g4row, mu, var, повні D21 і C_off. usage: chpair.py <k> <out.npz>"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192; LS = (12, 13, 14, 15)
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
S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}; n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = (p - sh[l]).double(); x2 = x * x
        S[l, 0] += x.sum(0); S[l, 1] += x2.sum(0); S[l, 2] += (x2 * x).sum(0); S[l, 3] += (x2 * x2).sum(0)
        h = torch.relu(p)
    n += B
a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
mt = np.stack([sh[l].double().cpu().numpy() + m1[l] for l in range(16)])
del W, S; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "pairrec": LS}
p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
pair = mod.ORAC["pair"]; mod.ORAC = None
out = {"k": k, "mse": float(np.mean((p[-1] - m_true[15]) ** 2)), "c3t": c3.astype(np.float32), "c4t": c4.astype(np.float32),
       "vart": var.astype(np.float32), "mt": mt.astype(np.float32)}
for li in LS:
    d = pair[li]; print(li, d["shapes"], flush=True)
    for nm in ("D3", "g4", "mu", "var", "D21", "Coff"):
        if d[nm] is not None: out["%s_%d" % (nm, li)] = cp.asnumpy(cp.asarray(d[nm])).astype(np.float32)
np.savez(OUT, **out); print("saved", OUT, "mse", out["mse"], flush=True)
