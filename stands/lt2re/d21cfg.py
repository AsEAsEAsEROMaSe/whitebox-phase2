"""21.10.2026 white-box-f9: точність D21 ланцюга (передактивації) від конфігу (env), істина з кешу d21t_n<k>.npy (d21orac). Шари 6..14.
usage: d21cfg.py <k> <tag>"""
import os, sys, types, importlib.util, math, json
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, TAG = int(sys.argv[1]), sys.argv[2]; LS = tuple(range(6, 15))
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
KT = np.load("/home/trim/wblogs/d21t_n%d.npy" % k)
Wg = [cp.asarray(x.numpy()) for x in w]
mod.ORAC = {"K": -1, "mode": "", "truth": [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)], "rec": [], "D21rec": LS}
p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
D21 = {l: cp.asnumpy(cp.asarray(v)).astype(np.float64) for l, v in mod.ORAC.get("D21", {}).items()}
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
out = []
for l in LS:
    live = np.abs(pre[l][2]) > 1e-12; off = ~np.eye(1024, dtype=bool) & np.outer(live, live)
    a, b = D21[l][off], KT[l].astype(np.float64)[off]
    out.append((np.corrcoef(a, b)[0, 1], a @ b / (b @ b), np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2))))
print("%-8s raw %.4e  " % (TAG, np.mean((p[-1] - m_true[15]) ** 2)) + " ".join("L%d %.3f/%.3f/%.3f" % (l, *o) for l, o in zip(LS, out) if l in (6, 10, 14)), flush=True)
