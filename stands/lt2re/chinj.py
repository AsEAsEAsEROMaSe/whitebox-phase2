"""chinj (c2, 22.10.2026): вкидання ПЕРЕДБАЧЕНОГО (крос-мережно, 12 власних статистик) c3 у ланцюг через D3set на шарах 12..14.
Порівняння: base · pred (12,13,14) · pred лише 14 · truth (12,13,14) · pred з lam3 0.5. usage: chinj.py <k> <pred.npz> <out.json>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, PRED, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
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
pz = np.load(PRED); zc = np.load("/home/trim/wblogs/chpair_n%d.npz" % k)
c3true = [cp.asarray(zc["c3t"][l].astype(np.float32)) for l in range(16)]
c3pred = [cp.asarray(pz["c3pred_%d" % l]) if ("c3pred_%d" % l) in pz else c3true[l] for l in range(16)]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(c3t, D3set=(), **kw):
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": tuple(D3set), "G4set": (), "K5set": (), "c3t": c3t, **kw}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "base": run(c3true)}
for nm, c3, ds, kw in (("pred_12-14", c3pred, (12, 13, 14), {}), ("pred_14", c3pred, (14,), {}), ("pred_half", c3pred, (12, 13, 14), {"lam3": 0.5}),
                       ("truth_12-14", c3true, (12, 13, 14), {}), ("truth_14", c3true, (14,), {})):
    res[nm] = run(c3, ds, **kw); print("net %d %-12s ×%.3f" % (k, nm, res["base"] / res[nm]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
