"""21.10.2026 white-box-f9: скаляр на D3 (хук d3a) ланцюга (усі шари з d21from), MSE виходу. Реалізовно за ціну одного множення.
ПЕРЕДБАЧЕННЯ: оптимум 1.01…1.03 (нахил D21 до істини 0.98…0.99 без прунінгу), ×1.02…1.08. Шанс ≥ ×1.05: 30%.
usage: d21scale.py <k> <scales comma>"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k = int(sys.argv[1]); SC = [float(x) for x in sys.argv[2].split(",")]; FROM = int(os.environ.get("D21FROM", "0"))
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
Wg = [cp.asarray(x.numpy()) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(sc=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if sc is not None: o.update(d3a=sc, d3from=FROM)
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
b = run()
print("net %d base %.4e " % (k, b) + " ".join("s%.3f ×%.4f" % (s, b / run(s)) for s in SC), flush=True)
