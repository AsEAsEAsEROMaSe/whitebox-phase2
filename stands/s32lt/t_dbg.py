import os, sys, types, importlib.util, math, io, contextlib
os.environ["LT_OFF"] = "1"
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
D = "/home/trim/wb2lean/s32lt"
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("e_alg", D + "/alg.py"); m = importlib.util.module_from_spec(spec); sys.modules["e_alg"] = m; spec.loader.exec_module(m)
gpufnp.restore(saved)
e = m.Estimator(); e.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=D))
k = 0
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    mt = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x.numpy()) for x in w]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    P = cp.asnumpy(cp.asarray(e.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
print(buf.getvalue()[-2000:])
for l in range(16): print(l, "%.3e" % np.mean((P[l] - mt[l]) ** 2), "%.3e" % np.mean(mt[l] ** 2))
print("V26_STRASSEN", os.environ.get("V26_STRASSEN"), "riders cfg", m.CORR_BETA.__len__())
