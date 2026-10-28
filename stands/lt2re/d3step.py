"""21.10.2026 white-box-f9: чи похибку D3 породжує похибка g4row вище по ланцюгу. Прогони: base; g4 (справжній c4 на шарах 1..15);
g4up (справжній c4 лише на шарах 1..l−1 відносно шару виміру — тут: на 1..9, вимір D3 на 10..14). Міра: rms(D3 − c3)/sd(c3) на живих.
ПЕРЕДБАЧЕННЯ: справжній c4 зменшить похибку D3 на 10…30% (κ4 входить у перенос κ3 через Віка (3,1)). Шанс > 30%: 25%.
usage: d3src.py <k>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k = int(sys.argv[1])
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).numpy()
z = np.load("/home/trim/wblogs/c34_n%d.npz" % k); c3, c4 = z["c3"], z["c4"]
Wg = [cp.asarray(x) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
c4g = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
def run(G4set=()):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if G4set: o.update(D3set=tuple(G4set), c3t=[cp.asarray(c3[l].astype(np.float32)) for l in range(16)])
    mod.ORAC = o
    est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
    mod.ORAC = None
    return pre
out = {}
for nm, gs in (("base", ()), ("d3all", range(1, 16))):
    pre = run(gs); row = []
    for l in range(6, 15):
        D3 = pre[l][2]; live = np.abs(D3) > 1e-12
        row.append(float(np.sqrt(np.mean((D3 - c3[l])[live] ** 2)) / c3[l][live].std()))
    out[nm] = row
    print("net %d %-6s rel err D3 l6..14: %s" % (k, nm, " ".join("%.4f" % x for x in row)), flush=True)
