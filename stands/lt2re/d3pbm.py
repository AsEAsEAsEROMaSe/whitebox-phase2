"""d3new (21.10.2026): D3 з власної повної K21 ланцюга і діагонального k3 (реалізовне, без оракула). c3parts2: corr diag+pair(K21) з c3 = 0.957 проти D3 0.84.\nПЕРЕДБАЧЕННЯ: new_all ×1.15…×1.3. Шанс >×1.1: 55%.\nd3scan (21.10.2026 white-box-89): c3parts — похибка D3 ланцюга ≈ −0.34·(парна частина), парна = 89% розкиду c3 ⟹ скаляр на D3 (і g4row)
може бути безкоштовним важелем. Скан множника a на D3 / g4row на всіх шарах, MSE виходу без оракула.
ПЕРЕДБАЧЕННЯ: оптимум d3a 1.15…1.35 з ×1.05…×1.15; шанс >×1.1: 35%.
usage: d3scan.py <k> <out.json>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, OUT = int(sys.argv[1]), sys.argv[2]
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc.py")
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
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(**kw):
    mod.ORAC = dict(K=-1, mode="", truth=T0, rec=[], **kw)
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
A = tuple(range(1, 16))
def kw(**x):
    d = dict(K21rec=A, k3p={}, Wn=Wg); d.update(x); return d
res = {"k": k, "base": run()}
res["recbase"] = run(**kw())
for pb in (-0.01, 0.01, 0.02, 0.03):
    res["pbm%.3f" % pb] = run(**kw(d3new=A, pb=pb, pbmask=True))
print("net %d " % k + " ".join("%s ×%.4f" % (kk, res["base"] / v) for kk, v in res.items() if kk not in ("k", "base")), flush=True)
json.dump(res, open(OUT, "w"))
