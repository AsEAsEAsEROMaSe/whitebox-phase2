"""21.10.2026 white-box-f9: реалізовна поправка D3 := D3 − σ³·ê, ê = лінійна функція ознак ланцюга (t, |t|, log var, γ3, γ4,
ранг t, ‖W‖², ΣW³/σ³, 1) з nu_feat; коефіцієнти з ІНШИХ мереж (крос) і власні (стеля лінійного класу). Шари 6..14, живі нейрони.
Скан сили λ (D3 − λσ³ê): своя мусить мати оптимум λ>0. MSE виходу проти істини корпусу.
ПЕРЕДБАЧЕННЯ: крос ×1.03…1.10 при λ≈1; власна ×1.05…1.15. Шанс крос ≥ ×1.05: 35%.
usage: d3fix.py <k> <betas.json: {"cross": [...], "own": [...]}> <out.jsonl>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, BJ, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]
BET = json.load(open(BJ))
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).numpy()
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
LS = tuple(range(6, 15))
def run(c3t=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if c3t is not None: o.update(D3set=LS, c3t=c3t)
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2)), pre
base, pre = run()
def feats(l):
    mu, var, D3, g4 = pre[l]; sd = np.sqrt(var); t = mu / sd; Wl = w[l].astype(np.float64)
    F = np.stack([t, np.abs(t), np.log(var), D3 / sd ** 3, g4 / var ** 2, np.argsort(np.argsort(-t)) / 1024.0,
                  (Wl ** 2).sum(0), (Wl ** 3).sum(0) / sd ** 3, np.ones(1024)], 1)
    return F, D3, sd
res = {"k": k, "base": base}
for nm in ("cross", "own"):
    b = np.array(BET[nm])
    for lam in (0.5, 1.0, 1.5):
        c3t = []
        for l in range(16):
            if l in LS:
                F, D3, sd = feats(l); live = np.abs(D3) > 1e-12
                v = np.where(live, D3 - lam * sd ** 3 * (F @ b), D3)
            else:
                v = np.zeros(1024)
            c3t.append(cp.asarray(v.astype(np.float32)))
        r = run(c3t)[0]; res["%s_%.1f" % (nm, lam)] = r
        print("net %d %s lam %.1f ×%.4f" % (k, nm, lam, base / r), flush=True)
open(OUT, "a").write(json.dumps(res) + "\n")
