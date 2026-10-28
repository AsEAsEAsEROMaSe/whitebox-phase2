"""21.10.2026 white-box-f9: чим корелює похибка D3 ланцюга (c3 передактивації) на живих нейронах.
e = D3 − c3 (c3 з МК c34_n<k>.npz, 2^24). Ознаки по нейрону шару l: t = mu/σ, |t|, log var, γ3 ланцюга D3/σ³, γ4 g4row/σ⁴,
ранг t у шарі, ‖W_:j‖², Σ_a W³_aj, позиція відносно межі прунінгу. Кореляції по шарах 6..14 і МНК на половині нейронів (крос).
ПЕРЕДБАЧЕННЯ: e/σ³ корелює з t (вузькі ворота, |t| малі) і з γ3 ланцюга (стиск), крос R² 0.1…0.3. Шанс R² > 0.3: 25%.
usage: nu_feat.py <k> <out.json>"""
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
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).numpy()
c3 = np.load("/home/trim/wblogs/c34_n%d.npz" % k)["c3"]
Wg = [cp.asarray(x) for x in w]
mod.ORAC = {"K": -1, "mode": "", "truth": [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)], "rec": []}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
mod.ORAC = None
res = {"k": k}; Xall, Yall, Lall = [], [], []
for l in range(6, 15):
    mu, var, D3, g4 = pre[l]
    live = np.abs(D3) > 1e-12
    sd = np.sqrt(var); t = mu / sd
    e = (D3 - c3[l]) / sd ** 3
    Wl = w[l].astype(np.float64)
    F = {"t": t, "abs_t": np.abs(t), "logvar": np.log(var), "g3": D3 / sd ** 3, "g4": g4 / var ** 2,
         "rank_t": np.argsort(np.argsort(-t)) / 1024.0, "w2": (Wl ** 2).sum(0), "w3": (Wl ** 3).sum(0) / sd ** 3}
    out = {"live": int(live.sum()), "rms_e": float(np.sqrt(np.mean(e[live] ** 2))), "corr": {}}
    for nm, v in F.items():
        out["corr"][nm] = float(np.corrcoef(v[live], e[live])[0, 1])
    res[str(l)] = out
    X = np.stack([F[nm][live] for nm in F] + [np.ones(live.sum())], 1)
    Xall.append(X); Yall.append(e[live]); Lall.append(np.full(live.sum(), l))
    print("L%d live %d rms e/σ³ %.3e " % (l, live.sum(), out["rms_e"]) + " ".join("%s %+.3f" % (a, b) for a, b in out["corr"].items()), flush=True)
X = np.concatenate(Xall); Y = np.concatenate(Yall)
rng = np.random.default_rng(0); h = rng.random(len(Y)) < 0.5
b, *_ = np.linalg.lstsq(X[h], Y[h], rcond=None)
r2 = 1 - np.mean((Y[~h] - X[~h] @ b) ** 2) / np.mean((Y[~h] - Y[~h].mean()) ** 2)
res["cross_R2_linear"] = float(r2); res["names"] = list(F) + ["1"]; res["beta"] = b.tolist()
print("cross R2 linear (all layers 6..14) %.3f" % r2, flush=True)
json.dump(res, open(OUT, "w"), indent=1)
