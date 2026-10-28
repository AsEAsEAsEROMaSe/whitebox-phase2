"""21.10.2026 white-box-f9 (питання юзера): наш ланцюг проти оракула — наскільки різняться D3/c3, g4row/c4, D21/κ(a,a,b) і чи є
система в різниці. Дамп ланцюга (PRUNE=1, як у сабміті), живі нейрони, шари 6..14. Аналіз:
 (1) рівень: rel rms, нахил, corr; (2) спільний зсув: mean(e)/rms(e); (3) функція від t: частка дисперсії e/σ^k, яку знімає
 біннінг по t (20 бінів, крос між мережами); (4) зв'язок між каналами: corr(e3, e4), corr(e3, рядкові суми похибки D21 з вагами W);
 (5) спадковість: corr(e_l, перенос e_{l-1}) через W³ (діагональний перенос κ3).
usage: diffsys.py <k> -> /home/trim/wblogs/diffsys_n<k>.npz"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k = int(sys.argv[1]); LS = tuple(range(6, 15))
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
Wg = [cp.asarray(x.numpy()) for x in w]
mod.ORAC = {"K": -1, "mode": "", "truth": [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)], "rec": [], "D21rec": LS}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
D21 = {l: cp.asnumpy(cp.asarray(v)).astype(np.float64) for l, v in mod.ORAC["D21"].items()}
mod.ORAC = None
z = np.load("/home/trim/wblogs/c34_n%d.npz" % k); c3, c4 = z["c3"], z["c4"]
KT = np.load("/home/trim/wblogs/d21t_all_n%d.npy" % k, mmap_mode="r") if os.path.exists("/home/trim/wblogs/d21t_all_n%d.npy" % k) else None
out = {}
for l in LS:
    mu, var, D3, g4 = pre[l]; sd = np.sqrt(var)
    out["mu%d" % l], out["sd%d" % l], out["D3_%d" % l], out["g4_%d" % l] = mu, sd, D3, g4
    out["c3_%d" % l], out["c4_%d" % l] = c3[l], c4[l]
    if KT is not None:
        E21 = D21[l] - np.asarray(KT[l], dtype=np.float64); np.fill_diagonal(E21, 0.0)
        live = np.abs(D3) > 1e-12; E21[~live, :] = 0; E21[:, ~live] = 0
        out["e21row%d" % l] = E21.sum(1); out["e21col%d" % l] = E21.sum(0)
        out["e21rel%d" % l] = np.array([np.sqrt(np.mean(E21[np.outer(live, live)] ** 2) / np.mean(np.asarray(KT[l], dtype=np.float64)[np.outer(live, live)] ** 2))])
np.savez("/home/trim/wblogs/diffsys_n%d.npz" % k, **out)
print("saved", k, flush=True)
