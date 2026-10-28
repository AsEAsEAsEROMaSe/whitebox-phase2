"""21.10.2026 white-box-f9: e2e-лінеаризація поправок усередині ланцюга. Сім'ї: D3 += Σ_f β_f σ³ F_f і g4row += Σ_f γ_f σ⁴ F_f
на живих нейронах шарів 6..14, F = (t, |t|, log var, γ3, γ4, ранг t, ‖W‖², ΣW³/σ³, 1). Для кожної ознаки одностороння
кінцева різниця виходу (eps) ⟹ якобіан J (1024 × 18); далі МНК по мережах мінімізує ‖p0 − truth + J θ‖² (крос по мережах).
Пишемо p0, J, truth у /home/trim/wblogs/jac/jac_n<k>.npz. usage: jac.py <a-b> [step off]"""
import json, os, sys, types, importlib.util, math, time
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1)
lo, hi = (int(x) for x in sys.argv[1].split("-")); step = int(sys.argv[2]) if len(sys.argv) > 2 else 1; off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
EPS3 = float(os.environ.get("EPS3", "1e-3")); EPS4 = float(os.environ.get("EPS4", "1e-2"))
D = "/home/trim/wblogs/jac"; os.makedirs(D, exist_ok=True)
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
LS = tuple(range(6, 15))
for k in range(lo, hi + 1):
    if (k - lo) % step != off: continue
    out = "%s/jac_n%d.npz" % (D, k)
    if os.path.exists(out): continue
    t0 = time.time()
    g = torch.Generator(device="cpu").manual_seed(770000 + k)
    w = (torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)).numpy()
    with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
        assert int(z["idx"][k // 16]) == k
        truth = np.asarray(z["moments"][k // 16][0, 15], dtype=np.float64)
    Wg = [cp.asarray(x) for x in w]
    def run(c3t=None, c4t=None):
        o = {"K": -1, "mode": "", "truth": T0, "rec": []}
        if c3t is not None: o.update(D3set=LS, c3t=c3t)
        if c4t is not None: o.update(G4set=LS, c4t=c4t)
        mod.ORAC = o
        p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
        pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
        mod.ORAC = None
        return p[-1], pre
    p0, pre = run()
    FF = {}
    for l in LS:
        mu, var, D3, g4 = pre[l]; sd = np.sqrt(var); t = mu / sd; Wl = w[l].astype(np.float64)
        F = np.stack([t, np.abs(t), np.log(var), D3 / sd ** 3, g4 / var ** 2, np.argsort(np.argsort(-t)) / 1024.0,
                      (Wl ** 2).sum(0), (Wl ** 3).sum(0) / sd ** 3, np.ones(1024)], 1)
        FF[l] = (F, D3, g4, sd, np.abs(D3) > 1e-12)
    J = np.zeros((1024, 18))
    for f in range(9):
        c3t = [cp.asarray((np.where(FF[l][4], FF[l][1] + EPS3 * FF[l][3] ** 3 * FF[l][0][:, f], FF[l][1]) if l in LS else np.zeros(1024)).astype(np.float32)) for l in range(16)]
        J[:, f] = (run(c3t=c3t)[0] - p0) / EPS3
        c4t = [cp.asarray((np.where(FF[l][4], FF[l][2] + EPS4 * FF[l][3] ** 4 * FF[l][0][:, f], FF[l][2]) if l in LS else np.zeros(1024)).astype(np.float32)) for l in range(16)]
        J[:, 9 + f] = (run(c4t=c4t)[0] - p0) / EPS4
    np.savez(out + ".tmp.npz", p0=p0, J=J, truth=truth)
    os.replace(out + ".tmp.npz", out)
    print("net %d raw %.4e %.0fs" % (k, np.mean((p0 - truth) ** 2), time.time() - t0), flush=True)
