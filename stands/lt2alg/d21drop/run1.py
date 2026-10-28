"""d21drop run (22.10.2026 b9): raw фіналу оцінювача з викиданням молодих джерел у D21 (env D21_DROP_*), mini a..b."""
import os, sys, types, importlib.util
os.environ.pop("LT_OFF", None)
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
EST = "/home/trim/wb2lean/lt2alg/d21drop/est_drop.py"
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
SCALE = (2.0 / 1024.0) ** 0.5
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_gpu", EST)
mod = importlib.util.module_from_spec(spec); sys.modules["est_gpu"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=os.path.dirname(EST)))
ow, ob = est._nn_ow, est._nn_ob
for k in range(int(sys.argv[1]), int(sys.argv[2])):
    s = int(t["mlp_seed"][k])
    wss, _, ess = np.random.SeedSequence(s).spawn(3)
    rng = np.random.default_rng(wss)
    Wn = [(rng.standard_normal((1024, 1024)) * SCALE).astype(np.float32) for _ in range(16)]
    tr = t["all_layer_means"][k][-1].astype(np.float64)
    out = []
    CS = (1.0,)
    for c in CS:
        est._nn_ow, est._nn_ob = (ow * c, ob * c)
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=[cp.asarray(x) for x in Wn], seed=int(ess.generate_state(1)[0]), name=str(k))
        p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41)))[-1].astype(np.float64)
        out.append(float(np.mean((p - tr) ** 2)))
    print("mini %d " % k + " ".join("c%.1f %.4e x%.4f" % (c, o, out[0] / o) for c, o in zip(CS, out)), flush=True)
st = mod.D21_DROP_STAT
print("DROPSTAT young-slices kept/total", sum(b for a, b in st), "/", sum(a for a, b in st), flush=True)
