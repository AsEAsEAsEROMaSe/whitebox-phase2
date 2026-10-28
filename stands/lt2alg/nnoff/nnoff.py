"""nnoff (21.10.2026, white-box-b9): виграш nn_asset ALG058 на mini 0..K (GPU-шим): raw фіналу з nn_asset і з нульовими ow/ob."""
import os, sys, types, importlib.util
os.environ["LT_OFF"] = "1"
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
EST = "/home/trim/wb2lean/s32lt/estimator_s32lt.py"
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
    for zero in (False, True):
        est._nn_ow, est._nn_ob = (ow * 0, ob * 0) if zero else (ow, ob)
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=[cp.asarray(x) for x in Wn], seed=int(ess.generate_state(1)[0]), name=str(k))
        p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41)))[-1].astype(np.float64)
        out.append(float(np.mean((p - tr) ** 2)))
    print("mini %d raw nn %.4e off %.4e nn-gain x%.4f" % (k, out[0], out[1], out[1] / out[0]), flush=True)
