"""gpu_dump_test (19.10.2026, white-box-89): ланцюг sb2 з LT_DUMP на GPU через gpufnp; звірка з CPU-дампом сусіда.
usage: gpu_dump_test.py <mini idx> [cpu_dump.npz]"""
import os, sys, time, types, importlib.util
os.environ["SRC_DIAG"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext   # real flopscope behind whestbench, before the shim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpufnp
cp = gpufnp.cp

EST = os.environ.get("EST", "/home/trim/wb2lean/v39fija/estimator_sb2.py")
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz")
k = int(sys.argv[1])
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_gpu", EST)
mod = importlib.util.module_from_spec(spec); sys.modules["est_gpu"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=os.path.dirname(EST)))


def weights_from_seed(s):
    wss, _, _ = np.random.SeedSequence(int(s)).spawn(3)
    rng = np.random.default_rng(wss)
    sc = float(np.sqrt(2.0 / 1024))
    return [(rng.standard_normal((1024, 1024)) * sc).astype(np.float32) for _ in range(16)]


def estimator_seed(s):
    _, _, ess = np.random.SeedSequence(int(s)).spawn(3)
    return int(ess.generate_state(1)[0])


s = int(t["mlp_seed"][k])
W = [cp.asarray(w) for w in weights_from_seed(s)]
mlp = types.SimpleNamespace(width=1024, depth=16, weights=W, seed=estimator_seed(s), name=str(t["mlp_name"][k]))
for rep in range(1):
    mod.LT_DUMP.clear()
    cp.cuda.Device().synchronize(); t0 = time.time()
    pred = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41))).astype(np.float32)
    cp.cuda.Device().synchronize(); wall = time.time() - t0
    print(f"rep {rep} wall {wall:.1f}s", flush=True)
import json
raw = float(np.mean((pred[-1].astype(np.float64) - t["all_layer_means"][k][-1]) ** 2))
print("raw", raw)
json.dump({"raw": raw, "calls": [[a, b, c] for a, b, c in mod.SRC_DIAG]}, open(sys.argv[2], "w"))
for a, b, c in mod.SRC_DIAG:
    tot = sum(c)
    print(a, b, " ".join(f"{x / tot:.3f}" for x in c))
