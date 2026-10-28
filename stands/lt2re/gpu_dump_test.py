"""gpu_dump_test (19.10.2026, white-box-89): ланцюг sb2 з LT_DUMP на GPU через gpufnp; звірка з CPU-дампом сусіда.
usage: gpu_dump_test.py <mini idx> [cpu_dump.npz]"""
import os, sys, time, types, importlib.util
os.environ["LT_DUMP"] = "1"
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext   # real flopscope behind whestbench, before the shim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpufnp
cp = gpufnp.cp

EST = "/home/trim/wb2lean/v39fija/estimator_sb2.py"
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
for rep in range(2):
    mod.LT_DUMP.clear()
    cp.cuda.Device().synchronize(); t0 = time.time()
    pred = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41))).astype(np.float32)
    cp.cuda.Device().synchronize(); wall = time.time() - t0
    print(f"rep {rep} wall {wall:.1f}s", flush=True)
X = np.stack([cp.asnumpy(x).astype(np.float32) for _, x, _ in mod.LT_DUMP])
Phi = np.stack([cp.asnumpy(p).astype(np.float32) for _, _, p in mod.LT_DUMP])
raw = float(np.mean((pred[-1].astype(np.float64) - t["all_layer_means"][k][-1]) ** 2))
print(f"mini {k} raw {raw:.6e} X {X.shape} finite {np.isfinite(X).all()} gpu_mem {cp.get_default_memory_pool().total_bytes() / 2**30:.2f}G")
if len(sys.argv) > 2:
    c = np.load(sys.argv[2])
    rp = np.max(np.abs(pred - c["pred"])) / np.max(np.abs(c["pred"]))
    rx = np.max(np.abs(X - c["X"]), axis=(0, 1)) / (np.max(np.abs(c["X"]), axis=(0, 1)) + 1e-30)
    craw = float(np.mean((c["pred"][-1].astype(np.float64) - t["all_layer_means"][k][-1]) ** 2))
    print(f"cpu raw {craw:.6e}  pred rel maxdiff {rp:.3e}  X rel maxdiff per col max {rx.max():.3e} median {np.median(rx):.3e}")
    print("worst cols", np.argsort(-rx)[:6], np.sort(rx)[::-1][:6])
