"""run_direct (19.10.2026, white-box-89): легкий прогін естиматора без whest (RSS ~2 ГБ проти 5 у whest run).
usage: run_direct.py <estimator.py> <idx,idx,..> <out.json>   (env естиматора задавати ззовні)
Кожна мережа: raw MSE останнього шару проти mini_truth Susupro, C/B. Захист: вихід, якщо MemAvailable < 3 ГБ."""
import sys, os, json, time, importlib.util, threading
sys.path.insert(0, "/home/trim/ext/fc/repo/tools")
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, SetupContext
from mlpdata import weights_from_seed, estimator_seed

B = 2 ** 41


def guard():
    while True:
        a = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 2**20
        if a < 3.0:
            print(f"MEMGUARD avail {a:.1f}G", flush=True)
            os._exit(3)
        time.sleep(2)


threading.Thread(target=guard, daemon=True).start()
est_path, idxs, out = os.path.abspath(sys.argv[1]), [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
spec = importlib.util.spec_from_file_location("est_mod", est_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["est_mod"] = mod
spec.loader.exec_module(mod)
est = mod.Estimator()
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz")
est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1", submission_dir=os.path.dirname(est_path)))
res = []
for i in idxs:
    W = [fnp.array(w, dtype=fnp.float32) for w in weights_from_seed(t["mlp_seed"][i])]
    mlp = MLP(width=1024, depth=16, weights=W, seed=estimator_seed(t["mlp_seed"][i]), name=str(t["mlp_name"][i]))
    t0 = time.time()
    with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
        pred = np.asarray(fnp.asarray(est.predict(mlp, B)), dtype=np.float64)
    raw = float(np.mean((pred[-1] - t["all_layer_means"][i][-1].astype(np.float64)) ** 2))
    res.append({"idx": i, "raw": raw, "CB": ctx.flops_used / B, "wall": time.time() - t0})
    print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(out, "w"))
