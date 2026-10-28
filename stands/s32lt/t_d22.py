"""22.10.2026 a0: точний перенос (2,2)-зрізу κ4 у wk4m s32refire (LT вимкнений), бленд β = X_D22_B, mini через GPU-шим.
ПЕРЕДБАЧЕННЯ (до прогону): β=0.5 дає ×1.00…1.03 raw; шанс ≥ ×1.07 15%, бо (2,2) входить у середнє лише через поправки до C.
Контроль: X_D22_B=0 мусить дати базу біт у біт (лише лог кореляцій). usage: t_d22.py <a-b> <out.json>"""
import json, os, sys, types, importlib.util
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.insert(0, "/home/trim/wb2lean/lt2alg")
from gpu_dump_alg import weights_mini, estimator_seed, TRUTH
est_path, rng_, out = "/home/trim/wb2lean/s32lt/est_d22.py", sys.argv[1], sys.argv[2]
os.environ["LT_OFF"] = "1"
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("e_x", est_path); m = importlib.util.module_from_spec(spec); sys.modules["e_x"] = m; spec.loader.exec_module(m)
gpufnp.restore(saved)
e = m.Estimator(); e.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=os.path.dirname(os.path.abspath(est_path))))
t = np.load(TRUTH, allow_pickle=False); res = {}
lo, hi = (int(x) for x in rng_.split("-"))
for k in range(lo, hi + 1):
    s = int(t["mlp_seed"][k]); W = [cp.asarray(x) for x in weights_mini(s)]
    p = cp.asnumpy(cp.asarray(e.predict(types.SimpleNamespace(width=1024, depth=16, weights=W, seed=estimator_seed(s), name=str(k)), 2 ** 41))).astype(np.float64)
    res[k] = float(np.mean((p[-1] - t["all_layer_means"][k][-1].astype(np.float64)) ** 2))
    res["log%d" % k] = [list(x) for x in m.D22_LOG]; m.D22_LOG.clear()
    print(k, "%.6e" % res[k], flush=True); json.dump(res, open(out, "w"))
    cp.get_default_memory_pool().free_all_blocks()
