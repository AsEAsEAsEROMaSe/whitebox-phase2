"""21.10.2026 f0 t_knob: скан безкоштовних ручок s32refire (LAM_SCALE, LAM_TILT, BETA) через env, TAG у назві виходу.
ПЕРЕДБАЧЕННЯ: автор підбирав їх на 5…8 мережах; шанс знайти ×1.02+ на корпусі 25%.
було: raw s32refire (оригінал), s32lt з LT_OFF=1 (мусить збігтися біт у біт) і s32lt з LT2 sb2 (без перенавчання) на корпусі через GPU-шим.
ПЕРЕДБАЧЕННЯ: LT2 sb2 на чужому ланцюгу без перенавчання гірший за базу (як 19.10 ×2.5 у зворотний бік), шанс кращого 15%.
usage: t_s32.py <k,k,..>"""
import json, os, sys, types, importlib.util, math, io, contextlib
# V26_STRASSEN=0 ламає s32refire (raw ×13), не ставити
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False); torch.set_num_threads(1)
D = "/home/trim/wb2lean/s32lt"
def load(name, path, ltoff):
    os.environ["LT_OFF"] = "1" if ltoff else "0"
    saved = gpufnp.install()
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    gpufnp.restore(saved)
    e = m.Estimator(); e.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=D))
    return e
WH = os.environ["WHICH"]   # одна модель на процес: три разом не влазять у 8 ГБ (OOM пулів Штрассена)
E = {WH: {"alg": lambda: load("e_alg", D + "/alg.py", True), "s32lt_off": lambda: load("e_off", D + "/estimator_s32lt.py", True),
          "s32lt_lt2sb2": lambda: load("e_on", D + "/estimator_s32lt.py", False), "zero": lambda: load("e_z", D + "/est_zero.py", False)}[WH]()}
TAG = os.environ["TAG"]
out = "/home/trim/wblogs/t_knob_%s.json" % TAG; res = {}
for k in [int(x) for x in sys.argv[1].split(",")]:
    g = torch.Generator(device="cpu").manual_seed(770000 + k)
    w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
    with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
        m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
    Wg = [cp.asarray(x.numpy()) for x in w]; P = {}
    for nm, e in E.items():
        with contextlib.redirect_stdout(io.StringIO()):
            P[nm] = cp.asnumpy(cp.asarray(e.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
        res["%d_%s" % (k, nm)] = float(np.mean((P[nm][-1] - m_true[15]) ** 2))
        cp.get_default_memory_pool().free_all_blocks()
    print("net %d %s raw %.4e" % (k, TAG, res["%d_%s" % (k, WH)]), flush=True)
    
    json.dump(res, open(out, "w"), indent=1)
