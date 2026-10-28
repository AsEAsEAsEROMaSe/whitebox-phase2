"""21.10.2026 f0 (гілка ціни від bf): скан V30_S_IN / V30_S_OUT на шарах 8..14 на sb2 через GPU-шим (як ycan.py f9).
Шим не рахує флопи ⟹ тут лише raw MSE проти m_true корпусу; ціну переможця міряти одним прогоном run_direct (CPU, з дозволу юзера).
Розклади правляться на місці в mod.S_IN / mod.S_OUT (LR0 = 3 не змінюється, бо шари 8..14 ≥ 3).
ПЕРЕДБАЧЕННЯ: S_IN −32 дає raw +0.5…2%, −64 +2…6%; S_OUT чутливіший (тримає ноги K3), −32 дає +2…5%. Шанс, що якийсь варіант дасть бал ×1.03+ при C/B −5%: 35%.
usage: scan_f0.py <k,k,..>"""
import json, os, sys, types, importlib.util, math, io, contextlib
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False); torch.set_num_threads(1)
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_sb2", "/home/trim/wb2lean/v39fija/estimator_sb2.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_sb2"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
SI0, SO0 = list(mod.S_IN), list(mod.S_OUT)
L = range(8, 15)
def sh(base, d): return [b - d if i in L else b for i, b in enumerate(base)]
CF = {"base": (SI0, SO0), "in32": (sh(SI0, 32), SO0), "in64": (sh(SI0, 64), SO0), "out32": (SI0, sh(SO0, 32)),
      "out64": (SI0, sh(SO0, 64)), "both32": (sh(SI0, 32), sh(SO0, 32))}
ks = [int(x) for x in sys.argv[1].split(",")]
out = "/home/trim/wblogs/scan_f0.json"
res = json.load(open(out)) if os.path.exists(out) else {}
for k in ks:
    g = torch.Generator(device="cpu").manual_seed(770000 + k)
    w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
    with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
        m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
    Wg = [cp.asarray(x.numpy()) for x in w]
    for nm, (si, so) in CF.items():
        key = "%d_%s" % (k, nm)
        if key in res: continue
        mod.S_IN[:] = si; mod.S_OUT[:] = so
        with contextlib.redirect_stdout(io.StringIO()):
            p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
        raw = float(np.mean((p[-1] - m_true[15]) ** 2))
        res[key] = {"raw": raw, "rows_in": sum(si), "rows_out": sum(so)}
        cp.get_default_memory_pool().free_all_blocks()
        print("net %d %s raw %.4e  ×base %.4f" % (k, nm, raw, raw / res.get("%d_base" % k, {"raw": raw})["raw"]), flush=True)
        json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
