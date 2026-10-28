"""21.10.2026 white-box-2e (= f9): скан AW_Y_EPS (обрізання молодих атомів = стовпців народження) на sb2 через GPU-шим.
Шим не рахує флопи ⟹ ціна оцінюється через Σk_list (ширини молодих атомів; хаб молодих 43.5% + перенос молодих ніг 26% ціни
лінійні по k). raw MSE проти m_true корпусу. Мережі корпусу k (сід 770000+k як accC2).
ПЕРЕДБАЧЕННЯ: Y_EPS 0.01 дає Σk −8…−15% (C/B −4…−7%) при raw +1…3%; бал ×1.02…1.05 (шанс 40%).
usage: ycan.py <k,k,..> <eps,eps,..>"""
import json, os, sys, types, importlib.util, math, io, contextlib, re
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0"); os.environ["AW_Y_DEBUG"] = "1"
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
ks = [int(x) for x in sys.argv[1].split(",")]; eps = [float(x) for x in sys.argv[2].split(",")]
out = "/home/trim/wblogs/ycan.json"
res = json.load(open(out)) if os.path.exists(out) else {}
for k in ks:
    g = torch.Generator(device="cpu").manual_seed(770000 + k)
    w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
    with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
        m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
    Wg = [cp.asarray(x.numpy()) for x in w]
    for e in eps:
        key = "%d_%g" % (k, e)
        if key in res: continue
        est.Y_EPS = e
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
        txt = buf.getvalue(); mm = re.findall(r"YCAP k_list \[([^\]]*)\]", txt)
        kl = [int(x) for x in mm[-1].split(",")] if mm else []
        raw = float(np.mean((p[-1] - m_true[15]) ** 2))
        res[key] = {"raw": raw, "ksum": sum(kl), "klist": kl}
        cp.get_default_memory_pool().free_all_blocks()
        print("net %d eps %g raw %.4e ksum %d" % (k, e, raw, sum(kl)), flush=True)
        json.dump(res, open(out + ".tmp", "w")); os.replace(out + ".tmp", out)
