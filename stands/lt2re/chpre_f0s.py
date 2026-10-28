"""chpre_f0s (21.10.2026 f0, ідея 5f): скаляр на g4row ЛИШЕ обрізаних рядків (idx >= S_OUT), без МК-оракула.
ПЕРЕДБАЧЕННЯ: оптимум 1.2…1.5, виграш не більший за g4_prun. Шанс помітного (≥×1.02) 30%.
chpre_f0 (21.10.2026 white-box-f0, на прохання 5f): оракул c4 окремо на обрізаних (індекси >= S_OUT) і необрізаних нейронах.\nПЕРЕДБАЧЕННЯ: g4_live ≈ g4_all (×1.12…1.14), g4_prun ≈ ×1.01, бо обрізаних 96 з 1024 і D3 на них не читається. Шанс 70%.\nchpre2: оракул — справжні c3/c4 передактивацій (МК 2^24) замість D3/g4row ланцюга на наборах шарів; MSE виходу. ПЕРЕДБАЧЕННЯ d3_all ×1.5…×3.\nchpre (20.10.2026 white-box-89). edge14: Еджворт зі СПРАВЖНІМИ γ3,γ4 по нейрону знімає похибку E[ReLU] ×50 (1.0e-3σ → 2e-5σ).
Тут: γ3,γ4 передактивацій, які несе сам ланцюг (D3, g4row з хука), проти справжніх МК 2^24 на шарах 12,14; і Еджворт з γ ланцюга
проти справжнього E[ReLU]. ПЕРЕДБАЧЕННЯ: corr(γ3 ланц, справж) 0.8–0.95, γ4 ланцюга грубий (corr <0.5) ⟹ похибка кроку в γ4. Шанс: 50%.
usage: chpre.py <k> <out.json>"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 14); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc_f0.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(g4ps=None):
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": (), "G4set": (), "g4ps": g4ps}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "base": run()}
for s_ in (1.2, 1.35, 1.5):
    res["s%.2f" % s_] = run(s_)
    print("net %d scale %.2f ×%.4f" % (k, s_, res["base"] / res["s%.2f" % s_]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
