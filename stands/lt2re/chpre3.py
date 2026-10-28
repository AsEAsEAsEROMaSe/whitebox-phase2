"""chpre2: оракул — справжні c3/c4 передактивацій (МК 2^24) замість D3/g4row ланцюга на наборах шарів; MSE виходу. ПЕРЕДБАЧЕННЯ d3_all ×1.5…×3.\nchpre (20.10.2026 white-box-89). edge14: Еджворт зі СПРАВЖНІМИ γ3,γ4 по нейрону знімає похибку E[ReLU] ×50 (1.0e-3σ → 2e-5σ).
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
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(22000 + k)
S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}; n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = (p - sh[l]).double()
        S[l, 0] += x.sum(0); S[l, 1] += (x * x).sum(0); S[l, 2] += (x ** 3).sum(0); S[l, 3] += (x ** 4).sum(0)
        h = torch.relu(p)
    n += B
a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
del W, S; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
c3t = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]; c4t = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(D3set=(), G4set=(), lam=1.0):
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": tuple(D3set), "G4set": tuple(G4set), "c3t": c3t, "c4t": c4t, "lam3": lam}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "base": run()}
A = range(1, 16)
for lam in (0.25, 0.5, 0.75):
    res["lam%.2f" % lam] = run(A, A, lam)
    print("net %d both lam %.2f ×%.3f" % (k, lam, res["base"] / res["lam%.2f" % lam]), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
