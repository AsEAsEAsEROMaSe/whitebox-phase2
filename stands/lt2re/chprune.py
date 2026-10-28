"""21.10.2026 white-box-f9: чи оракул c3/c4 (chpre2 ×1.9) лікує лише нульовий D3/g4row обрізаних нейронів (прунінг V30).
Режими (MSE виходу проти істини корпусу): base; d3 (справжній c3 на всіх 1024, шари 1..15); d3pr (справжній c3 лише там, де
D3 ланцюга = 0, решта D3 ланцюга); both / bothpr (те саме для c3 і c4). Запуск двічі: env V30_PRUNE=1 і 0.
МК c3/c4 2^24 кешується в /home/trim/wblogs/c34_n<k>.npz (той самий сід, що chpre2).
ПЕРЕДБАЧЕННЯ: d3pr ≈ d3 (уся вигода в обрізаних); при PRUNE=0 оракул d3 ×1.0…1.1. Шанс: 60%.
usage: chprune.py <k> <out.jsonl>"""
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
k, OUT = int(sys.argv[1]), sys.argv[2]; N = 2 ** 24; B = 8192
PR = os.environ.get("V30_PRUNE", "1")
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
cache = "/home/trim/wblogs/c34_n%d.npz" % k
if os.path.exists(cache):
    z = np.load(cache); c3, c4 = z["c3"], z["c4"]
else:
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
    np.savez(cache, c3=c3, c4=c4)
    del W, S; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
A = tuple(range(1, 16))
def run(c3t=None, c4t=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if c3t is not None: o.update(D3set=A, c3t=c3t)
    if c4t is not None: o.update(G4set=A, c4t=c4t)
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2)), pre
base, pre = run()
g = lambda arr: [cp.asarray(arr[l].astype(np.float32)) for l in range(16)]
D3c = np.stack([pre[l][2] if l in pre and pre[l][2] is not None else np.zeros(1024) for l in range(16)])
G4c = np.stack([pre[l][3] if l in pre and pre[l][3] is not None else np.zeros(1024) for l in range(16)])
z3 = np.abs(D3c) < 1e-12; z4 = np.abs(G4c) < 1e-12
c3pr = np.where(z3, c3, D3c); c4pr = np.where(z4, c4, G4c)
res = {"k": k, "prune": PR, "base": base, "nzero_d3": z3.sum(1).tolist(), "nzero_g4": z4.sum(1).tolist()}
for nm, a3, a4 in (("d3", c3, None), ("d3pr", c3pr, None), ("both", c3, c4), ("bothpr", c3pr, c4pr)):
    res[nm] = run(g(a3), None if a4 is None else g(a4))[0]
print("net %d prune %s base %.4e " % (k, PR, base) + " ".join("%s ×%.3f" % (nm, base / res[nm]) for nm in ("d3", "d3pr", "both", "bothpr")), flush=True)
print("   zeros d3 per layer", res["nzero_d3"], flush=True)
open(OUT, "a").write(json.dumps(res) + "\n")
