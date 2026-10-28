"""c4fit (21.10.2026 white-box-89): дамп ознак ланцюга для прогнозу c4 передактивацій + справжній c3/c4 (МК 2^24) усіх шарів.
Ознаки шару l (природний порядок): g4row, var, D3, mu, aabb з K22[l-1], d4 = Σ W⁴ K4v[l-1], s4 = (Σ W² K2v[l-1])².
usage: c4fit.py <k>   → /home/trim/wblogs/c4fit_n<k>.npz"""
import json, os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False)
k = int(sys.argv[1]); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
A = tuple(range(16))
mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "K22rec": A}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
tn = lambda v: None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64)
pre = {li: [tn(v) for v in vs] for li, *vs in mod.ORAC["pre"]}
rec = {li: [tn(v) for v in vs] for li, vs in mod.ORAC["rec"]}
K22 = {li: tn(v) for li, v in mod.ORAC["K22"].items()}
mod.ORAC = None; del Wg
cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
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
wn = w.numpy().astype(np.float64); out = {"c3": c3, "c4": c4, "var_true": var}
F = np.zeros((16, 7, 1024))
for l in range(1, 16):
    mu_, var_, D3_, g4_ = pre[l]
    W2 = wn[l] ** 2; Kb = K22[l - 1] - np.diag(np.diag(K22[l - 1]))
    F[l, 0] = g4_; F[l, 1] = var_; F[l, 2] = D3_ if D3_ is not None else 0; F[l, 3] = mu_
    F[l, 4] = 3 * (W2 * (Kb @ W2)).sum(0); F[l, 5] = (W2 ** 2 * rec[l - 1][3][:, None]).sum(0); F[l, 6] = ((W2 * rec[l - 1][1][:, None]).sum(0)) ** 2
out["F"] = F
np.savez("/home/trim/wblogs/c4fit_n%d.npz" % k, **out)
print("net %d done" % k, flush=True)
