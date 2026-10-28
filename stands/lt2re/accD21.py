"""accD21 (21.10.2026 f0, завдання 5f): чи лишок від C+c3c4 до mC14 сидить у D21. Як accC (той самий сід МК 55000+k, estimator_sb2_orc → trip
з хуком D21set), плюс справжній D21 = κ(a,a,b) передактивацій ш1..14 з d21t_all_n{k}.npy (f9).
Режими: C+c3c4 (пломба accC: мережа 0 ×4.49, мережа 1 ×6.43) · C+c3c4+D21 · c3c4+D21 · D21 · C+D21.
ПЕРЕДБАЧЕННЯ: C+c3c4+D21 ×5…8 (лишок до ×9 частково), шанс ≥ 70% лог-відстані до mC14: 30%.
usage: accD21.py <k> <out.json>"""
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
k, OUT = int(sys.argv[1]), sys.argv[2]; N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
cp.get_default_memory_pool().free_all_blocks()
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(55000 + k)
mt = torch.tensor(m_true, dtype=torch.float32, device=dev)
S = torch.zeros(16, 4, 1024, dtype=torch.float64, device=dev); sh = {}
G = torch.zeros(15, 1024, 1024, dtype=torch.float64, device=dev); n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        p = h @ W[l]
        if l not in sh: sh[l] = p.mean(0)
        x = (p - sh[l]).double()
        S[l, 0] += x.sum(0); S[l, 1] += (x * x).sum(0); S[l, 2] += (x ** 3).sum(0); S[l, 3] += (x ** 4).sum(0)
        h = torch.relu(p)
        if 1 <= l <= 14:
            xc = h - mt[l]; G[l] += (xc.T @ xc).double()
    n += B
a = (S / n).cpu().numpy(); m1, m2, m3, m4 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
var = m2 - m1 ** 2; c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
Cs = {l: cp.asarray((G[l] / n).float().cpu().numpy()) for l in range(1, 15)}
del W, S, G, h, p, x, xc, mt, sh; import gc; gc.collect(); torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
c3t = [cp.asarray(c3[l].astype(np.float32)) for l in range(16)]; c4t = [cp.asarray(c4[l].astype(np.float32)) for l in range(16)]
Tg = [[cp.asarray(m_true[l].astype(np.float32))] * 4 for l in range(16)]
def run(**x):
    d = dict(K=-1, mode="", truth=Tg, rec=[], C=Cs, c3t=c3t, c4t=c4t); d.update(x); mod.ORAC = d
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
A15 = tuple(range(1, 16)); A14 = tuple(range(1, 15))
KT = np.load("/home/trim/wblogs/d21t_all_n%d.npy" % k); KTg = {l: cp.asarray(KT[l]) for l in A14}; del KT
DD = dict(D21set=A14, d21t=KTg, lam21=1.0)
res = {"k": k, "base": run()}
for nm, x in (("C+c3c4", dict(Kset=A14, mode="c", D3set=A15, G4set=A15)), ("C+c3c4+D21", dict(Kset=A14, mode="c", D3set=A15, G4set=A15, **DD)),
              ("c3c4+D21", dict(D3set=A15, G4set=A15, **DD)), ("D21", dict(DD)), ("C+D21", dict(Kset=A14, mode="c", **DD))):
    res[nm] = run(**x)
    print("net %d %s ×%.3f" % (k, nm, res["base"] / res[nm]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
