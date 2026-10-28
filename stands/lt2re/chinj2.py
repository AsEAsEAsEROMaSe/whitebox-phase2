"""chinj2 (c2, 22.10.2026): ПАРА каналів: передбачене c3 (ridge з 12 власних статистик, крос-фолд) РАЗОМ зі справжньою
позадіагональною C після ReLU на шарі 14 (гачок mode "c", K=14). ПЕРЕДБАЧЕННЯ: C14 сама ×1.10; pred c3 + C14 ≥ ×1.3 з шансом 30%
(якщо так, то c3 розв'язано регресією, і проблема лише в C). usage: chinj2.py <k> <pred.npz> <out.json>"""
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
k, PRED, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; KS = [14]; NMC = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orc", "/home/trim/wb2lean/lt2re/estimator_sb2_orc_k5.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    assert int(z["idx"][k // 16]) == k
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(990000 + 2 * k)
S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in KS + [15]}
S2 = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in KS}; sh = {}; done = 0
while done < NMC:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(16):
        h = torch.relu(h @ W[l])
        if l in S2 and l not in sh: sh[l] = h.mean(0)
        if l in S1:
            hc = h - sh[l] if l in sh else h
            S1[l] += hc.sum(0, dtype=torch.float64)
        if l in S2: S2[l] += (hc.T @ hc).double()
    done += B
Cs = {}
for l in KS:
    m = S1[l] / done; Cs[l] = cp.asarray((S2[l] / done - torch.outer(m, m)).float().cpu().numpy())
seal = float(np.mean(((S1[15] / done).cpu().numpy() - m_true[15]) ** 2)); print("seal m15", seal, flush=True); assert seal < 1e-6
del W, S1, S2; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]
pz = np.load(PRED); zc = np.load("/home/trim/wblogs/chpair_n%d.npz" % k)
c3true = [cp.asarray(zc["c3t"][l].astype(np.float32)) for l in range(16)]
c3pred = [cp.asarray(pz["c3pred_%d" % l]) if ("c3pred_%d" % l) in pz else c3true[l] for l in range(16)]
Tg = [[cp.asarray(m_true[l].astype(np.float32))] * 4 for l in range(16)]
def run(c3t, D3set=(), K=-1, mode="", **kw):
    mod.ORAC = {"K": K, "mode": mode, "truth": Tg, "rec": [], "C": Cs, "D3set": tuple(D3set), "G4set": (), "K5set": (), "c3t": c3t, **kw}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2))
res = {"k": k, "base": run(c3true)}
for nm, c3, ds, K, mode, kw in (("C14", c3true, (), 14, "c", {}), ("pred_c3+C14", c3pred, (12, 13, 14), 14, "c", {}),
                                ("pred_c3_14+C14", c3pred, (14,), 14, "c", {}), ("true_c3+C14", c3true, (12, 13, 14), 14, "c", {}),
                                ("pred_half+C14", c3pred, (12, 13, 14), 14, "c", {"lam3": 0.5}), ("m14+C14", c3true, (), 14, "mc", {})):
    res[nm] = run(c3, ds, K, mode, **kw); print("net %d %-16s ×%.3f" % (k, nm, res["base"] / res[nm]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
