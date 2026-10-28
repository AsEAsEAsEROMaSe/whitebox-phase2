"""chdir2 (c2, 22.10.2026): похідні виходу за напрямками згорток рядків (C-зважені суми, канал C·κ3) і за крос-фіт передбаченнями
залишку. δD3 = ε·sd_y·X̃_j на шарах 12..14. usage: chdir2.py <k> <contr_te.npz> <out.npz>"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, CF, OUT = int(sys.argv[1]), sys.argv[2], sys.argv[3]; LS = (12, 13, 14); EPS = 0.3
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
Wg = [cp.asarray(x.numpy()) for x in w]
zc = np.load("/home/trim/wblogs/chpair_n%d.npz" % k); zf = np.load(CF); zd = np.load("/home/trim/wblogs/pair/dir_n%d.npz" % k)
D3r = {li: zc["D3_%d" % li].astype(np.float64) for li in LS}; names = list(zf["names"])
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(c3t):
    mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": LS, "G4set": (), "K5set": (), "c3t": [cp.asarray(c3t.get(l, np.zeros(1024)).astype(np.float32)) for l in range(16)]}
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return p[-1]
out0 = zd["out0"]; e0 = out0 - m_true[15]; ne = np.linalg.norm(e0); At = zd["A3_true"]
res = {"k": k, "names": np.array(names)}; A = []
for j, nm in enumerate(names):
    Aj = (run({li: D3r[li] + EPS * float(zf["sdy_%d" % li]) * zf["X_%d" % li][:, j] for li in LS}) - out0) / EPS; A.append(Aj)
    lam = -np.dot(e0, Aj) / np.dot(Aj, Aj); gain = np.mean(e0 ** 2) / np.mean((e0 + lam * Aj) ** 2)
    print("net %d dir %-10s |A| %.3e (%.2f|e0|) cos(e0,A) %+.3f cos(A_true,A) %+.3f λ* %+.3f gain ×%.3f" % (k, nm, np.linalg.norm(Aj), np.linalg.norm(Aj) / ne, np.dot(e0, Aj) / (ne * np.linalg.norm(Aj)), np.dot(At, Aj) / (np.linalg.norm(At) * np.linalg.norm(Aj)), lam, gain), flush=True)
    res["A"] = np.array(A); np.savez(OUT, **res)
print("saved", flush=True)
