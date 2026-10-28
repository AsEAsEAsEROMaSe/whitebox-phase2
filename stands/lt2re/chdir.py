"""chdir (c2, 22.10.2026, ідея юзера «переінакшити застосування»): похідні виходу за напрямками поправки стану.
Напрямки: 12 стандартизованих власних ознак нейрона (як у fit_pair) як δD3 на шарах 12..14 (спільний β), і ті самі 12 як δg4.
Кожен напрямок: один прогін ланцюга з D3set/G4set (c3t = D3_rec + ε·X̃ⱼ·sd_y). Плюс напрямок істини (ε·r) як еталон.
Вихід: out0, e0 = out0 − m_true, A_j = (out_j − out0)/ε. Фіт і LOO окремо (fitdir.py).
ПЕРЕДБАЧЕННЯ: LOO-виграш у MSE виходу ≥×1.10 з шансом 20%; ≤×1.03 з шансом 60%. usage: chdir.py <k> <out.npz>"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 13, 14); EPS = float(os.environ.get("EPS", "0.3"))
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
zc = np.load("/home/trim/wblogs/chpair_n%d.npz" % k)
def feats(li):
    D3, g4, mu, var = zc["D3_%d" % li], zc["g4_%d" % li], zc["mu_%d" % li], zc["var_%d" % li]
    D21, C = zc["D21_%d" % li], zc["Coff_%d" % li]; t = mu / np.sqrt(var)
    X = np.stack([D3, g4, mu, var, t, D21.sum(1), np.abs(D21).sum(1), (C * C).sum(1), (D21 * C).sum(1), (D21 * D21).sum(1), D21.sum(0), C.sum(1)], 1).astype(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return X, D3.astype(np.float64), g4.astype(np.float64), zc["c3t"][li].astype(np.float64), zc["c4t"][li].astype(np.float64)
F = {li: feats(li) for li in LS}
NAMES = ["D3", "g4", "mu", "var", "t", "d21sum", "d21abs", "csq", "d21c", "d21sq", "d21col", "csum"]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(c3t=None, c4t=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": [], "D3set": (), "G4set": (), "K5set": ()}
    if c3t is not None: o.update(D3set=LS, c3t=[cp.asarray(c3t.get(l, np.zeros(1024)).astype(np.float32)) for l in range(16)])
    if c4t is not None: o.update(G4set=LS, c4t=[cp.asarray(c4t.get(l, np.zeros(1024)).astype(np.float32)) for l in range(16)])
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    mod.ORAC = None
    return p[-1]
out0 = run(); e0 = out0 - m_true[15]; mse0 = float(np.mean(e0 ** 2)); print("base mse", mse0, flush=True)
res = {"k": k, "out0": out0, "m_true": m_true[15], "names": np.array(NAMES), "eps": EPS}
sd3 = {li: float((F[li][3] - F[li][1]).std()) for li in LS}; sd4 = {li: float((F[li][4] - F[li][2]).std()) for li in LS}
res["sd3"] = np.array([sd3[l] for l in LS]); res["sd4"] = np.array([sd4[l] for l in LS])
# truth directions at ε (linear-regime reference) and at full strength
res["A3_true"] = (run(c3t={li: F[li][1] + EPS * (F[li][3] - F[li][1]) for li in LS}) - out0) / EPS
res["A4_true"] = (run(c4t={li: F[li][2] + EPS * (F[li][4] - F[li][2]) for li in LS}) - out0) / EPS
res["out_true3"] = run(c3t={li: F[li][3] for li in LS}); res["out_true34"] = run(c3t={li: F[li][3] for li in LS}, c4t={li: F[li][4] for li in LS})
print("truth3 x%.3f truth34 x%.3f" % (mse0 / np.mean((res["out_true3"] - m_true[15]) ** 2), mse0 / np.mean((res["out_true34"] - m_true[15]) ** 2)), flush=True)
A3 = []; A4 = []
for j in range(12):
    A3.append((run(c3t={li: F[li][1] + EPS * sd3[li] * F[li][0][:, j] for li in LS}) - out0) / EPS)
    A4.append((run(c4t={li: F[li][2] + EPS * sd4[li] * F[li][0][:, j] for li in LS}) - out0) / EPS)
    print("dir %-7s |A3| %.3e |A4| %.3e  cos(e0,A3) %+.3f cos(e0,A4) %+.3f" % (NAMES[j], np.linalg.norm(A3[-1]), np.linalg.norm(A4[-1]),
          np.dot(e0, A3[-1]) / (np.linalg.norm(e0) * np.linalg.norm(A3[-1]) + 1e-30), np.dot(e0, A4[-1]) / (np.linalg.norm(e0) * np.linalg.norm(A4[-1]) + 1e-30)), flush=True)
    res["A3"] = np.array(A3); res["A4"] = np.array(A4); np.savez(OUT, **res)
print("saved", OUT, flush=True)
