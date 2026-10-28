"""21.10.2026 white-box-f9: точна парна форма c4 передактивації з K22 ланцюга замість рядкових сум.
g4new_j = Σ_a W_aj⁴ K4v_a + 3 Σ_{a≠b} W_aj² W_bj² K22_ab (K4v, K22: пост-ReLU шару l−1 з ланцюга). Порівняння з істиною c34 (γ4 на живих)
проти g4row ланцюга; далі в ланцюгу G4set: c4t = g4row + λ (g4new − g4row) на шарах 2..15 (реалізовно, без істини), MSE виходу.
ПЕРЕДБАЧЕННЯ: corr(γ4new, γ4) 0.6…0.9 (проти 0.2); у виході ×1.05…1.15 при λ≈1. Шанс ≥ ×1.05: 35%.
usage: g4new.py <k>"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
k = int(sys.argv[1]); LS = tuple(range(2, 16))
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
c4 = np.load("/home/trim/wblogs/c34_n%d.npz" % k)["c4"]
Wg = [cp.asarray(x.numpy()) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
def run(c4t=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": [], "K22rec": tuple(l - 1 for l in LS)}
    if c4t is not None: o.update(G4set=LS, c4t=c4t)
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in o["pre"]}
    K22 = o.get("K22", {}); rec = {li: vs for li, vs in o["rec"]}
    mod.ORAC = None
    return float(np.mean((p[-1] - m_true[15]) ** 2)), pre, K22, rec
b, pre, K22, rec = run()
g4n = []
for l in range(16):
    if l in LS and (l - 1) in K22:
        Wl = w[l].numpy().astype(np.float64); W2 = Wl * Wl
        Kp = cp.asnumpy(cp.asarray(K22[l - 1])).astype(np.float64); np.fill_diagonal(Kp, 0.0)
        k4 = cp.asnumpy(cp.asarray(rec[l - 1][3])).astype(np.float64)
        g4n.append((W2 * W2).T @ k4 + 3.0 * np.sum(W2 * (Kp @ W2), axis=0))
    else:
        g4n.append(None)
for l in (6, 10, 14):
    mu, var, D3, g4 = pre[l]; live = np.abs(D3) > 1e-12; s4 = var[live] ** 2
    gt, gc, gn = c4[l][live] / s4, g4[live] / s4, g4n[l][live] / s4
    print("net %d L%d corr γ4: g4row %.3f  g4new %.3f | нахил new/істина %.3f row/істина %.3f | rel new %.3f row %.3f" % (k, l,
          np.corrcoef(gc, gt)[0, 1], np.corrcoef(gn, gt)[0, 1], gn @ gt / (gt @ gt), gc @ gt / (gt @ gt),
          np.sqrt(np.mean((gn - gt) ** 2) / np.mean(gt ** 2)), np.sqrt(np.mean((gc - gt) ** 2) / np.mean(gt ** 2))), flush=True)
line = "net %d base %.4e" % (k, b)
for lam in (0.5, 1.0):
    c4t = [cp.asarray((pre[l][3] + lam * (g4n[l] - pre[l][3]) if g4n[l] is not None and pre.get(l) and pre[l][3] is not None else np.zeros(1024)).astype(np.float32)) for l in range(16)]
    line += "  λ%.1f ×%.4f" % (lam, b / run(c4t)[0])
print(line, flush=True)
