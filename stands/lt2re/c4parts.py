"""c3parts (20.10.2026 white-box-89). chpre2: справжні c3/c4 передактивацій дають ×1.90; ланцюг знає c3 з corr 0.84.
Розклад справжнього c3_j = Σ_a W³ κ_aaa (diag) + 3Σ_{a≠b} W_a² W_b E[x_a² x_b] (pair) + решта (triple, a,b,c різні),
x = h_{l-1} − m_true (центрування істиною корпусу). Похибка D3 ланцюга регресується на частини: яку частину ланцюг псує.
ПЕРЕДБАЧЕННЯ: diag і pair ланцюг тримає (corr > 0.95), похибка D3 іде за triple (факторизований K3). Шанс: 55%.
usage: c3parts.py <k> <out.json>"""
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
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 14); N = int(float(os.environ.get("NS", str(2 ** 23)))); B = 8192
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
Wg = [cp.asarray(x.numpy()) for x in w]
T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "K21rec": (11, 13), "K22rec": (11, 13)}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
K22c = {li: cp.asnumpy(cp.asarray(v)).astype(np.float64) for li, v in mod.ORAC["K22"].items()}
K21c = {li: cp.asnumpy(cp.asarray(v)).astype(np.float64) for li, v in mod.ORAC["K21"].items()}
rec = {li: [cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, vs in mod.ORAC["rec"]}
mod.ORAC = None; del Wg
cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(11000 + k)
mt = {l: torch.tensor(m_true[l - 1], dtype=torch.float32, device=dev) for l in LS}
acc = {l: {"x2": torch.zeros(1024, dtype=torch.float64, device=dev), "x3": torch.zeros(1024, dtype=torch.float64, device=dev),
           "G": torch.zeros(1024, 1024, dtype=torch.float64, device=dev), "p3": torch.zeros(1024, dtype=torch.float64, device=dev),
           "p1": torch.zeros(1024, dtype=torch.float64, device=dev), "p2": torch.zeros(1024, dtype=torch.float64, device=dev),
           "Q22": torch.zeros(1024, 1024, dtype=torch.float64, device=dev), "Q31": torch.zeros(1024, 1024, dtype=torch.float64, device=dev),
           "Q11": torch.zeros(1024, 1024, dtype=torch.float64, device=dev), "x4": torch.zeros(1024, dtype=torch.float64, device=dev),
           "p4": torch.zeros(1024, dtype=torch.float64, device=dev)} for l in LS}
n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(15):
        if l in acc:
            a = acc[l]; x = h - mt[l]; q = x @ W[l]
            a["x2"] += (x * x).double().sum(0); a["x3"] += (x ** 3).double().sum(0)
            a["G"] += ((x * x).T @ x).double()
            x2 = x * x; a["Q22"] += (x2.T @ x2).double(); a["Q31"] += ((x2 * x).T @ x).double(); a["Q11"] += (x.T @ x).double()
            a["x4"] += (x2 * x2).double().sum(0); a["p4"] += (q.double() ** 4).sum(0)
            a["p1"] += q.double().sum(0); a["p2"] += (q.double() ** 2).sum(0); a["p3"] += (q.double() ** 3).sum(0)
        h = torch.relu(h @ W[l])
    n += B
res = {"k": k}
for l in LS:
    a = {kk: (v / n).cpu().numpy() for kk, v in acc[l].items()}
    Wl = w[l].numpy().astype(np.float64)
    G = a["G"]; k3d = np.diag(G).copy()                     # E[x_a^3]
    q1, q2, q3 = a["p1"], a["p2"], a["p3"]                  # moments of q = x@W (centered by truth ⟹ q1 ≈ 0)
    c3 = q3 - 3 * q1 * q2 + 2 * q1 ** 3
    diag = (Wl ** 3 * k3d[:, None]).sum(0)
    Go = G - np.diag(k3d)                                   # E[x_a^2 x_b], a≠b
    pair = 3 * ((Wl ** 2) * (Go @ Wl)).sum(0)
    triple = c3 - diag - pair
    cmu, cvar, cD3, cg4 = pre[l]
    e = cD3 - c3
    out = {"var_c3": float(c3.var()), "share": {nm: float(np.var(v) / c3.var()) for nm, v in (("diag", diag), ("pair", pair), ("triple", triple))},
           "corr_D3_c3": float(np.corrcoef(cD3, c3)[0, 1]), "rms_e_over_sd_c3": float(np.sqrt(np.mean(e ** 2)) / c3.std()),
           "chain_k3diag_corr": float(np.corrcoef(rec[l - 1][2], k3d)[0, 1]), "chain_k3diag_ratio": float(np.median(rec[l - 1][2] / k3d))}
    X = np.stack([diag, pair, triple], 1); beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    out["e_on_parts_beta"] = beta.tolist(); out["e_R2_parts"] = float(1 - np.mean((e - X @ beta) ** 2) / np.mean(e ** 2))
    for nm, v in (("diag", diag), ("pair", pair), ("triple", triple)):
        b1 = float(v @ e / (v @ v)); out["e_on_" + nm] = [b1, float(1 - np.mean((e - b1 * v) ** 2) / np.mean(e ** 2))]
    Kc = K21c[l - 1]
    for nm, M in (("K21", Kc), ("K21T", Kc.T)):
        Mo = M - np.diag(np.diag(M))
        out["corr_%s_vs_Go_offdiag" % nm] = float(np.corrcoef(Mo[~np.eye(1024, dtype=bool)], Go[~np.eye(1024, dtype=bool)])[0, 1])
        pc = 3 * ((Wl ** 2) * (Mo @ Wl)).sum(0)
        out["corr_pair_%s_vs_pair" % nm] = float(np.corrcoef(pc, pair)[0, 1])
        out["slope_pair_%s" % nm] = float(pc @ pair / (pc @ pc))
        d3n = diag + pc + 0 * triple
        out["corr_diag+pair%s_vs_c3" % nm] = float(np.corrcoef(d3n, c3)[0, 1])
        out["corr_D3_vs_pair%s" % nm] = float(np.corrcoef(cD3, pc)[0, 1])
    q4 = a["p4"]; v2 = q2 - q1 ** 2
    c4 = q4 - 4 * q1 * q3 + 6 * q1 ** 2 * q2 - 3 * q1 ** 4 - 3 * v2 ** 2
    C = a["Q11"]; vx = np.diag(C).copy()
    k4d = a["x4"] - 3 * vx ** 2
    kaabb = a["Q22"] - np.outer(vx, vx) - 2 * C ** 2
    kaaab = a["Q31"] - 3 * vx[:, None] * C
    off = ~np.eye(1024, dtype=bool)
    p4d = (Wl ** 4 * k4d[:, None]).sum(0)
    Kb = np.where(off, kaabb, 0); Kab = np.where(off, kaaab, 0)
    p22 = 3 * ((Wl ** 2) * (Kb @ (Wl ** 2))).sum(0)
    p31 = 4 * ((Wl ** 3) * (Kab @ Wl)).sum(0)
    rest4 = c4 - p4d - p22 - p31
    K22 = K22c[l - 1]
    out["c4_share"] = {nm: float(np.var(v) / np.var(c4)) for nm, v in (("diag", p4d), ("aabb", p22), ("aaab", p31), ("rest", rest4))}
    out["c4_mean"] = float(c4.mean()); out["c4_parts_mean"] = [float(v.mean()) for v in (p4d, p22, p31, rest4)]
    out["corr_K22_kaabb"] = float(np.corrcoef(K22[off], kaabb[off])[0, 1]); out["slope_K22_kaabb"] = float(K22[off] @ kaabb[off] / (K22[off] @ K22[off]))
    p22c = 3 * ((Wl ** 2) * (np.where(off, K22, 0) @ (Wl ** 2))).sum(0)
    out["corr_p22chain_p22"] = float(np.corrcoef(p22c, p22)[0, 1])
    out["corr_g4row_c4"] = float(np.corrcoef(cg4, c4)[0, 1]); out["g4row_on_parts"] = np.linalg.lstsq(np.stack([p4d, p22, p31, rest4, np.ones(1024)], 1), cg4, rcond=None)[0].tolist()
    out["corr_g4row_p4d"] = float(np.corrcoef(cg4, p4d)[0, 1])
    np.savez("/home/trim/wblogs/c4arr_n%d_L%d.npz" % (k, l), c4=c4, p4d=p4d, p22=p22, p31=p31, rest=rest4, g4=cg4, p22c=p22c)
    np.savez("/home/trim/wblogs/c3arr_n%d_L%d.npz" % (k, l), c3=c3, diag=diag, pair=pair, triple=triple, D3=cD3, pK21=3 * ((Wl ** 2) * ((Kc - np.diag(np.diag(Kc))) @ Wl)).sum(0), k3chain=rec[l - 1][2], k3d=k3d)
    res[str(l)] = out
    print("net %d L%d " % (k, l) + json.dumps(out), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
