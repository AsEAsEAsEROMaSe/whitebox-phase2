"""21.10.2026 white-box-f9: РЕАЛІЗОВНЕ накопичене море — зі стану ланцюга: μ_s (передактивації), C_pre_s = W_sᵀ C_post(s−1) W_s
(C_post з хука Crec; шар 0: W_0ᵀW_0), глибина 6. g4row_L += β_L sea_L, β — середнє власних β мереж 0 і 1 (seaorac, істинний стан).
Також: corr(sea_chain, sea_true) там, де є кеш sea_n<k>.npz.
ПЕРЕДБАЧЕННЯ: на мережах 2,3 ×1.02…1.05. Шанс ≥ ×1.03: 40%.
usage: seachain.py <k>"""
import os, sys, math, types, importlib.util
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_grad_enabled(False); torch.set_num_threads(1)
k = int(sys.argv[1]); DEP = 6; dev = "cuda"
beta = np.mean([np.load("/home/trim/wblogs/sea_beta_n%d.npy" % j) for j in (0, 1)], 0)
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
with np.load(os.path.join("/home/trim/wb2data/corpus14k", "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
    m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
Wg = [cp.asarray(x.numpy()) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
LS = tuple(range(2, 16))
def run(c4t=None, crec=False):
    o = {"K": -1, "mode": "", "truth": T0, "rec": [], "Crec": crec}
    if c4t is not None: o.update(G4set=LS, c4t=[cp.asarray(x.astype(np.float32)) for x in c4t])
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in o["pre"]}
    Cp = {li: cp.asnumpy(cp.asarray(v)).astype(np.float64) for li, v in o.get("Cpost", {}).items()}
    mod.ORAC = None; o.clear()
    cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
    return float(np.mean((p[-1] - m_true[15]) ** 2)), pre, Cp
b, pre, Cp = run(crec=True)
print("recorded Cpost layers", len(Cp), flush=True)
Wd = w.to(dev).double(); st = {}
for s in range(16):
    Cpre = Wd[s].T @ Wd[s] if s == 0 else Wd[s].T @ torch.as_tensor(Cp[s - 1], device=dev) @ Wd[s]
    mu = torch.as_tensor(pre[s][0], device=dev); sd = torch.sqrt(torch.diag(Cpre)); t = mu / sd
    phi = torch.exp(-t ** 2 / 2) / math.sqrt(2 * math.pi); Phi = 0.5 * (1 + torch.erf(t / math.sqrt(2)))
    rho = Cpre / torch.outer(sd, sd); rho.fill_diagonal_(0.0)
    st[s] = (sd * Phi, sd * phi, -sd * t * phi, rho, Phi)
SEA = np.zeros((16, 1024))
for L in range(2, 16):
    M = Wd[L].clone(); acc = torch.zeros(1024, dtype=torch.float64, device=dev)
    for s in range(L - 1, max(L - 1 - DEP, -1), -1):
        a1, a2, a3, rho, Phi = st[s]
        V = rho @ (M * a1[:, None]); U = M * a2[:, None] * V
        acc += 4.0 * torch.sum(M * a3[:, None] * V ** 3, 0) + 12.0 * torch.sum(U * (rho @ U), 0)
        M = Wd[s] @ (Phi[:, None] * M)
    SEA[L] = acc.cpu().numpy()
del st, Wd, Cp; torch.cuda.empty_cache()
if os.path.exists("/home/trim/wblogs/sea_n%d.npz" % k):
    ST = np.load("/home/trim/wblogs/sea_n%d.npz" % k)["sea"]
    print("net %d corr(sea_chain, sea_true) L6 %.4f L10 %.4f L14 %.4f" % (k, *[np.corrcoef(SEA[l], ST[l])[0, 1] for l in (6, 10, 14)]), flush=True)
G4 = np.stack([pre[l][3] if (l in pre and pre[l][3] is not None) else np.zeros(1024) for l in range(16)])
res = {}
for f in (0.5, 1.0, 1.5):
    res["b%.1f" % f] = run(G4 + f * beta[:, None] * SEA)[0]
print("net %d base %.4e " % (k, b) + " ".join("%s ×%.4f" % (kk, b / v) for kk, v in res.items()), flush=True)
np.save("/home/trim/wblogs/seachain_n%d.npy" % k, SEA)
