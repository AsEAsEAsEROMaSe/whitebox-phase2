"""21.10.2026 white-box-f9: оракул справжнього D21 (парні κ(a,a,b) передактивацій, МК 2^23 з точним центруванням) на шарах 6..14.
Режими: base; d21 λ=0.5, 1.0. Вимір: MSE виходу проти істини корпусу і похибка D3 (запис до підстановки) на шарах 6..14 проти c34.
ПЕРЕДБАЧЕННЯ: якщо похибка D3 народжується з D21 у хабах, λ=1 зменшить похибку D3 вдвічі і дасть ×1.2…1.4. Шанс: 35%.
usage: d21orac.py <k>"""
import os, sys, types, importlib.util, math, json
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k = int(sys.argv[1]); LS = tuple(range(6, 15)); N = int(float(os.environ.get("NS", str(2 ** 23)))); B = 8192
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
c3 = np.load("/home/trim/wblogs/c34_n%d.npz" % k)["c3"]
cache = "/home/trim/wblogs/d21t_n%d.npy" % k
if os.path.exists(cache):
    KT = np.load(cache)
else:
    dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(33000 + k)
    S1 = torch.zeros(16, 1024, dtype=torch.float64, device=dev); S2 = torch.zeros(16, 1024, dtype=torch.float64, device=dev)
    G = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}; Q = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}
    sh = {}; n = 0
    while n < N:
        h = torch.randn(B, 1024, generator=gg, device=dev)
        for l in range(15):
            p = h @ W[l]
            if l in G:
                if l not in sh: sh[l] = p.mean(0)
                x = p - sh[l]; S1[l] += x.double().sum(0); S2[l] += (x.double() ** 2).sum(0)
                G[l] += ((x * x).T @ x).double(); Q[l] += (x.T @ x).double()
            h = torch.relu(p)
        n += B
    KT = np.zeros((16, 1024, 1024), dtype=np.float32)
    for l in LS:
        m1 = (S1[l] / n).cpu().numpy(); m2 = (S2[l] / n).cpu().numpy(); M = (G[l] / n).cpu().numpy(); S = (Q[l] / n).cpu().numpy()
        Kl = M - np.outer(m2, m1) - 2 * S * m1[:, None] + 2 * np.outer(m1 ** 2, m1)
        np.fill_diagonal(Kl, 0.0)
        KT[l] = Kl.astype(np.float32)
    np.save(cache, KT); del W, G, Q; torch.cuda.empty_cache()
Wg = [cp.asarray(x.numpy()) for x in w]; T0 = [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)]
KTg = {l: cp.asarray(KT[l]) for l in LS}
def run(lam=None, mask=None):
    o = {"K": -1, "mode": "", "truth": T0, "rec": []}
    if lam is not None: o.update(D21set=LS, d21t=KTg, lam21=lam, d21mask=mask)
    mod.ORAC = o
    p = cp.asnumpy(cp.asarray(est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41))).astype(np.float64)
    pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
    mod.ORAC = None
    errs = []
    for l in range(6, 15):
        D3 = pre[l][2]; live = np.abs(D3) > 1e-12
        errs.append(np.sqrt(np.mean((D3 - c3[l])[live] ** 2)) / c3[l][live].std())
    return float(np.mean((p[-1] - m_true[15]) ** 2)), errs
b, eb = run()
print("net %d base %.4e  D3err %s" % (k, b, " ".join("%.4f" % x for x in eb)), flush=True)
for lam, mask in ((1.0, None), (1.0, "zero"), (1.0, "nonzero"), (0.5, "nonzero")):
    r, e = run(lam, mask)
    print("net %d prune %s d21 lam %.1f mask %s ×%.4f  D3err %s" % (k, os.environ.get("V30_PRUNE", "1"), lam, mask, b / r, " ".join("%.4f" % x for x in e[-3:])), flush=True)
