"""21.10.2026 white-box-f9: D21 ланцюга (повна матриця передактивацій, шари 10,12,14) проти справжніх парних κ(a,a,b) =
E[x_a² x_b], x = p − E p (передактивації, МК 2^23, центрування власним середнім; сід 33000+k). Орієнтації: D21 і D21.T.
Міра: corr і нахил на позадіагоналі (живі рядки/стовпці), rel rms; і окремо внесок у D3: 3Σ_{a≠b} W_ja²W_jb D21 — тут не потрібен
(D21 уже передактиваційний, тобто p = h@W вже враховано). ПЕРЕДБАЧЕННЯ: corr 0.95…0.99, похибка D21 росте з глибиною як D3.
usage: d21cmp.py <k>"""
import os, sys, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
k = int(sys.argv[1]); LS = (10, 12, 14); N = int(float(os.environ.get("NS", str(2 ** 23)))); B = 8192
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_trip", "/home/trim/wb2lean/lt2re/estimator_sb2_trip.py")
mod = importlib.util.module_from_spec(spec); sys.modules["est_trip"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
g = torch.Generator(device="cpu").manual_seed(770000 + k)
w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * math.sqrt(2.0 / 1024)
Wg = [cp.asarray(x.numpy()) for x in w]
mod.ORAC = {"K": -1, "mode": "", "truth": [[cp.zeros(1024, dtype=cp.float32)] * 4 for _ in range(16)], "rec": [], "D21rec": LS}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
D21 = {l: cp.asnumpy(cp.asarray(v)).astype(np.float64) for l, v in mod.ORAC.get("D21", {}).items()}
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
mod.ORAC = None; del Wg
cp.get_default_memory_pool().free_all_blocks()
print("recorded D21 layers", sorted(D21), flush=True)
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(33000 + k)
mu = {l: torch.tensor(pre[l][0], dtype=torch.float32, device=dev) for l in LS}   # chain mean as centre (shift-invariant up to O(err))
S1 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in LS}; S2 = {l: torch.zeros(1024, dtype=torch.float64, device=dev) for l in LS}
G = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}; G2 = {l: torch.zeros(1024, 1024, dtype=torch.float64, device=dev) for l in LS}; n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(15):
        p = h @ W[l]
        if l in G:
            x = p - mu[l]; S1[l] += x.double().sum(0); S2[l] += (x.double() ** 2).sum(0)
            (G[l] if (n // B) % 2 == 0 else G2[l]).add_(((x * x).T @ x).double())
        h = torch.relu(p)
    n += B
for l in LS:
    m1 = (S1[l] / n).cpu().numpy(); m2 = (S2[l] / n).cpu().numpy(); Ma = (2 * G[l] / n).cpu().numpy(); Mb = (2 * G2[l] / n).cpu().numpy(); M = (Ma + Mb) / 2
    # third cumulant κ(a,a,b) = E[x_a²x_b] − m2_a m1_b − 2 E[x_a x_b] m1_a + 2 m1_a² m1_b ; E[x_a x_b] unknown here, m1 ≈ 0 (chain centre) ⟹ drop O(m1) terms, check size
    K = M - np.outer(m2, m1)
    live = np.abs(pre[l][2]) > 1e-12
    off = ~np.eye(1024, dtype=bool) & np.outer(live, live)
    if l not in D21:
        print("L%d no D21" % l); continue
    Ka, Kb = Ma - np.outer(m2, m1), Mb - np.outer(m2, m1)
    ra = np.corrcoef(Ka[off], Kb[off])[0, 1]
    noise = np.sqrt(np.mean((Ka[off] - Kb[off]) ** 2) / 4); sig = np.sqrt(np.mean(K[off] ** 2))
    print("L%d halves corr %.4f  noise(full) %.3e  rms truth %.3e  ⟹ reliability(full) %.4f" % (l, ra, noise, sig, 1 - noise ** 2 / sig ** 2), flush=True)
    a = D21[l][off]; b = K[off]; c = np.corrcoef(a, b)[0, 1]
    print("L%d D21 corr disattenuated %.4f  slope corrected %.4f" % (l, c / np.sqrt(1 - noise ** 2 / sig ** 2), (a @ b / (b @ b)) / (1 - noise ** 2 / sig ** 2)), flush=True)
    for nm, Dm in (("D21", D21[l]),):
        a, b = Dm[off], K[off]
        print("L%d %-4s corr %.4f slope %.4f rel_rms %.4f | max|m1|/sd %.2e" % (l, nm, np.corrcoef(a, b)[0, 1], a @ b / (b @ b),
              np.sqrt(np.mean((a - b) ** 2) / np.mean(b ** 2)), np.max(np.abs(m1)) / np.sqrt(np.mean(m2))), flush=True)
