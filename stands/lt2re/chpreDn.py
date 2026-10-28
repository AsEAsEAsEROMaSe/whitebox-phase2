"""chpre (20.10.2026 white-box-89). edge14: Еджворт зі СПРАВЖНІМИ γ3,γ4 по нейрону знімає похибку E[ReLU] ×50 (1.0e-3σ → 2e-5σ).
Тут: γ3,γ4 передактивацій, які несе сам ланцюг (D3, g4row з хука), проти справжніх МК 2^24 на шарах 12,14; і Еджворт з γ ланцюга
проти справжнього E[ReLU]. ПЕРЕДБАЧЕННЯ: corr(γ3 ланц, справж) 0.8–0.95, γ4 ланцюга грубий (corr <0.5) ⟹ похибка кроку в γ4. Шанс: 50%.
usage: chpre.py <k> <out.json>"""
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
k, OUT = int(sys.argv[1]), sys.argv[2]; LS = (12, 14); N = int(float(os.environ.get("NS", str(2 ** 24)))); B = 8192
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
A = tuple(range(1, 16))
mod.ORAC = {"K": -1, "mode": "", "truth": T0, "rec": [], "K21rec": A, "k3p": {}, "Wn": Wg, "d3new": A, "recDn_only": True}
est.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k)), 2 ** 41)
pre = {li: [None if v is None else cp.asnumpy(cp.asarray(v)).astype(np.float64) for v in vs] for li, *vs in mod.ORAC["pre"]}
DN = {li: (cp.asnumpy(cp.asarray(a_)).astype(np.float64), cp.asnumpy(cp.asarray(b_)).astype(np.float64)) for li, (a_, b_) in mod.ORAC["Dn"].items()}
mod.ORAC = None; del Wg
cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
dev = "cuda"; W = w.to(dev); gg = torch.Generator(device=dev).manual_seed(22000 + k)
S = {l: torch.zeros(5, 1024, dtype=torch.float64, device=dev) for l in LS}; sh = {}; n = 0
while n < N:
    h = torch.randn(B, 1024, generator=gg, device=dev)
    for l in range(15):
        p = h @ W[l]
        if l in LS:
            if l not in sh: sh[l] = p.mean(0)
            x = (p - sh[l]).double(); a = S[l]
            a[0] += x.sum(0); a[1] += (x * x).sum(0); a[2] += (x ** 3).sum(0); a[3] += (x ** 4).sum(0); a[4] += torch.relu(p).double().sum(0)
        h = torch.relu(p)
    n += B
from math import erf
res = {"k": k}
def edge(t, g3, g4):
    ph = np.exp(-t * t / 2) / math.sqrt(2 * math.pi); Ph = 0.5 * (1 + np.vectorize(erf)(t / math.sqrt(2)))
    return ph + t * Ph - g3 * t * ph / 6 + g4 * (t * t - 1) * ph / 24 + g3 * g3 * (t ** 4 - 6 * t * t + 3) * ph / 72
for l in LS:
    a = (S[l] / n).cpu().numpy(); m1, m2, m3, m4, rl = a
    var = m2 - m1 ** 2; sd = np.sqrt(var); c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3; c4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4 - 3 * var ** 2
    mu = m1 + sh[l].double().cpu().numpy(); t = mu / sd; g3 = c3 / sd ** 3; g4 = c4 / sd ** 4; R = rl / sd
    cmu, cvar, cD3, cg4 = pre[l]
    csd = np.sqrt(cvar); out = {"mu_relerr_rms": float(np.sqrt(np.mean((cmu - mu) ** 2)) / sd.mean()), "sd_relerr_rms": float(np.sqrt(np.mean((csd / sd - 1) ** 2)))}
    for nm, v in (("D3", cD3), ("g4row", cg4)):
        if v is None: continue
        for tgt_nm, tgt in (("g3", g3), ("g4", g4), ("c3", c3), ("c4", c4)):
            out["corr_%s_%s" % (nm, tgt_nm)] = float(np.corrcoef(v, tgt)[0, 1])
        out["ratio_%s_c3" % nm] = float(np.median(v / c3)); out["ratio_%s_c4" % nm] = float(np.median(v / c4))
    out["resid_gauss"] = float(np.sqrt(np.mean((R - edge(t, 0 * t, 0 * t)) ** 2)))
    out["resid_true_g34"] = float(np.sqrt(np.mean((R - edge(t, g3, g4)) ** 2)))
    if cD3 is not None:
        g3c = cD3 / csd ** 3
        out["resid_chain_g3_true_g4"] = float(np.sqrt(np.mean((R - edge(t, g3c, g4)) ** 2)))
        out["g3chain_vs_true_rms"] = float(np.sqrt(np.mean((g3c - g3) ** 2))); out["g3_true_sd"] = float(g3.std())
    dn, d3r = DN[l]
    out = {"corr_Dn_c3": float(np.corrcoef(dn, c3)[0, 1]), "slope_c3_on_Dn": float(dn @ c3 / (dn @ dn)), "corr_D3rec_c3": float(np.corrcoef(d3r, c3)[0, 1]),
           "corr_D3rec_vs_pre": float(np.corrcoef(d3r, cD3)[0, 1]), "rms_Dn_minus_c3_over_sd": float(np.sqrt(np.mean((dn - c3) ** 2)) / c3.std()),
           "rms_D3_minus_c3_over_sd": float(np.sqrt(np.mean((cD3 - c3) ** 2)) / c3.std()), "corr_D3_c3": out["corr_D3_c3"] if "corr_D3_c3" in out else None}
    res[str(l)] = out
    print("net %d L%d " % (k, l) + " ".join("%s %.4g" % kv for kv in out.items()), flush=True)
json.dump(res, open(OUT, "w"), indent=1)
