"""chres (20.10.2026 white-box-89): похибка середнього ланцюга sb2 по нейронах проти кривої переломів breaks2.
breaks2: справжнє E[ReLU]/σ мінус гаус з ТИМИ САМИМИ μ,σ = −1e-3 σ систематично при |t|<2 (усі 3 мережі, шари 10..14).
Тут: e_l = (K1v ланцюга − істина корпусу)/σ_pre по нейронах, біни за справжнім t. Питання: чи похибка ланцюга має ту
саму форму (+1e-3 у |t|<2, ~0 при |t|>3) і яку частку rms(e) вона пояснює (R² біновою кривою, крос між мережами).
ПЕРЕДБАЧЕННЯ: бінова крива пояснює 10–30% rms(e) на шарі 14 (решта перенесена з попередніх шарів, не локальна). Шанс >30%: 25%.
usage: chres.py <a-b> <out.json>"""
import json, os, sys, time, types, importlib.util, math
os.environ["LT_OFF"] = "1"; os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1); torch.set_grad_enabled(False)
EST = "/home/trim/wb2lean/lt2re/estimator_sb2_orc.py"; SH = "/home/trim/wb2data/corpus14k"
SCALE = (2.0 / 1024.0) ** 0.5; LS = list(range(1, 16)); NMC = int(float(os.environ.get("NMC", str(2 ** 22)))); B = 32768

def pre_stats(w, k):
    dev = "cuda"; W = w.to(dev); g = torch.Generator(device=dev).manual_seed(880000 + k)
    s1 = torch.zeros(16, 1024, dtype=torch.float64, device=dev); s2 = torch.zeros_like(s1); n = 0
    while n < NMC:
        h = torch.randn(B, 1024, generator=g, device=dev)
        for l in range(16):
            p = h @ W[l]; pd = p.double(); s1[l] += pd.sum(0); s2[l] += (pd * pd).sum(0); h = torch.relu(p)
        n += B
    mu = s1 / n; sd = torch.sqrt(s2 / n - mu * mu); del W
    return mu.cpu().numpy(), sd.cpu().numpy()

def main():
    lo, hi = (int(x) for x in sys.argv[1].split("-")); out = sys.argv[2]
    res = json.load(open(out)) if os.path.exists(out) else {}
    saved = gpufnp.install()
    spec = importlib.util.spec_from_file_location("est_orc", EST)
    mod = importlib.util.module_from_spec(spec); sys.modules["est_orc"] = mod; spec.loader.exec_module(mod)
    gpufnp.restore(saved)
    est = mod.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
    for k in range(lo, hi + 1):
        if str(k) in res: continue
        t0 = time.time()
        with np.load(os.path.join(SH, "shard%02d.npz" % (k % 16)), allow_pickle=False) as z:
            assert int(z["idx"][k // 16]) == k
            m_true = np.asarray(z["moments"][k // 16][0], dtype=np.float64)
        g = torch.Generator(device="cpu").manual_seed(770000 + k)
        w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE
        cp.get_default_memory_pool().free_all_blocks(); cp.get_default_pinned_memory_pool().free_all_blocks()
        mu, sd = pre_stats(w, k)
        Wg = [cp.asarray(x.numpy()) for x in w]; del w; torch.cuda.empty_cache()
        Tg = [[cp.asarray(m_true[l].astype(np.float32))] * 4 for l in range(16)]
        mod.ORAC = {"K": -1, "mode": "", "truth": Tg, "rec": []}
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k))
        p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41))).astype(np.float64)
        rec = {li: cp.asnumpy(cp.asarray(v[0])).astype(np.float64) for li, v in mod.ORAC["rec"]}
        mod.ORAC = None
        r = {"base": float(np.mean((p[-1] - m_true[15]) ** 2)), "layers": {}}
        for l in LS:
            if l not in rec: continue
            t = mu[l] / sd[l]; e = (rec[l] - m_true[l]) / sd[l]
            r["layers"][str(l)] = {"t": t.tolist(), "e": e.tolist(), "rms_e": float(np.sqrt(np.mean(e * e)))}
        r["seal_pred_vs_rec15"] = float(np.max(np.abs(p[-1] - rec.get(15, p[-1]))))
        res[str(k)] = r
        tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
        print("net %d base %.3e seal %.1e rms_e(10,12,14,15) %s (%.0fs)" % (k, r["base"], r["seal_pred_vs_rec15"],
              " ".join("%.2e" % r["layers"][str(l)]["rms_e"] for l in (10, 12, 14, 15) if str(l) in r["layers"]), time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
