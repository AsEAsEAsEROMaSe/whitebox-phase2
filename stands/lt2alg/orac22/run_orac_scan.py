"""run_orac (22.10.2026, white-box-b9): оракул справжніх зрізів κ4 передактивацій у ланцюгу 337568 (GPU-шим).
wk4m ← справжній (2,2), wk431 ← справжній (3,1) з МК 2^22 (mc22.py), на шарах режиму regen.
Конфіги: base · self (самопідстановка, мусить дати базу біт у біт) · 22A/22B (половини) · 22 (середнє) · 22h (×0.5) · 31 · 31h · both.
Вирівнювання шарів: дисперсії ланцюга (var) проти v з МК, найкращий збіг.
СКАН (після першого прогону): масштаб справжнього зрізу і розділення форма/рівень. ПЕРЕДБАЧЕННЯ: рівень (середнє) важить більше за форму; форма справжнього на рівні анзацу ≈ база ±2%.
usage: run_orac.py <nets через кому> <out.json>"""
import os, sys, types, importlib.util, json, time
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
EST = "/home/trim/wb2lean/lt2alg/orac22/est_orac.py"
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
SCALE = (2.0 / 1024.0) ** 0.5
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("est_orac", EST)
mod = importlib.util.module_from_spec(spec); sys.modules["est_orac"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=os.path.dirname(EST)))
nets = [int(x) for x in sys.argv[1].split(",")]; out = sys.argv[2]
res = {}
for k in nets:
    s = int(t["mlp_seed"][k])
    wss, _, ess = np.random.SeedSequence(s).spawn(3)
    rng = np.random.default_rng(wss)
    Wn = [(rng.standard_normal((1024, 1024)) * SCALE).astype(np.float32) for _ in range(16)]
    tr = t["all_layer_means"][k][-1].astype(np.float64)
    mcp = "/mnt/gguf/wb_lt2alg/orac22/mc_n%d.npz" % k
    while not os.path.exists(mcp):
        time.sleep(20)
    mc = np.load(mcp)

    def run(cfg):
        mod._ORAC.clear(); mod._ORAC.update(cfg)
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=[cp.asarray(x) for x in Wn], seed=int(ess.generate_state(1)[0]), name=str(k))
        p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41)))[-1].astype(np.float64)
        return float(np.mean((p - tr) ** 2))

    t0 = time.time()
    base = run({})
    mse_diag = run({"diag": True}); log = list(mod._ORAC.get("log", [])); mod._ORAC.clear()
    print("net %d: хук спрацював на %d шарах" % (k, len(log)), flush=True)
    vmc = mc["vA"]                                        # (16, n)
    amap = {}
    for li, var, w22, w31 in log:
        vv = cp.asnumpy(var).astype(np.float64)
        errs = [float(np.mean(np.abs(vv - vmc[l]) / (vmc[l] + 1e-12))) for l in range(16)]
        amap[int(li)] = int(np.argmin(errs))
        m = amap[int(li)]
        off = ~np.eye(1024, dtype=bool)
        c22 = float(np.corrcoef(cp.asnumpy(w22)[off], mc["k22A"][m][off])[0, 1])
        r22 = float(np.corrcoef(mc["k22A"][m][off], mc["k22B"][m][off])[0, 1])
        c31 = float(np.corrcoef(cp.asnumpy(w31)[off], mc["k31A"][m][off])[0, 1]) if w31 is not None else float("nan")
        r31 = float(np.corrcoef(mc["k31A"][m][off], mc["k31B"][m][off])[0, 1])
        print("net %d li %2d ↔ МК ш%2d (відн. похибка var %.2e) | corr анзац/істина: (2,2) %.3f (3,1) %.3f | надійність половин %.3f %.3f" % (
            k, li, m, min(errs), c22, c31, r22, r31), flush=True)
    del log
    tab = lambda key: {li: cp.asarray(mc[key][m]) for li, m in amap.items()}
    avg = lambda a, b: {li: (cp.asarray(mc[a][m]) + cp.asarray(mc[b][m])) * 0.5 for li, m in amap.items()}
    A22 = avg("k22A", "k22B"); A31 = avg("k31A", "k31B")
    cfgs = [("22x1.25", {"k22": A22, "s22": 1.25}), ("22x1.6", {"k22": A22, "s22": 1.6}), ("22x2", {"k22": A22, "s22": 2.0}),
            ("22lvl", {"k22": A22, "lvl22": True}), ("ans_lvl22", {"k22": A22, "ans_lvl22": True}),
            ("31x0.25", {"k31": A31, "s31": 0.25}), ("31lvl", {"k31": A31, "lvl31": True})]
    r = {"base": base, "diag": mse_diag}
    for name, cfg in cfgs:
        r[name] = run(cfg)
        print("net %d %-5s MSE %.6e  ×%.4f проти бази" % (k, name, r[name], base / r[name]), flush=True)
    res[k] = r
    json.dump(res, open(out, "w"), indent=1)
    print("net %d готово за %.0f с; база %.6e (diag %.6e)" % (k, time.time() - t0, base, mse_diag), flush=True)
names = ["22x1.25", "22x1.6", "22x2", "22lvl", "ans_lvl22", "31x0.25", "31lvl"]
for nm in names:
    g = np.exp(np.mean([np.log(res[k]["base"] / res[k][nm]) for k in nets]))
    print("РАЗОМ %-5s геосер. ×%.4f  (%s)" % (nm, g, " ".join("%.4f" % (res[k]["base"] / res[k][nm]) for k in nets)), flush=True)
print("ORACDONE", flush=True)
