"""run_orac31t (27.10.2026): перевірка орієнтації (3,1) в оракулі 22.10 (зауваження Nurlanggs 18252 / kai 18246: оракул має входити як Uᵀ).
МК k31[i,j] = E[x_i³ x_j] (куб на рядку). Анзац ланцюга wk431 = C_off·скаляр симетричний ⟹ corr орієнтацію не розрізняє, лише вихід.
Конфіги: self · 31 (як 22.10, мусить дати ×0.8040 на 0,2,1) · 31T (транспонована) · 31T s0.5/0.75 · 31S (симетрична частина) · 31Tl (форма 31T на рівні анзацу).
ДОГІН bothT (27.10, після першого прогону 31T ×1.009, 31T75 ×1.055): передбачення bothT ×0.90…0.95 (22 ×0.904 домінує). ПЕРЕДБАЧЕННЯ до прогону: 31T теж <1 (баланс), найімовірніше ×0.80…0.95; шанс, що 31T >1.0, 25%; якщо так, твердження §4.2 про (3,1) у write-up хибне.
usage: run_orac31t.py <nets> <out.json>"""
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
    T = lambda d: {li: v.T.copy() for li, v in d.items()}
    k31 = avg("k31A", "k31B"); k31T = T(k31); k31S = {li: 0.5 * (v + v.T) for li, v in k31.items()}
    cfgs = [("self", {"self": True}), ("31Th", {"k31": k31T, "s31": 0.5}), ("31T75", {"k31": k31T, "s31": 0.75}),
            ("bothT", {"k22": avg("k22A", "k22B"), "k31": k31T}), ("bothT75", {"k22": avg("k22A", "k22B"), "k31": k31T, "s31": 0.75})]
    r = {"base": base, "diag": mse_diag}
    for name, cfg in cfgs:
        r[name] = run(cfg)
        print("net %d %-5s MSE %.6e  ×%.4f проти бази" % (k, name, r[name], base / r[name]), flush=True)
    res[k] = r
    json.dump(res, open(out, "w"), indent=1)
    print("net %d готово за %.0f с; база %.6e (diag %.6e)" % (k, time.time() - t0, base, mse_diag), flush=True)
names = ["self", "31Th", "31T75", "31S", "bothT", "bothT75"]
for nm in names:
    g = np.exp(np.mean([np.log(res[k]["base"] / res[k][nm]) for k in nets]))
    print("РАЗОМ %-5s геосер. ×%.4f  (%s)" % (nm, g, " ".join("%.4f" % (res[k]["base"] / res[k][nm]) for k in nets)), flush=True)
print("ORACDONE", flush=True)
