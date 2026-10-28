"""20.10.2026 white-box-16: оракульна драбина на ланцюгу B (sb2, LT_OFF=1, GPU-шим gpufnp).
На шарі K у ланцюг підставляються справжні діагональні кумулянти після ReLU (корпус N=1e8: m1..m4 -> k1..k4):
режими m (k1), v (k2), mv, mv34 (k1..k4), 34 (k3,k4). Міряємо final raw MSE проти бази (той самий ланцюг без підстановки).
Пломби: K=15 m мусить дати MSE 0 (перевіряє перестановку нейронів); шар 0 кумулянти ланцюга = істина з точністю шуму.
Побічно: власна похибка кумулянтів ланцюга по шарах (де народжується похибка).
ПЕРЕДБАЧЕННЯ (до запуску): m на K=14 дає ×3…×10 (але спільний шум істини завищує біля виходу); m на K=1..4 ×1.0…1.2;
v сам ×1.0…1.1 (при alpha~3 дисперсія у відповідь майже не входить); mv34 ≈ m. Шанс, що якийсь режим на K<=10 дасть ×2+: ~30%.
usage: orac_ladder.py <a-b> <out.json>"""
import json, os, sys, time, types, importlib.util
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import torch
torch.set_num_threads(1)

EST = "/home/trim/wb2lean/lt2re/estimator_sb2_orac.py"
SH = "/home/trim/wb2data/corpus14k"
SCALE = (2.0 / 1024.0) ** 0.5
KS = [int(x) for x in os.environ.get("KS", "1,4,7,10,12,14").split(",")]
MODES = os.environ.get("MODES", "m,v,mv,mv34,34").split(",")


def cumulants(M):
    m1, m2, m3, m4 = (M[i].astype(np.float64) for i in range(4))   # M (4,16,1024)
    k2 = m2 - m1 ** 2
    k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
    k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
    return np.stack([m1, k2, k3, k4], axis=1)   # (16,4,1024)


def main():
    lo, hi = (int(x) for x in sys.argv[1].split("-")); out = sys.argv[2]
    res = json.load(open(out)) if os.path.exists(out) else {}
    saved = gpufnp.install()
    spec = importlib.util.spec_from_file_location("est_orac", EST)
    mod = importlib.util.module_from_spec(spec); sys.modules["est_orac"] = mod; spec.loader.exec_module(mod)
    gpufnp.restore(saved)
    est = mod.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/v39fija"))
    for k in range(lo, hi + 1):
        if str(k) in res:
            continue
        fn = "shard%02d.npz" % (k % 16); row = k // 16
        with np.load(os.path.join(SH, fn), allow_pickle=False) as z:
            assert int(z["idx"][row]) == k
            M = np.asarray(z["moments"][row], dtype=np.float64)   # (4,16,1024)
        T = cumulants(M)
        Tg = [[cp.asarray(T[l, j].astype(np.float32)) for j in range(4)] for l in range(16)]
        g = torch.Generator(device="cpu").manual_seed(770000 + k)
        w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE
        Wg = [cp.asarray(x.numpy()) for x in w]; del w

        def run(K, mode):
            mod.ORAC = {"K": K, "mode": mode, "truth": Tg, "rec": []}
            mlp = types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=770000 + k, name=str(k))
            p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41))).astype(np.float64)
            rec = mod.ORAC["rec"]; inj = mod.ORAC.get("inj"); mod.ORAC = None
            if inj is not None:   # seal: injected values, back in natural order, equal the truth
                for j in range(4):
                    if (j == 0 and "m" in mode) or (j == 1 and "v" in mode) or (str(j + 1) in mode and j >= 2):
                        e = np.max(np.abs(cp.asnumpy(inj[j]).astype(np.float64) - T[K, j])) / (np.max(np.abs(T[K, j])) + 1e-30)
                        assert e < 1e-6, (K, mode, j, e)
            return float(np.mean((p[-1] - T[-1, 0]) ** 2)), rec

        t0 = time.time()
        base, rec = run(-1, "")
        assert base < 1e-6, base
        # chain's own cumulant error per layer: rel = ||chain - truth|| / ||truth||
        cerr = []
        for li, vs in rec:
            cerr.append([float(np.linalg.norm(cp.asnumpy(vs[j]).astype(np.float64) - T[li, j]) / (np.linalg.norm(T[li, j]) + 1e-30))
                         for j in range(4)])
        seal = len(rec)
        r = {"base": base, "seal15": seal, "cerr": cerr, "lad": {}}
        for K in KS:
            for mode in MODES:
                r["lad"]["%d_%s" % (K, mode)] = run(K, mode)[0]
        res[str(k)] = r
        tmp = out + ".tmp"; json.dump(res, open(tmp, "w")); os.replace(tmp, out)
        line = " ".join("%s:%.3f" % (key, base / v) for key, v in r["lad"].items() if v > 0)
        print("net %d base %.4e layers %d (%.0fs)  %s" % (k, base, seal, time.time() - t0, line), flush=True)
        print("   cerr l0 %s  l7 %s  last %s" % (["%.2e" % x for x in cerr[0]], ["%.2e" % x for x in cerr[7]],
              ["%.2e" % x for x in cerr[-1]]), flush=True)


if __name__ == "__main__":
    main()
