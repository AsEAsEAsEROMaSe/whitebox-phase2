"""19.10.2026 white-box-16: дамп ознак LT2 з ланцюга sb2 на офіційних mini 0..99 (відкладена оцінка), за зразком lab/lt_dump.py (MIT).
Ваги з whestbench-сіда (SeedSequence, як tools/mlpdata.py), істина data/mini_truth.npz з репо Susupro.
usage: dump_mini.py <indices> <out> [--step k --off r]"""
import argparse, os, sys, time
os.environ["LT_DUMP"] = "1"
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
import importlib.util
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, SetupContext

B = 2 ** 41
TRUTH = "/home/trim/ext/fc/repo/data/mini_truth.npz"


def weights_from_seed(s):
    wss, _, _ = np.random.SeedSequence(int(s)).spawn(3)
    rng = np.random.default_rng(wss)
    sc = float(np.sqrt(2.0 / 1024))
    return [(rng.standard_normal((1024, 1024)) * sc).astype(np.float32) for _ in range(16)]


def estimator_seed(s):
    _, _, ess = np.random.SeedSequence(int(s)).spawn(3)
    return int(ess.generate_state(1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indices"); ap.add_argument("out")
    ap.add_argument("--est", default="/home/trim/wb2lean/v39fija/estimator_sb2.py")
    ap.add_argument("--step", type=int, default=1); ap.add_argument("--off", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = np.load(TRUTH, allow_pickle=False)
    spec = importlib.util.spec_from_file_location("est_mod", a.est)
    mod = importlib.util.module_from_spec(spec); sys.modules["est_mod"] = mod; spec.loader.exec_module(mod)
    est = mod.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1",
                           submission_dir=os.path.dirname(os.path.abspath(a.est))))
    lo, hi = (int(x) for x in a.indices.split("-"))
    for k in range(lo, hi + 1):
        if (k - lo) % a.step != a.off:
            continue
        out = os.path.join(a.out, "mini_%04d.npz" % k)
        if os.path.exists(out):
            continue
        s = int(t["mlp_seed"][k])
        W = [fnp.array(w, dtype=fnp.float32) for w in weights_from_seed(s)]
        mlp = MLP(width=1024, depth=16, weights=W, seed=estimator_seed(s), name=str(t["mlp_name"][k]))
        mod.LT_DUMP.clear()
        t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            pred = np.asarray(fnp.asarray(est.predict(mlp, B)), dtype=np.float32)
        wall = time.time() - t0
        X = np.stack([np.asarray(x, dtype=np.float32) for _, x, _ in mod.LT_DUMP])
        Phi = np.stack([np.asarray(p, dtype=np.float32) for _, _, p in mod.LT_DUMP])
        assert X.shape[:2] == (16, 1024) and np.isfinite(X).all()
        raw = float(np.mean((pred[-1].astype(np.float64) - t["all_layer_means"][k][-1]) ** 2))
        tmp = out + ".tmp.npz"
        np.savez(tmp, X=X, Phi=Phi, pred=pred, idx=k, seed=s, flops=float(ctx.flops_used), wall=wall)
        os.replace(tmp, out)
        print("mini %d raw %.4e C/B %.4f wall %.1fs" % (k, raw, ctx.flops_used / B, wall), flush=True)


if __name__ == "__main__":
    main()
