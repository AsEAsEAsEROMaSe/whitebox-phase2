"""gpu_dump (19.10.2026, white-box-89): дамп ознак LT2 з ланцюга sb2 на GPU (шим gpufnp, cupy), формат як у
dump_mini.py / dump_corpus.py white-box-16 (X, Phi, pred, idx, seed, wall; flops = -1, бо облік вимкнено; gpu = 1).
Звірка на mini 0 з CPU-дампом: pred rel 9e-7, X rel max 3.5e-4 (f32), 3.8 с на мережу проти 35 с на ядрі CPU.
usage: gpu_dump.py mini <a-b> <out> [--step k --off r]
       gpu_dump.py corpus <a-b> <out> --shards <dir> [--step k --off r]"""
import argparse, os, sys, time, types, importlib.util
os.environ["LT_DUMP"] = "1"
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
import numpy as np
from whestbench import SetupContext   # real flopscope behind whestbench, before the shim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpufnp
cp = gpufnp.cp

EST = "/home/trim/wb2lean/v39fija/estimator_sb2.py"
TRUTH = "/home/trim/ext/fc/repo/data/mini_truth.npz"
SCALE = (2.0 / 1024.0) ** 0.5


def weights_mini(s):
    wss, _, _ = np.random.SeedSequence(int(s)).spawn(3)
    rng = np.random.default_rng(wss)
    return [(rng.standard_normal((1024, 1024)) * SCALE).astype(np.float32) for _ in range(16)]


def estimator_seed(s):
    _, _, ess = np.random.SeedSequence(int(s)).spawn(3)
    return int(ess.generate_state(1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split"); ap.add_argument("indices"); ap.add_argument("out")
    ap.add_argument("--shards", default="/home/trim/wb2data/corpus14k")
    ap.add_argument("--step", type=int, default=1); ap.add_argument("--off", type=int, default=0)
    ap.add_argument("--rev", action="store_true")   # third worker: from the top end down (existing files are skipped)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    saved = gpufnp.install()
    spec = importlib.util.spec_from_file_location("est_gpu", EST)
    mod = importlib.util.module_from_spec(spec); sys.modules["est_gpu"] = mod; spec.loader.exec_module(mod)
    gpufnp.restore(saved)
    est = mod.Estimator()
    est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=os.path.dirname(EST)))
    lo, hi = (int(x) for x in a.indices.split("-"))
    todo = [i for i in range(lo, hi + 1) if (i - lo) % a.step == a.off]
    if a.rev:
        todo = todo[::-1]
    if a.split == "mini":
        t = np.load(TRUTH, allow_pickle=False)
    else:
        sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
        import torch
        torch.set_num_threads(1)
        loc = {}
        for fn in sorted(os.listdir(a.shards)):
            if fn.endswith(".npz") and fn.startswith("shard"):
                with np.load(os.path.join(a.shards, fn), allow_pickle=False) as z:
                    for r, i in enumerate(np.asarray(z["idx"], dtype=np.int64)):
                        loc[int(i)] = (fn, r)
        cache = (None, None)
    for k in todo:
        out = os.path.join(a.out, "%s_%0*d.npz" % (a.split, 4 if a.split == "mini" else 5, k))
        if os.path.exists(out):
            continue
        if a.split == "mini":
            s = int(t["mlp_seed"][k])
            Wn = weights_mini(s)
            eseed, truth = estimator_seed(s), t["all_layer_means"][k].astype(np.float64)
        else:
            if k not in loc:
                continue
            fn, row = loc[k]
            if cache[0] != fn:
                with np.load(os.path.join(a.shards, fn), allow_pickle=False) as z:
                    cache = (fn, np.asarray(z["moments"][:, 0], dtype=np.float32))
            truth = cache[1][row].astype(np.float64)
            g = torch.Generator(device="cpu").manual_seed(770000 + k)
            w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE
            Wn = [x.numpy() for x in w]
            s = eseed = 770000 + k
        mlp = types.SimpleNamespace(width=1024, depth=16, weights=[cp.asarray(x) for x in Wn], seed=eseed, name=str(k))
        del Wn
        mod.LT_DUMP.clear()
        t0 = time.time()
        pred = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41))).astype(np.float32)
        wall = time.time() - t0
        assert len(mod.LT_DUMP) == 16, len(mod.LT_DUMP)
        X = np.stack([cp.asnumpy(x).astype(np.float32) for _, x, _ in mod.LT_DUMP])
        Phi = np.stack([cp.asnumpy(p).astype(np.float32) for _, _, p in mod.LT_DUMP])
        mod.LT_DUMP.clear()
        assert X.shape[:2] == (16, 1024) and np.isfinite(X).all()
        raw = float(np.mean((pred[-1].astype(np.float64) - truth[-1]) ** 2))
        assert raw < 1e-6, "weight/truth mismatch at %d: %g" % (k, raw)
        tmp = out + ".tmp.npz"
        np.savez(tmp, X=X, Phi=Phi, pred=pred, idx=k, seed=s, flops=-1.0, wall=wall, gpu=1)
        os.replace(tmp, out)
        print("%s %d raw %.4e wall %.1fs" % (a.split, k, raw, wall), flush=True)


if __name__ == "__main__":
    main()
