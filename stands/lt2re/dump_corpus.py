"""19.10.2026 white-box-16: дамп ознак коректора LT2 з ланцюга sb2 (Susupro #336537 MIT + наш FIJA + точні заощадження)
на корпусі keenanpepper/arc-whestbench-p2-d8b-corpus-14k. За зразком lab/corpus_lt/corpus_dump.py (MIT), з відмінностями:
естиматор estimator_sb2.py; ваги з torch-сіда (torch з notqa-venv, лише читання); шарди корпусу і ext (індекси 0..14047).
Запис: <out>/corpus_<idx:05d>.npz з X (16,1024,F), Phi, pred (корекція вимкнена, LT_OFF=1), flops, wall.
usage: dump_corpus.py <indices a-b|i,j> <shard_dir> <out> [--est path] [--step k --off r]"""
import argparse, os, sys, time
os.environ["LT_DUMP"] = "1"
os.environ["LT_OFF"] = "1"
os.environ.setdefault("V26_STRASSEN", "0")
sys.path.append("/home/trim/notqa-venv/lib/python3.12/site-packages")
import importlib.util
import numpy as np
import torch
torch.set_num_threads(1)
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import MLP, SetupContext

B = 2 ** 41
SCALE = (2.0 / 1024.0) ** 0.5


def load_est(path):
    spec = importlib.util.spec_from_file_location("est_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["est_mod"] = mod
    spec.loader.exec_module(mod)
    return mod.Estimator()


def parse_idx(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def shard_of(idx):
    """(file, row): original shards hold idx%16 at row idx//16; ext shards are located by their idx arrays."""
    if idx < 2048:
        return "shard%02d.npz" % (idx % 16), None
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indices"); ap.add_argument("shards"); ap.add_argument("out")
    ap.add_argument("--est", default="/home/trim/wb2lean/v39fija/estimator_sb2.py")
    ap.add_argument("--step", type=int, default=1); ap.add_argument("--off", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    est = load_est(a.est)
    mod = sys.modules["est_mod"]
    est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1",
                           submission_dir=os.path.dirname(os.path.abspath(a.est))))
    # truth index: idx -> (file, row), built from every shard file present (numeric arrays only, no pickle)
    loc = {}
    for fn in sorted(os.listdir(a.shards)):
        if fn.endswith(".npz") and (fn.startswith("shard")):
            with np.load(os.path.join(a.shards, fn), allow_pickle=False) as z:
                for r, i in enumerate(np.asarray(z["idx"], dtype=np.int64)):
                    loc[int(i)] = (fn, r)
    cache = (None, None)
    todo = [i for k, i in enumerate(parse_idx(a.indices)) if k % a.step == a.off]
    for idx in todo:
        out = os.path.join(a.out, "corpus_%05d.npz" % idx)
        if os.path.exists(out) or idx not in loc:
            continue
        fn, row = loc[idx]
        if cache[0] != fn:
            with np.load(os.path.join(a.shards, fn), allow_pickle=False) as z:
                cache = (fn, np.asarray(z["moments"][:, 0], dtype=np.float32))
        truth = cache[1][row]
        g = torch.Generator(device="cpu").manual_seed(770000 + idx)
        w = torch.randn(16, 1024, 1024, generator=g, dtype=torch.float32) * SCALE
        W = [fnp.array(x.numpy(), dtype=fnp.float32) for x in w]
        del w
        mlp = MLP(width=1024, depth=16, weights=W, seed=770000 + idx, name="corpus_%d" % idx)
        mod.LT_DUMP.clear()
        t0 = time.time()
        with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
            pred = np.asarray(fnp.asarray(est.predict(mlp, B)), dtype=np.float32)
        wall = time.time() - t0
        assert len(mod.LT_DUMP) == 16, len(mod.LT_DUMP)
        X = np.stack([np.asarray(x, dtype=np.float32) for _, x, _ in mod.LT_DUMP])
        Phi = np.stack([np.asarray(p, dtype=np.float32) for _, _, p in mod.LT_DUMP])
        assert X.shape[:2] == (16, 1024) and np.isfinite(X).all()
        raw = float(np.mean((pred[-1].astype(np.float64) - truth[-1]) ** 2))
        assert raw < 1e-6, "weight/truth mismatch at %d: %g" % (idx, raw)
        tmp = out + ".tmp.npz"
        np.savez(tmp, X=X, Phi=Phi, pred=pred, idx=idx, seed=770000 + idx, flops=float(ctx.flops_used), wall=wall)
        os.replace(tmp, out)
        print("corpus %d raw %.4e C/B %.4f wall %.1fs" % (idx, raw, ctx.flops_used / B, wall), flush=True)


if __name__ == "__main__":
    main()
