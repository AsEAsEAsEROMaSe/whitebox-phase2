"""lineprof (22.10.2026, white-box-b9): ціна fnp-викликів сабміту 337568 по рядках estimator.py (GPU-шим).
Модель ціни (як flopscope): поелементні = numel виходу, copyto = numel dst, take = numel виходу, matmul = 2mnk·batch.
Оператори +,-,* на масивах (не fnp.*) НЕ бачить. usage: lineprof.py <mini idx>"""
import os, sys, types, importlib.util, collections, math
import numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re")
import gpufnp
cp = gpufnp.cp
EST = "/home/trim/wb2sub/s32f_mid/unpack/estimator.py"
AGG = collections.defaultdict(lambda: [0.0, 0])


def site():
    f = sys._getframe(2)
    while f is not None and not f.f_code.co_filename.endswith("estimator.py"):
        f = f.f_back
    return f.f_lineno if f is not None else -1


def wrap_elem(name):
    fn = getattr(cp, name)

    def w(*a, **k):
        r = fn(*a, **k)
        o = k.get("out", r)
        AGG[(name, site())][0] += float(getattr(o, "size", 1)); AGG[(name, site())][1] += 1
        return r
    return w


def w_copyto(dst, src, **k):
    cp.copyto(dst, src, **k)
    AGG[("copyto", site())][0] += float(dst.size); AGG[("copyto", site())][1] += 1


def w_matmul(a, b, out=None, **k):
    r = cp.matmul(a, b, out=out, **k)
    m, kk = a.shape[-2] if a.ndim > 1 else 1, a.shape[-1]
    n = b.shape[-1] if b.ndim > 1 else 1
    bt = r.size / max(1, (m * n))
    AGG[("matmul", site())][0] += 2.0 * m * kk * n * bt; AGG[("matmul", site())][1] += 1
    return r


saved = gpufnp.install()
for nm in ("add", "subtract", "multiply", "take", "true_divide", "divide", "maximum", "where", "concatenate", "column_stack", "stack", "sum"):
    setattr(gpufnp.fnp, nm, wrap_elem(nm))
gpufnp.fnp.copyto = w_copyto
gpufnp.fnp.matmul = w_matmul
spec = importlib.util.spec_from_file_location("est_gpu", EST)
mod = importlib.util.module_from_spec(spec); sys.modules["est_gpu"] = mod; spec.loader.exec_module(mod)
gpufnp.restore(saved)
est = mod.Estimator()
est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir=os.path.dirname(EST)))
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz", allow_pickle=False)
SCALE = (2.0 / 1024.0) ** 0.5
k = int(sys.argv[1]); s = int(t["mlp_seed"][k])
wss, _, ess = np.random.SeedSequence(s).spawn(3)
rng = np.random.default_rng(wss)
Wn = [(rng.standard_normal((1024, 1024)) * SCALE).astype(np.float32) for _ in range(16)]
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[cp.asarray(x) for x in Wn], seed=int(ess.generate_state(1)[0]), name=str(k))
AGG.clear()
p = cp.asnumpy(cp.asarray(est.predict(mlp, 2 ** 41)))[-1].astype(np.float64)
print("raw %.4e" % float(np.mean((p - t["all_layer_means"][k][-1]) ** 2)))
tot = sum(v[0] for v in AGG.values())
byop = collections.defaultdict(float)
for (o, l), v in AGG.items(): byop[o] += v[0]
print("seen total %.4e" % tot, {o: "%.3e" % v for o, v in sorted(byop.items(), key=lambda x: -x[1])})
for (o, l), v in sorted(AGG.items(), key=lambda x: -x[1][0])[:60]:
    if o != "matmul" or True: print("%-12s line %5d  %.3e  calls %d" % (o, l, v[0], v[1]))
