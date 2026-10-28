"""prof_lines (19.10.2026, white-box-89): тарифіковані флопи естиматора за рядками коду (і за функціями).
usage: prof_lines.py <estimator.py> <mini idx> <out.json>
Кожен OpRecord приписується найглибшому кадру стеку, що лежить у файлі естиматора; плюс підсумок за op_name.
Без whest: SetupContext/MLP як у lab/lt_dump.py Susupro (MIT)."""
import sys, os, json, time, importlib.util, collections
sys.path.insert(0, "/home/trim/ext/fc/repo/tools")
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget as FB
from whestbench import MLP, SetupContext
from mlpdata import weights_from_seed, estimator_seed

B = 2 ** 41
est_path, idx, out = os.path.abspath(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
by_line = collections.Counter()
by_line_calls = collections.Counter()
by_op = collections.Counter()
by_core = collections.Counter()
by_layer = collections.Counter()
orig = FB.BudgetContext._append_op_record


def patched(self, rec):
    f = sys._getframe(1)
    key = None
    while f is not None:
        if f.f_code.co_filename == est_path:
            key = (f.f_lineno, f.f_code.co_name)
            break
        f = f.f_back
    by_line[key] += rec.flop_cost
    g = f
    while g is not None and g.f_code.co_name != "_predict_core":
        g = g.f_back
    by_core[g.f_lineno if g is not None else None] += rec.flop_cost
    by_layer[(g.f_locals.get('li') if g is not None else None, g.f_lineno if g is not None else None)] += rec.flop_cost
    by_line_calls[key] += 1
    by_op[rec.op_name] += rec.flop_cost
    return orig(self, rec)


FB.BudgetContext._append_op_record = patched
spec = importlib.util.spec_from_file_location("est_mod", est_path)
mod = importlib.util.module_from_spec(spec)
sys.modules["est_mod"] = mod
spec.loader.exec_module(mod)
est = mod.Estimator()
t = np.load("/home/trim/ext/fc/repo/data/mini_truth.npz")
est.setup(SetupContext(width=1024, depth=16, flop_budget=B, api_version="1",
                       submission_dir=os.path.dirname(est_path)))
W = [fnp.array(w, dtype=fnp.float32) for w in weights_from_seed(t["mlp_seed"][idx])]
mlp = MLP(width=1024, depth=16, weights=W, seed=estimator_seed(t["mlp_seed"][idx]), name=str(t["mlp_name"][idx]))
by_line.clear(); by_line_calls.clear(); by_op.clear(); by_core.clear(); by_layer.clear()
t0 = time.time()
with flops.BudgetContext(flop_budget=int(1e14), quiet=True) as ctx:
    pred = np.asarray(fnp.asarray(est.predict(mlp, B)), dtype=np.float64)
wall = time.time() - t0
truth = t["all_layer_means"][idx].astype(np.float64)
raw = float(np.mean((pred[-1] - truth[-1]) ** 2))
tot = float(ctx.flops_used)
rows = sorted(((v, k) for k, v in by_line.items()), key=lambda x: -x[0])
json.dump({"idx": idx, "raw": raw, "CB": tot / B, "wall": wall,
           "lines": [[k[0] if k else None, k[1] if k else None, v, by_line_calls[k]] for v, k in rows],
           "ops": by_op.most_common(), "core": [[k, v] for k, v in by_core.most_common()], "layer": [[k[0], k[1], v] for k, v in by_layer.items()]}, open(out, "w"))
print(f"idx {idx} raw {raw:.4e} C/B {tot / B:.4f} wall {wall:.0f}s")
for v, k in rows[:40]:
    print(f"{v / B:8.4f}  {v / tot * 100:5.1f}%  calls {by_line_calls[k]:6d}  {k}")
print("ops:", [(o, round(c / B, 4)) for o, c in by_op.most_common(12)])
print("core:")
for k, v in by_core.most_common(30):
    print(f"{v / B:8.4f}  {v / tot * 100:5.1f}%  line {k}")
