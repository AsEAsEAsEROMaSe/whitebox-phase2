import gc, time, sys, runpy
# лічильник пауз gc під час whest run (локальний прилад, не сабміт)
T = {"n": 0, "t": 0.0, "t0": 0.0, "gen2": 0}
def cb(phase, info):
    if phase == "start": T["t0"] = time.perf_counter()
    else:
        T["n"] += 1; T["t"] += time.perf_counter() - T["t0"]
        if info.get("generation") == 2: T["gen2"] += 1
gc.callbacks.append(cb)
import atexit
atexit.register(lambda: sys.stderr.write("GCPROBE колекцій %d (gen2 %d), сумарно %.3f с\n" % (T["n"], T["gen2"], T["t"])))
sys.argv = ["whest"] + sys.argv[1:]
from whestbench.cli import main
sys.exit(main())
