"""22.10.2026 a0: яку частку залишку c4t − g4 (діагональ κ4 передактивацій, ланцюг s32refire проти МК) дає
похибка НАРОДЖЕННЯ K22 (рядкові суми (2,2)-зрізу κ4(h)) і K4v (діагональ κ4(h)), перенесена анзацом W² ряд. сум.
ПЕРЕДБАЧЕННЯ: K22 бере 20…50% залишку на глибоких шарах; шанс > 50% 25%.
МК: 2^22 семплів, центрування істинними середніми mini_truth; шум оцінено половинами.
usage: t_k22.py <net> <out.npz>"""
import os, sys, types, importlib.util, numpy as np
from whestbench import SetupContext
sys.path.insert(0, "/home/trim/wb2lean/lt2re"); import gpufnp; cp = gpufnp.cp
sys.path.insert(0, "/home/trim/wb2lean/lt2alg"); from gpu_dump_alg import weights_mini, estimator_seed, TRUTH
k, out = int(sys.argv[1]), sys.argv[2]
os.environ.update(LT_OFF="1", X_D22_LOG="1", X_D22_B="0")
saved = gpufnp.install()
spec = importlib.util.spec_from_file_location("e_x", "/home/trim/wb2lean/s32lt/est_d22.py"); m = importlib.util.module_from_spec(spec); sys.modules["e_x"] = m; spec.loader.exec_module(m)
gpufnp.restore(saved)
e = m.Estimator(); e.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="1", submission_dir="/home/trim/wb2lean/s32lt"))
t = np.load(TRUTH); s = int(t["mlp_seed"][k]); Wn = weights_mini(s); mu = t["all_layer_means"][k].astype(np.float64)
Wg = [cp.asarray(x) for x in Wn]
m.D22_LOG.clear()
pred = cp.asnumpy(cp.asarray(e.predict(types.SimpleNamespace(width=1024, depth=16, weights=Wg, seed=estimator_seed(s), name=str(k)), 2 ** 41)))
R = {}
for ent in m.D22_LOG:
    if isinstance(ent[0], str):
        R["%s_%d_a" % (ent[0], ent[1])] = cp.asnumpy(cp.asarray(ent[2])); R["%s_%d_b" % (ent[0], ent[1])] = cp.asnumpy(cp.asarray(ent[3]))
del e; cp.get_default_memory_pool().free_all_blocks()
n, B, NB = 1024, 32768, 128          # 2^22
acc = {}
def add(key, v):
    acc[key] = acc.get(key, 0) + v
for b in range(NB):
    hh = cp.random.standard_normal((B, n), dtype=cp.float32); half = b % 2
    for li in range(16):
        p = hh @ Wg[li]
        pm = (cp.asarray(mu[li - 1], dtype=cp.float32) @ Wg[li]) if li > 0 else cp.zeros(n, cp.float32)
        pt = (p - pm).astype(cp.float64)
        add(("p2", li, half), (pt ** 2).sum(0)); add(("p4", li, half), (pt ** 4).sum(0))
        hh = cp.maximum(p, 0)
        ht = hh - cp.asarray(mu[li], dtype=cp.float32)
        h2 = ht * ht
        add(("h2", li, half), h2.sum(0, dtype=cp.float64)); add(("h4", li, half), (h2 * h2).sum(0, dtype=cp.float64))
        add(("C", li, half), (ht.T @ ht).astype(cp.float64))
        add(("R", li, half), (h2.T @ h2.sum(1)).astype(cp.float64))
N2 = B * NB / 2
T = {}
for half in (0, 1):
    for li in range(16):
        E2 = acc[("h2", li, half)] / N2; E4 = acc[("h4", li, half)] / N2; C = acc[("C", li, half)] / N2; Rr = acc[("R", li, half)] / N2
        k22 = (Rr - E4) - E2 * (E2.sum() - E2) - 2 * ((C * C).sum(1) - cp.diag(C) ** 2)
        T["k22t_%d_%d" % (li, half)] = cp.asnumpy(k22); T["k4t_%d_%d" % (li, half)] = cp.asnumpy(E4 - 3 * E2 ** 2)
        P2 = acc[("p2", li, half)] / N2; P4 = acc[("p4", li, half)] / N2
        T["c4t_%d_%d" % (li, half)] = cp.asnumpy(P4 - 3 * P2 ** 2); T["vart_%d_%d" % (li, half)] = cp.asnumpy(P2)
np.savez(out, **R, **T, W=np.stack(Wn).astype(np.float32))
print("done", k, flush=True)
