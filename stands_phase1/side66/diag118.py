"""С118 — ЧОМУ 329517 ДАВ MSE 2.0e-6: сирa vs косинусна похибка, оцінювач vs стенд, одна мережа."""
import sys, time, numpy as np
sys.path.insert(0, "/home/trim/fs16/lib/python3.12/site-packages"); sys.path.insert(0, "/mnt/win_d/llm/White-Box-P2")
import pyarrow.parquet as pq
NET = "regina-johnston"; OUT = "/home/trim/wblogs"
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]; idx = NAMES.index(NET)
W = np.asarray(tb["weights"][idx].as_py(), np.float64).reshape(16, 1024, 1024)
T = np.load("%s/covbank118_%s.npz" % (OUT, NET))["mu_16"]
from whestbench import MLP
import flopscope as flops
import estimator_p2_k3 as E
mlp = MLP(width=1024, depth=16, weights=[w for w in W], name=NET)
t0 = time.time(); est = E.Estimator(); out = np.asarray(est.predict(mlp, 2199023255552)); m = out[-1]
print("predict %.1f s, shape %s" % (time.time()-t0, out.shape))
def rep(tag, x):
    a = float(x@T)/float(x@x); raw = 100*np.sqrt(float((x-T)@(x-T))/float(T@T)); r = x*a-T
    cos = 100*np.sqrt(float(r@r)/float(T@T)); print("%-10s СИРА %.4f%% · косинусна %.4f%% · a* %.6f · MSE_raw %.3e" % (tag, raw, cos, a, float(((x-T)**2).mean())))
rep("estimator", m)
try:
    ms = np.load("%s/rec118h_m_%s.npz" % (OUT, NET))["m"]; rep("стенд", ms)
    d = m-ms; print("оцінювач−стенд: ‖Δ‖/‖T‖ %.4f%% · cos(Δ, m−T) %.3f" % (100*np.linalg.norm(d)/np.linalg.norm(T), float(d@(m-T))/np.linalg.norm(d)/np.linalg.norm(m-T)))
except Exception as ex: print("стенд нема:", ex)
# пошарово проти банку mu_l (l=1..16)
bk = np.load("%s/covbank118_%s.npz" % (OUT, NET))
for l in range(1, 17):
    if "mu_%d" % l in bk:
        Tl = bk["mu_%d" % l]; x = out[l-1]; a = float(x@Tl)/float(x@x); raw = 100*np.linalg.norm(x-Tl)/np.linalg.norm(Tl); cos = 100*np.linalg.norm(x*a-Tl)/np.linalg.norm(Tl)
        print("  ш%2d сира %.4f%% кос %.4f%% a* %.5f" % (l, raw, cos, a))
print("DONE_DIAG")
