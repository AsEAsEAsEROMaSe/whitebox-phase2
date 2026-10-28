"""С118 — виправлений estimator_p2_k3 на всіх 24 мережах dev24: сира/косинусна похибка, a*, ефект α (LOO по половинах)."""
import sys, os, time, numpy as np
sys.path.insert(0, "/mnt/win_d/llm/White-Box-P2"); sys.path.insert(0, "/mnt/win_d/llm/White-Box-P2/side66")
import pyarrow.parquet as pq, importlib
from whestbench import MLP
E = importlib.import_module(os.environ.get("EST", "estimator_p2_k3"))
OUT = "/home/trim/wblogs"; TAG = os.environ.get("TAG", "k3fix")
tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
rows = []
for i in range(tb.num_rows):
    nm = tb["mlp_name"][i].as_py(); ck = "%s/diag118all_%s_%s.npz" % (OUT, TAG, nm)
    if os.path.exists(ck):
        d = np.load(ck); m, T, av = d["m"], d["T"], float(d["av"])
    else:
        W = np.asarray(tb["weights"][i].as_py(), np.float64).reshape(16, 1024, 1024)
        T = np.asarray(tb["final_means"][i].as_py(), np.float64); av = float(tb["avg_variance"][i].as_py())
        t0 = time.time(); out = np.asarray(E.Estimator().predict(MLP(width=1024, depth=16, weights=[w for w in W], name=nm), 2199023255552)); m = out[-1]
        np.savez(ck + ".tmp.npz", m=m, T=T, av=av); os.replace(ck + ".tmp.npz", ck)
        print("[%s] %.0f s" % (nm, time.time()-t0), flush=True)
    a = float(m@T)/float(m@m); raw = float(((m-T)**2).mean()); cos = float(((m*a-T)**2).mean()); floor = av/4e5
    rows.append((nm, raw, cos, a, floor))
    print("%-20s MSE сира %.3e · кос %.3e · a* %.6f · підлога %.2e" % (nm, raw, cos, a, floor), flush=True)
R = np.array([r[1:] for r in rows]); print("СЕРЕДНЄ: сира %.4e · косинусна %.4e · підлога %.3e · a* mean %.6f sd %.6f" % (R[:,0].mean(), R[:,1].mean(), R[:,3].mean(), R[:,2].mean(), R[:,2].std()))
# α: спільний масштаб, підбір на половині A, вирок на B і навпаки
ms = [np.load("%s/diag118all_%s_%s.npz" % (OUT, TAG, r[0])) for r in rows]
def mse_with(alpha, idx): return np.mean([(((1+alpha)*ms[i]["m"] - ms[i]["T"])**2).mean() for i in idx])
def fit(idx):
    num = sum(float(ms[i]["m"]@ms[i]["T"]) for i in idx); den = sum(float(ms[i]["m"]@ms[i]["m"]) for i in idx); return num/den - 1
n = len(rows); A = list(range(0, n, 2)); B = list(range(1, n, 2)); aA, aB, aAll = fit(A), fit(B), fit(range(n))
print("α: A %.5f B %.5f all %.5f · крос A→B %.4e (без α %.4e) · B→A %.4e (без α %.4e) · in-sample all %.4e" % (
    aA, aB, aAll, mse_with(aA, B), mse_with(0, B), mse_with(aB, A), mse_with(0, A), mse_with(aAll, range(n))))
print("DONE_ALL")
