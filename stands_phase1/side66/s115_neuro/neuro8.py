"""ЧОМУ СПРАВЖНІЙ ВИБІРКОВИЙ ШУМ ГІРШИЙ ЗА СИНТЕТИЧНИЙ ТІЄЇ САМОЇ ВЕЛИЧИНИ."""
import sys, numpy as np, pyarrow.parquet as pq, glob
sys.path.insert(0, "/home/trim/gpupkg"); import cupy as cp
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
NPAIR = 3952; NS = 2.0*NPAIR
exec(open("/home/trim/wbstand/neuro7.py").read().split("def run(")[0].split("NPAIR = 3952")[1])

tb = pq.read_table(PQ)
print(f"{'ш':>2} {'РЕАЛЬНА похибка':>16} {'МОДЕЛЬ √(dd/N)':>15} {'×':>6} | {'проєкція на m':>14} {'ортогональна':>13} | {'r між шар':>10}")
prev = None
for j in range(4):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(16, 1024, 1024)
    T = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(16, 1024)
    SM = sample_means(W, 555+j)
    MS, SS, TS, VV = closure(W)
    if j: print("  ---")
    pe = None
    for i in [1, 4, 8, 12, 15]:
        e = SM[i] - T[i]; nt = np.sqrt((T[i]*T[i]).sum())
        real = np.sqrt((e*e).sum())/nt
        mod = np.sqrt(np.maximum(VV[i], 0).sum()/NS)/nt
        g = T[i]/np.sqrt((T[i]*T[i]).sum()); par = abs(float(e@g))/nt
        orth = np.sqrt(max(real*real - par*par, 0))
        r = "" if pe is None else "%10.3f" % float(np.corrcoef(e/nt, pe)[0, 1])
        pe = e/nt
        print(f"{i:2d} {real*100:15.4f}% {mod*100:14.4f}% {real/mod:6.2f} | {par*100:13.4f}% {orth*100:12.4f}% |{r}")
