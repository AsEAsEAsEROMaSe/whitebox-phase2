"""🔒 ПЛОМБА + КРИТИЧНИЙ ВИМІР: який НАСПРАВДІ шум істини датасету.

Якщо він ~0.038%, то зміряні 0.0381% двомоментного оракула [С115-е] є ПРИЛАДОМ,
а не властивістю класу — і вирок «клас не містить #1» стоїть на шумі.

  RMS(truthhi − датасет) = sqrt(шум_датасету² + шум_банку²), шум_банку відомий
"""
import numpy as np, pyarrow.parquet as pq, glob, os
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
print("%-20s %10s %10s %10s %10s" % ("мережа", "розб.%", "банк%", "⟹датасет%", "шар0 розб%"))
ds, bk = [], []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"]); k = NAMES[nm]
    T = np.asarray(tb["all_layer_means"][k].as_py(), dtype=np.float64).reshape(16, 1024)
    m = d["m"]; nz = d["noise"]
    n15 = np.sqrt((m[15]**2).sum())
    diff = np.sqrt(((m[15] - T[15])**2).sum())/n15
    bank = np.sqrt((nz[15]**2).sum())/n15
    dsn = np.sqrt(max(diff**2 - bank**2, 0.0))
    d0 = np.sqrt(((m[0] - T[0])**2).sum())/np.sqrt((m[0]**2).sum())
    ds.append(dsn); bk.append(bank)
    print("%-20s %10.5f %10.5f %10.5f %10.5f" % (nm, diff*100, bank*100, dsn*100, d0*100))
print("\nмедіана шуму ДАТАСЕТУ на фін. шарі: %.5f%%" % (np.median(ds)*100))
print("медіана шуму НАШОГО банку:          %.5f%%" % (np.median(bk)*100))
print("\n⟹ зміряні числа треба чистити: справжнє = sqrt(зміряне² − шум²)")
for lbl, v in (("наше замикання", 0.1269), ("двомоментний оракул", 0.0381), ("Puffi", 0.0170)):
    q = np.median(ds)*100
    print("   %-22s зміряне %.4f%%  ⟹ очищене %.4f%%" % (lbl, v, np.sqrt(max(v*v-q*q,0.0))))
