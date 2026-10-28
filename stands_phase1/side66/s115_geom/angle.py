"""С115-р: КУТОВА РІЗНИЦЯ МІЖ СЕМПЛАМИ по глибині, фаза 2 (питання ЮЗЕРА).

Записане «попарний cos 0.32→0.94» — ФАЗА 1 (256×32). Для фази 2 у нас було лише
cos(ĥ, m̂) = 0.958 на ш15, а попарний я вивів МОДЕЛЛЮ (c² ≈ 0.918) — за законом
«число, здобуте всередині моделі, не є виміром» його цитувати не можна. Міряю.

🔒 КОНТРОЛЬ ОРТАНТУ: те саме на ПЕРЕМІШАНИХ компонентах кожного h (ті самі
маргіналі, та сама невід'ємність, структура знищена) ⟹ скільки з cos є просто
«усі координати додатні з великим середнім», а скільки — справжнім вирівнюванням.
🔒 ПЛОМБА: cos(z₁,z₂) на вході мусить дати 0 ± 1/√1024 = 0.031.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 попарний cos на ш15 = 0.88..0.94                         70%
  P2 контроль ортанту дасть >0.6 ⟹ БІЛЬШІСТЬ cos — це просто
     невід'ємність + велике середнє, а не вирівнювання         60%
  P3 надлишок (cos − контроль) РОСТЕ з глибиною                65%
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp
D, NW = 16, 1024
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
NN = int(sys.argv[2]) if len(sys.argv) > 2 else 4
T0 = time.time()
def log(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def pcos(X, rng):
    n = X.shape[0]
    a = rng.permutation(n)[:n//2]; b = rng.permutation(n)[:n//2]
    ok = a != b; a, b = a[ok], b[ok]
    na = cp.linalg.norm(X, axis=1)
    c = (X[a]*X[b]).sum(1)/cp.maximum(na[a]*na[b], 1e-30)
    return cp.asnumpy(c)

RES = []
for f in sorted(glob.glob("/home/trim/wblogs/w2_*.npy"))[:NN]:
    nm = os.path.basename(f)[3:-4]
    W = np.load(f); Wg = [cp.asarray(W[i]) for i in range(D)]
    rs = cp.random.RandomState(seed=1234)
    z = rs.standard_normal((NS, NW), dtype=cp.float32)
    rng = cp.random.RandomState(seed=99)
    row = {"name": nm, "z": float(np.median(np.abs(pcos(z, rng))))}
    cs, ct, cm = [], [], []
    x = z
    for i in range(D):
        x = cp.maximum(x @ Wg[i], 0.0)
        cs.append(pcos(x, rng))
        # 🔒 контроль ортанту: перемішати компоненти КОЖНОГО вектора незалежно
        idx = cp.argsort(rs.standard_normal((NS, NW), dtype=cp.float32), axis=1)
        xs = cp.take_along_axis(x, idx, axis=1)
        ct.append(pcos(xs, rng))
        mu = x.mean(0); mu = mu/cp.linalg.norm(mu)
        cm.append(float(cp.median((x @ mu)/cp.maximum(cp.linalg.norm(x, axis=1), 1e-30))))
    row["pair"] = [float(np.median(c)) for c in cs]
    row["p01"]  = [float(np.percentile(c, 1)) for c in cs]
    row["p99"]  = [float(np.percentile(c, 99)) for c in cs]
    row["orth"] = [float(np.median(c)) for c in ct]
    row["mean"] = cm
    RES.append(row); log("%-18s cos пар ш0 %.3f · ш8 %.3f · ш15 %.3f" % (nm, row["pair"][0], row["pair"][8], row["pair"][15]))
    del Wg; cp.get_default_memory_pool().free_all_blocks()

P = np.array([r["pair"] for r in RES]); O = np.array([r["orth"] for r in RES])
M = np.array([r["mean"] for r in RES]); L1 = np.array([r["p01"] for r in RES]); L9 = np.array([r["p99"] for r in RES])
print("\n🔒 ПЛОМБА cos(z₁,z₂) на вході: %.4f  [теорія 1/√1024 = 0.0312]" % np.median([r["z"] for r in RES]))
print("\nшар    cos(семпл,семпл)   кут°    🔒ортант   надлишок   cos(семпл,середнє)   смуга 1-99%%")
for i in range(D):
    p, o = np.median(P[:,i]), np.median(O[:,i])
    print("%3d      %.4f        %5.1f    %.4f    %+.4f       %.4f          %.3f"
          % (i, p, np.degrees(np.arccos(min(p,1.0))), o, p-o, np.median(M[:,i]),
             np.median(L9[:,i])-np.median(L1[:,i])))
log("ГОТОВО")
