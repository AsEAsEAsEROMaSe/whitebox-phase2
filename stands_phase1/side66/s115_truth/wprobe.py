"""С115-м (ідея ЮЗЕРА): ЯК ЖИВЕ В ГЛИБИНУ ВЕКТОР, УЗЯТИЙ ІЗ РОЗРІЗАННЯ ШАРУ?

Дві постановки:
  A  стовпець W_L подаємо як ВХІД (шар 0) і женемо крізь мережу
  B  стовпець W_8 ВКИДАЄМО як стан НА шарі 8 (він максимально збуджує нейрон
     наступного шару ⟹ детермінований зонд структури)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  A1 стовпці W_8 як вхід НЕ відрізняються від випадкових (He-init незалежний) 85%
  B1 вкинутий на ш8 стовпець розмивається до тієї самої хмари (cos~0.95)      70%
  B2 він тримає особливість до виходу (cos помітно інший)                     30%
🔒 КОНТРОЛЬ: базовий cos двох незалежних векторів у ДОДАТНОМУ ортанті = 0.318
   (не 0!) — усе порівняння мусить це віднімати [С115-і].
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 4
K = 256
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)

def cosrow(X, m):
    mn = m/cp.linalg.norm(m)
    return (X @ mn)/cp.maximum(cp.linalg.norm(X, axis=1), 1e-30)

rows = {}
for j in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][j].as_py()
    W = [cp.asarray(x) for x in np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16,1024,1024)]
    d = np.load("/home/trim/wblogs/truthhi_%02d.npz" % j, allow_pickle=True) \
        if os.path.exists("/home/trim/wblogs/truthhi_%02d.npz" % j) else None
    M = [cp.asarray(d["m"][i]) for i in range(16)] if d is not None else None
    rs = cp.random.RandomState(seed=99000+j)
    probes = {}
    g = rs.standard_normal((K,1024), dtype=cp.float32); probes["випадковий"] = g/cp.linalg.norm(g,axis=1,keepdims=True)
    for L in (0, 8, 15):
        c = W[L][:, :K].T.copy()                     # стовпці шару L як напрямки
        probes["W%d стовпці" % L] = c/cp.linalg.norm(c, axis=1, keepdims=True)
    for tag, X in list(probes.items()):
        x = X.copy(); cs = []
        for i in range(16):
            x = cp.maximum(x @ W[i], 0.0)
            if M is not None: cs.append(float(cp.median(cosrow(x, M[i]))))
        rows.setdefault(("A", tag), []).append(cs)
    # B: вкидання НА шар 8
    for tag in ("випадковий", "W8 стовпці"):
        x = probes[tag].copy(); cs = [np.nan]*8
        for i in range(8, 16):
            x = cp.maximum(x @ W[i], 0.0)
            if M is not None: cs.append(float(cp.median(cosrow(x, M[i]))))
        rows.setdefault(("B@8", tag), []).append(cs)
    log(nm)
    for w in W: del w
    cp.get_default_memory_pool().free_all_blocks()
print("\ncos(ĥ, m̂) по глибині  🔒 база двох незалежних у додатному ортанті = 0.318")
print("%-6s %-14s %s" % ("режим", "зонд", "".join("%7s" % ("ш%d"%i) for i in (0,1,2,4,8,11,13,15))))
for k, v in rows.items():
    a = np.nanmedian(np.array(v), 0)
    print("%-6s %-14s %s" % (k[0], k[1], "".join("%7.3f" % a[i] if not np.isnan(a[i]) else "%7s"%"-" for i in (0,1,2,4,8,11,13,15))))
log("ГОТОВО")
