"""С115-м: СПЕКТР ДИСПЕРСІЇ ПО СТЕПЕНЯХ — версія з КЕРОВАНИМ кутом.

Помилка v1: при випадкових парах t ~ 1/√d = 0.03, високі степені пригнічені як
t^k ⟹ витягувати їх множенням на d^{k/2} = множити шум на 1e13. Пломба впала
(Σg_k/E‖out‖² = 2e8) і одразу це показала.

v2: генеруємо пари з ЗАДАНИМ t = u·v на сітці [0..1], міряємо
    F(t) = Σ_a out_a(u)·out_a(v) і фітуємо F(t) = Σ_k a_k·t^k.
    (для сфери R_k(t) = t^k·(1 + O(k²/d)), при d=1024 поправка <8% для k<=9)
🔒 ПЛОМБА: t=1 дає F(1) = E‖out‖², і Σ_k a_k мусить дорівнювати йому.
🔒 ПЛОМБА-2: непарні a_k МУСЯТЬ бути помітні (антитетика їх бере) — інакше
    прилад не бачить асиметрії.

ПЕРЕДБАЧЕННЯ (ті самі):
  P1 непарні 50-60%  ·  P2 степінь 2 = 15-20%
  P3 степені >=6 < 3% ⟹ 5-дизайн дає VR >= x30            45%
  P4 степені >=6 > 10% ⟹ кубатура дає лише x5-10          35%
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NPER = int(sys.argv[1]) if len(sys.argv) > 1 else 1 << 14
NNET = int(sys.argv[2]) if len(sys.argv) > 2 else 6
KMAX = 9
D = 1024
CH = 1 << 12
TS = np.concatenate([np.linspace(-0.95, 0.95, 20), [0.98, 1.0]])   # СИМЕТРИЧНА ⟹ непарні розділяються
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)
res = []
for j in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][j].as_py()
    W = [cp.asarray(x) for x in np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16,1024,1024)]
    F = np.zeros(len(TS))
    for ti, t in enumerate(TS):
        s = 0.0; n = 0
        for off in range(0, NPER, CH):
            b = min(CH, NPER-off)
            rs = cp.random.RandomState(seed=71000+1000*j+31*ti+off//CH)
            u = rs.standard_normal((b, D), dtype=cp.float32); u /= cp.linalg.norm(u,axis=1,keepdims=True)
            w = rs.standard_normal((b, D), dtype=cp.float32)
            w -= (w*u).sum(1, keepdims=True)*u
            w /= cp.linalg.norm(w, axis=1, keepdims=True)
            v = (cp.float32(t)*u + cp.float32(np.sqrt(max(1-t*t,0.0)))*w)
            a, c = u, v
            for i in range(16):
                a = cp.maximum(a @ W[i], 0.0); c = cp.maximum(c @ W[i], 0.0)
            s += float((a.astype(cp.float64)*c.astype(cp.float64)).sum()); n += b
            del u, v, w, a, c
        F[ti] = s/n
    # 🚨 a_k — це КВАДРАТИ НОРМ ⟹ невід'ємні. Мономи t^k колінеарні, і звичайний
    #    МНК дає осцилюючі коефіцієнти ±1e6 при правильній сумі (спіймано v2a).
    V = np.vander(TS, KMAX+1, increasing=True)
    sc = np.sqrt((V*V).sum(0)); Vn = V/sc
    a = np.linalg.lstsq(Vn, F, rcond=None)[0]
    a = np.maximum(a, 0.0)
    for _ in range(20000):                      # проєкційний градієнт із невід'ємністю
        g = Vn.T @ (Vn @ a - F)
        a = np.maximum(a - 0.5*g/np.linalg.norm(Vn, 2)**2, 0.0)
    a = a/sc
    res.append((nm, a, F[-1]))
    log("%-20s  Σa_k %.6g  F(1) %.6g  збіг %.4f  нев'язка %.2e" % (nm, a.sum(), F[-1], a.sum()/F[-1], np.sqrt(((V@a-F)**2).mean())/abs(F[-1])))
    for x in W: del x
    cp.get_default_memory_pool().free_all_blocks()
A = np.stack([r[1] for r in res]); F1 = np.array([r[2] for r in res])
var = A[:,1:].sum(1); frac = A[:,1:]/var[:,None]
print("\n%-8s %10s" % ("степінь", "частка%"))
for k in range(1, KMAX+1): print("%-8d %10.3f" % (k, np.median(frac[:,k-1])*100))
print("\nнепарні %.1f%%  ·  степінь 2 %.1f%%  ·  парні>=4 %.1f%%" % (
    np.median(frac[:,0::2].sum(1))*100, np.median(frac[:,1])*100, np.median(frac[:,3::2].sum(1))*100))
print("\n%-20s %10s %10s" % ("правило точне до", "лишок%", "VR"))
for d in (1, 2, 3, 5, 7):
    r = np.median(frac[:, d:].sum(1)); print("%-20s %10.3f %10.2f" % ("степеня %d" % d, r*100, 1.0/max(r,1e-9)))
print("\n🔒 медіана Σa_k/F(1) = %.4f (мусить бути ≈1)" % np.median(A.sum(1)/F1))
log("ГОТОВО")
