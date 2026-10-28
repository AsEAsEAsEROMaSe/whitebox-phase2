"""С115-м (ідея ЮЗЕРА): ЧИ ІСНУЄ ВХІД, ЩО ДАЄ НА ГЛИБИНІ ВАГОВИЙ НАПРЯМОК?

Питання про ДОСЯЖНІСТЬ: наскільки далеко можна відхилити стан шару L у бік
довільного вагового вектора `w` (стовпця W_{L+1}), коли всі ReLU і всі
попередні ваги мусять це пропустити.

  типовий семпл        cos(h_L, w) ~ 0 ± 1/sqrt(n)
  МАКСИМАЛЬНО ДОСЯЖНЕ  ?                     ← міряємо
  стеля ОРТАНТУ        ‖w₊‖/‖w‖ ~ 0.707      ← якби єдиним обмеженням було h>=0

Проміжок між досяжним і стелею ортанту показує, наскільки мережа ЖОРСТКІША
за просту невід'ємність. Це край досяжної множини — та сама форма, брак якої
дає ×5.2 у двомоментному поданні [С115-л].

МЕТОД: градієнтний підйом по z (мережа кусково-лінійна ⟹ підйом чесний):
  вперед із масками D_i;  g = (ŵ − cos·ĥ)/‖h‖;  назад g ← (g⊙D_i) W_iᵀ
🔒 КОНТРОЛЬ-1: максимум по великій ВИПАДКОВІЙ вибірці (скільки дає везіння)
🔒 КОНТРОЛЬ-2: та сама процедура на ЧУЖИЙ w (з іншої мережі) — мусить дати менше
🔒 КОНТРОЛЬ-3: ціль = m̂ шару L (напрямок хмари) — мусить дати ~0.99

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 досяжне 0.3…0.6 (мережа гнучка, але не як вільний ортант)   45%
  P2 досяжне < 0.15 (мережа ЖОРСТКА, конус вузький)              30%
  P3 досяжне > 0.65 (майже вільний ортант)                       25%
  P4 глибші шари ЖОРСТКІШІ за мілкі                              65%
"""
import os, sys, time, numpy as np
sys.path.insert(0, "/home/trim/gpupkg")
import cupy as cp
import pyarrow.parquet as pq
PQ = "/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet"
NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NTGT = int(sys.argv[2]) if len(sys.argv) > 2 else 24
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
LAYERS = [2, 4, 8, 12]
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
tb = pq.read_table(PQ)

def fwd(W, z, L):
    x = z; masks = []
    for i in range(L+1):
        p = x @ W[i]; d = (p > 0)
        masks.append(d); x = cp.maximum(p, 0.0)
    return x, masks

def ascend(W, tgt, L, steps, seed):
    """підйом cos(h_L(z), tgt) по z. tgt нормований. Повертає найкращий cos."""
    n = W[0].shape[0]; B = tgt.shape[0]
    rs = cp.random.RandomState(seed=seed)
    z = rs.standard_normal((B, n), dtype=cp.float32)
    z /= cp.linalg.norm(z, axis=1, keepdims=True)
    best = cp.zeros(B, dtype=cp.float32); lr = cp.float32(0.5)
    for it in range(steps):
        x, masks = fwd(W, z, L)
        nx = cp.maximum(cp.linalg.norm(x, axis=1, keepdims=True), 1e-20)
        c = (x*tgt).sum(1, keepdims=True)/nx
        best = cp.maximum(best, c[:, 0])
        g = (tgt - (c/nx)*x)/nx                       # d cos / d h
        for i in range(L, -1, -1):
            g = (g*masks[i]) @ W[i].T
        z = z + lr*g/cp.maximum(cp.linalg.norm(g, axis=1, keepdims=True), 1e-20)
        z /= cp.linalg.norm(z, axis=1, keepdims=True)
        if it == steps//2: lr = cp.float32(0.15)
    return best

res = {}
for j in range(min(NNET, tb.num_rows)):
    nm = tb["mlp_name"][j].as_py()
    W = [cp.asarray(x) for x in np.asarray(tb["weights"][j].as_py(), dtype=np.float32).reshape(16,1024,1024)]
    Wo = [cp.asarray(x) for x in np.asarray(tb["weights"][(j+1) % tb.num_rows].as_py(), dtype=np.float32).reshape(16,1024,1024)]
    for L in LAYERS:
        tgt = W[L+1][:, :NTGT].T.copy()
        tgt = tgt/cp.linalg.norm(tgt, axis=1, keepdims=True)
        ceil_ = cp.linalg.norm(cp.maximum(tgt,0.0), axis=1)          # ‖w₊‖/‖w‖
        # 🔒 контроль-1: максимум по випадковій вибірці
        rs = cp.random.RandomState(seed=1234+j)
        zr = rs.standard_normal((4096, 1024), dtype=cp.float32)
        zr /= cp.linalg.norm(zr, axis=1, keepdims=True)
        xr, _ = fwd(W, zr, L)
        xr = xr/cp.maximum(cp.linalg.norm(xr, axis=1, keepdims=True), 1e-20)
        rnd = (xr @ tgt.T).max(0)
        got = ascend(W, tgt, L, STEPS, 4242+j)
        # 🔒 контроль-2: ЧУЖИЙ w
        tf = Wo[L+1][:, :NTGT].T.copy(); tf = tf/cp.linalg.norm(tf, axis=1, keepdims=True)
        gotf = ascend(W, tf, L, STEPS, 555+j)
        res.setdefault(L, []).append((float(cp.median(got)), float(cp.median(rnd)),
                                      float(cp.median(ceil_)), float(cp.median(gotf))))
    log(nm)
    for w in W: del w
    for w in Wo: del w
    cp.get_default_memory_pool().free_all_blocks()
print("\n%-5s %10s %10s %10s %10s %10s" % ("шар L","досяжне","випадк.max","стеля орт.","🔒чужий w","досяг.%стелі"))
for L in LAYERS:
    a = np.array(res[L])
    g, r, c, f = np.median(a,0)
    print("%-5d %10.4f %10.4f %10.4f %10.4f %10.1f" % (L, g, r, c, f, 100*g/c))
log("ГОТОВО")
