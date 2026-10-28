"""ПОНЕЙРОННА ВАГА БЛЕНДУ З ВИМІРЯНОЇ ДИСПЕРСІЇ МК — МЕТОД, НЕ КОНСТАНТА.

МОТИВ (два ЗМІРЯНІ числа, не назва методу):
  [shift116] похибка ЗАМИКАННЯ рівномірна за |t| (18.9/29.3/19.7/11.0/21.2%)
  але дисперсія МК рівномірною бути НЕ МОЖЕ: при великому t ReLU майже
  лінійний ⟹ Var(x)/E[x]² ≈ 1/(N t²); при t<0 (половина нейронів!) ReLU
  майже завжди нуль ⟹ відносна дисперсія величезна.
⟹ `W_j = V_j/(V_j+E²_j)` мусить сильно варіювати ПО НЕЙРОНАХ, а ми вживаємо
   ОДНЕ число 0.820 на всі 1024.
`V_j` ВИМІРЮЄТЬСЯ з наших же семплів (`Var(x_j)/N`) — одна зайва редукція.

🚨 ЧОМУ ЦЕ НЕ `blend101` (помережна вага, ×0.9667): там механізм смерті —
   «avg_variance тягне ОБИДВІ компоненти однаково ⟹ у ВІДНОШЕННІ скорочується».
   Тут скорочення НЕМА: `V_j/E²_j ∝ 1/t_j²`, а `E²_j/m_j²` ≈ const.

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 N1 (60%) оракульна понейронна вага дає ≥×1.10 проти найкращої глобальної
 N2 (35%) ≥×1.30
 N3 (50%) реалістична `V_j/(V_j+κ m_j²)` бере ≥половину стелі
🔒 КОНТРОЛЬ: перемішані `V_j` між нейронами мусять обвалити результат
🔒 ПЛОМБА: при сталій `V_j` формула вироджується у глобальну W
"""
import os, sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
NPAIR = 6144
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def closure(W, g3=None, cm=0.0):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if g3 is not None and cm != 0.0 and 9 <= i <= 15:
            m = m - cm*s*(g3[i]/6.0)*(t*ph)
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

def mc_run(W, seed):
    """антитетичні вибілені семпли крізь мережу ⟹ xm (оцінка) і V_j (дисперсія/N)"""
    rs = cp.random.RandomState(seed=seed)
    z = rs.standard_normal((NPAIR, NW), dtype=cp.float32)
    cov = (z.T @ z)/NPAIR
    L = cp.linalg.cholesky(cov.astype(cp.float64))
    wh = cp.linalg.inv(L).T.astype(cp.float32)
    Wg = [cp.asarray(W[k].astype(cp.float32)) for k in range(D)]
    y = (z @ wh) @ Wg[0]
    x = cp.concatenate([cp.maximum(y, 0.0), cp.maximum(-y, 0.0)], axis=0)
    n2 = 2*NPAIR
    for k in range(1, D):
        p = x @ Wg[k]
        x = cp.maximum(p, 0.0)
    xm = cp.asnumpy(x.mean(0).astype(cp.float64))
    xv = cp.asnumpy(x.var(0).astype(cp.float64))/n2      # V_j = Var/N
    del Wg, z, x
    cp.get_default_memory_pool().free_all_blocks()
    return xm, xv

def mse(v, T): return float((v-T) @ (v-T))/len(T)
def cmse(v, T):
    a = float(v @ T)/float(v @ v); return mse(a*v, T)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
FS = sorted(glob.glob("/home/trim/wblogs/gamall_*.npz"))
lg("мереж %d · пар %d" % (len(FS), NPAIR))
R = {k: [] for k in ("glob", "orac", "real", "shuf", "mc", "cl")}
WSTAR = []
for f in FS:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    T = d["relu_15"]
    g3 = np.stack([d["g3_%d" % L] for L in range(D)])
    mdeep = closure(W, g3, 0.40)
    xm, V = mc_run(W, 3300 + NAMES.index(nm))
    a = float(xm @ mdeep)/float(mdeep @ mdeep)      # масштаб замикання під МК
    md = a*mdeep
    e = md - T; vv = xm - T
    # найкраща ГЛОБАЛЬНА вага
    ws = np.linspace(0, 1, 101)
    gm = [mse(xm + w*(md - xm), T) for w in ws]
    j = int(np.argmin(gm)); R["glob"].append(gm[j]); WSTAR.append(ws[j])
    # ОРАКУЛЬНА понейронна: W_j = V_j/(V_j+e_j²)
    wj = V/(V + e*e + 1e-300)
    R["orac"].append(mse(xm + wj*(md - xm), T))
    # РЕАЛІСТИЧНА: e_j² ≈ κ·md_j², κ скануємо
    best = None
    for kap in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
        wr = V/(V + kap*md*md + 1e-300)
        val = mse(xm + wr*(md - xm), T)
        if best is None or val < best[0]: best = (val, kap)
    R["real"].append(best[0])
    # 🔒 контроль: перемішані V_j
    rs = np.random.RandomState(11 + NAMES.index(nm))
    Vs = rs.permutation(V)
    wr = Vs/(Vs + best[1]*md*md + 1e-300)
    R["shuf"].append(mse(xm + wr*(md - xm), T))
    R["mc"].append(mse(xm, T)); R["cl"].append(mse(md, T))
    lg("%-18s глоб %.4e орак %.4e реал %.4e 🔒переміш %.4e | W*=%.2f κ=%.0e"
       % (nm, R["glob"][-1], R["orac"][-1], R["real"][-1], R["shuf"][-1], ws[j], best[1]))
print("\n%-28s %14s %10s" % ("режим", "MSE", "×глоб"))
g = float(np.mean(R["glob"]))
for k, nm2 in (("mc","чиста МК (V)"),("cl","чисте замикання (E²)"),
               ("glob","найкраща ГЛОБАЛЬНА W"),("orac","🔒 ОРАКУЛЬНА понейронна"),
               ("real","РЕАЛІСТИЧНА V/(V+κm²)"),("shuf","🔒 перемішані V_j")):
    v = float(np.mean(R[k])); print("%-28s %14.5e %10.4f" % (nm2, v, g/v))
print("\nсередня найкраща глобальна W = %.3f  (у сабміті 0.820)" % float(np.mean(WSTAR)))
lg("ГОТОВО")
