"""С116: `_A_SHRINK` І `_ALPHA` — СПІЛЬНИЙ СКАН, ЗНАКОВИЙ ТЕСТ, ДРУГИЙ СІД.

[xnet3] σ²між = Var(a_i) − σ²шум = −4.5e-07 ВІД'ЄМНЕ ⟹ оптимальний масштаб
ОДНАКОВИЙ для всіх мереж, розкид `a_i` — чистий шум МК ⟹ вшите `_A_SHRINK = 0.5`
завелике; скан дав 0.25 і ×1.010. Але бракувало ДВОХ речей, і без них не сабмітити:
  (а) знакового тесту по мережах
  (б) СПІЛЬНОГО скану з `_ALPHA` — при S→0 уся вага масштабу лягає на `_ALPHA`,
      а константи МІЖ КОНФІГУРАЦІЯМИ НЕ ПЕРЕНОСЯТЬСЯ (закон С100, два хибні прогони)

🔑 ПІДЛОГА ПРИЛАДУ ВІДНІМАЄТЬСЯ: MSE_справж = MSE_dev24 − avg_variance_i/400000.
   [С116] це 15.4% нашого MSE і воно СТИСКАЄ всі відношення.
🔑 ЗАКОН NEXT_SESSION: ефекти на `a` (статистику) перевіряти на ДРУГІЙ вибірці.
   `_ALPHA`/`_A_SHRINK` не міняють `z` побітово ⟹ парне порівняння чесне.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 спільний оптимум має S < 0.35 і ALPHA негативніше за −0.00134       70%
  P2 виграш після спільного скану ×1.01…1.03                            60%
  P3 знаковий тест ≥17/24                                               55%
  P4 оптимум на ДРУГОМУ сіді зсунеться менше ніж на пів кроку сітки     65%
ГЕЙТ: знаковий <15/24 АБО оптимум скаче між сідами ⟹ НЕ сабмітити.
"""
import sys, time, numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
INV = 0.3989422804014327
NW, D = 1024, 16
NPAIR = 3072; NREP = 4
WB = 0.820
A0, S0 = -0.00134, 0.5           # чинні константи сабміту
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def closure(W):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(D))
        if i == D-1: return m
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

def sample(W, z):
    cov = (z.T @ z)/z.shape[0]
    wz = np.linalg.inv(np.linalg.cholesky(cov)).T
    y = (z @ wz) @ W[0]
    x = np.concatenate([np.maximum(y,0.0), np.maximum(-y,0.0)],0).astype(np.float32)
    sg0 = np.sqrt(np.maximum((W[0]*W[0]).sum(0),1e-30))
    v = x.mean(0).astype(np.float64) - sg0*INV
    for i in range(1, D):
        p = x @ W[i].astype(np.float32)
        v = (v @ W[i]) * (p > 0.0).mean(0).astype(np.float64)
        x = np.maximum(p, 0.0)
    return x.astype(np.float64).mean(0) - v

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NN = tb.num_rows
REC = []
for j in range(NN):
    W = np.asarray(tb["weights"][j].as_py(), dtype=np.float64).reshape(D,NW,NW)
    tr = np.asarray(tb["all_layer_means"][j].as_py(), dtype=np.float64).reshape(D,NW)[15]
    av = tb["avg_variance"][j].as_py()
    md = closure(W); den = max((md*md).sum(), 1e-300)
    for sd_ in (0, 1):                                   # ДВА СІДИ
        for r in range(NREP):
            rng = np.random.default_rng(60000 + 100000*sd_ + 100*j + r)
            z = rng.standard_normal((NPAIR, NW)).astype(np.float32)
            ours = sample(W, z)
            REC.append(dict(j=j, sd=sd_, ours=ours, md=md, tr=tr,
                            ai=(ours @ md)/den, floor=av/400000.0))
    lg("%2d  ai=%.5f" % (j, REC[-1]["ai"]))
lg("зібрано %d записів" % len(REC))

def mse(q, A, S):
    a = (1.0-S)*(1.0+A) + S*q["ai"]
    o = q["ours"] + WB*(a*q["md"] - q["ours"])
    return float(np.mean((o - q["tr"])**2))

def agg(A, S, sd_=None, corr=True):
    v = [mse(q, A, S) - (q["floor"] if corr else 0.0)
         for q in REC if sd_ is None or q["sd"] == sd_]
    return float(np.mean(v))

ALPHAS = [-0.0026, -0.0022, -0.0018, -0.00163, -0.00134, -0.0010, -0.0006, 0.0]
SS     = [0.0, 0.10, 0.20, 0.25, 0.35, 0.50]
base = agg(A0, S0)
raw0 = agg(A0, S0, corr=False)
print("\n🔒 ПІДЛОГА: сира MSE %.6e · після віднімання %.6e ⟹ підлога %.1f%%"
      % (raw0, base, 100*(raw0-base)/raw0))
print("\n── СПІЛЬНИЙ СКАН (MSE з відн. підлогою, ×база) ──")
print("  ALPHA \\ S " + "".join("%9.2f" % s for s in SS))
best = None
for A in ALPHAS:
    row = []
    for S in SS:
        m_ = agg(A, S); row.append(base/m_)
        if best is None or m_ < best[2]: best = (A, S, m_)
    star = " ← чинне" if abs(A-A0) < 1e-9 else ""
    print("  %8.5f " % A + "".join("%9.4f" % r for r in row) + star)
print("  ⟹ оптимум ALPHA=%.5f  S=%.2f  ×%.4f" % (best[0], best[1], base/best[2]))

print("\n── 🔒 ПЛОМБА ДРУГОГО СІДА (оптимум мусить стояти) ──")
for sd_ in (0, 1):
    b2 = agg(A0, S0, sd_); bb = None
    for A in ALPHAS:
        for S in SS:
            m_ = agg(A, S, sd_)
            if bb is None or m_ < bb[2]: bb = (A, S, m_)
    print("  сід %d: оптимум ALPHA=%.5f S=%.2f ×%.4f · у ТОЧЦІ спільного оптимуму ×%.4f"
          % (sd_, bb[0], bb[1], b2/bb[2], b2/agg(best[0], best[1], sd_)))

print("\n── ЗНАКОВИЙ ТЕСТ ПО МЕРЕЖАХ (усереднено по репліках і сідах) ──")
for A, S, lab in [(best[0], best[1], "спільний оптимум"),
                  (A0, 0.25, "лише S=0.25 (xnet3)"),
                  (-0.00163, 0.0, "S=0 + ALPHA=−0.00163")]:
    w = 0; deltas = []
    for j in range(NN):
        qs = [q for q in REC if q["j"] == j]
        b_ = np.mean([mse(q, A0, S0) - q["floor"] for q in qs])
        n_ = np.mean([mse(q, A,  S ) - q["floor"] for q in qs])
        w += (n_ < b_); deltas.append(b_/n_)
    print("  %-24s ALPHA=%8.5f S=%.2f · %2d/%d краще · сер ×%.4f · медіана ×%.4f"
          % (lab, A, S, w, NN, np.mean(deltas), np.median(deltas)))

print("\n🎯 ГЕЙТ: знаковий ≥15/24 І оптимум стоїть на обох сідах")
