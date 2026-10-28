"""С115-н: 8-ОЗНАКОВА МОДЕЛЬ ЗМІЩЕННЯ ЗАМИКАННЯ У ФАЗІ 2. ЗАМІР ПІДСТАНОВКОЮ.

Аудит (С70/С75/С79): `b` пояснюється на R²_oos=0.792 вісьмома безрозмірними
ознаками; підстановка дає ×1.085 (дві пломби). ⚠️ Той вирок винесено при вазі
гілки 7.6% MSE; у фазі 2 гілка замикання важить 70-82% ⟹ важіль ×10.

🚨 ЗАКОН С70: `R²` НЕ характеризує залишок ⟹ вирок ЛИШЕ підстановкою.
🚨 «ЧУЖА β» ТУТ ПОРОЖНЯ ЗА ПОБУДОВОЮ — функція універсальна між мережами.
   Робочі контролі: (1) `shuf` — переставити ФОРМУ поправки між нейронами шару
   (константа лишається, бо вона однакова); (2) скан λ — свій оптимум мусить
   бути при λ>0, у контролю при λ→0.

РЕЖИМИ: const (1 ознака) · feat8 (8) · shuf (контроль) · orac (стеля, точне `b`
з половини A). Ціль `y = b/s`, підстановка `m ← m − λ·s·(F@β)` НА КОЖНОМУ шарі.
Вирок — косинусна похибка фінального `m` проти половини B.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 оракул дає ×3..×10                                          70%
  P2 feat8 LOO дає ×1.05..×1.4                                   55%
  P3 оптимум λ НИЖЧЕ 1.0 («поправка стиснена», 7-й раз)          75%
  P4 shuf має оптимум ближче до 0, ніж feat8                     70%
  P5 база 0.118..0.125%  [записане 0.1202%]                      80%
  P6 const бере >60% від feat8 (як 0.729 у фазі 1)               60%
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, sys, glob, time
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
import pyarrow.parquet as pq

NNET  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LAMS  = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
RIDGE = 1e-3
D, NW = 16, 1024
RNG   = np.random.default_rng(20260830)

def feats(t, Ph, ph, q):
    o = np.ones_like(t)
    return np.stack([o, t, Ph, ph, t*ph, t*t, q, q*q], 1)

def closure_corr(W, corr=None):
    m = np.zeros(NW); C = np.eye(NW); MS, FS, SS = [], [], []
    for i in range(D):
        w = W[i]
        cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0:
            m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        q = np.sqrt(np.maximum(dd, 0.0))/s
        F = feats(t, Ph, ph, q)
        if corr is not None:
            c = corr(i, F, s)
            if c is not None:
                m = m - c
                dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); FS.append(F); SS.append(s.copy())
        if i == D-1: break
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS, FS, SS

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(), 1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

def wls(F, y, ridge=RIDGE):
    G = F.T @ F
    dg = np.sqrt(np.maximum(np.diag(G), 1e-300))
    Gn = G/np.outer(dg, dg)
    return np.linalg.solve(Gn + ridge*np.eye(F.shape[1]), (F.T @ y)/dg)/dg

# ─────────── дані (ваги кешуємо на ext4: as_py() дуже повільний) ───────────
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
banks = sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:NNET]
NETS = []
for f in banks:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    if nm not in NAMES: continue
    cache = "/home/trim/wblogs/w2_%s.npy" % nm
    if os.path.exists(cache):
        W = np.load(cache).astype(np.float64)
    else:
        W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(D, NW, NW)
        np.save(cache, W); W = W.astype(np.float64)
    NETS.append(dict(name=nm, W=W, mA=np.asarray(d["m_half"])[:, 0, :],
                     mB=np.asarray(d["m_half"])[:, 1, :]))
    log("  %s" % nm)
NN = len(NETS); log("мереж: %d" % NN)

# ─────────── база, ознаки, цілі ───────────
BASE, YS, FSS, BT = [], [], [], []
for N in NETS:
    MS, FS, SS = closure_corr(N["W"])
    BASE.append(cerr(MS[15], N["mB"][15]))
    YS.append([(MS[i] - N["mA"][i])/SS[i] for i in range(D)])
    BT.append([MS[i] - N["mA"][i] for i in range(D)])
    FSS.append(FS)
    log("%-18s база %.4f%%   (шум цілі A vs B: %.4f%%)"
        % (N["name"], 100*BASE[-1], 100*cerr(N["mA"][15], N["mB"][15])))
B0 = float(np.median(BASE))
log("🔒 P5 БАЗА (медіана) %.4f%%   [записане 0.1202%%]" % (100*B0))

# ─────────── LOO-фіт по мережах, пошарово, два набори ───────────
BET = {}
for kind, cols in (("const", [0]), ("feat8", list(range(8)))):
    for j in range(NN):
        for i in range(D):
            F = np.concatenate([FSS[k][i][:, cols] for k in range(NN) if k != j], 0)
            y = np.concatenate([YS[k][i] for k in range(NN) if k != j], 0)
            BET[(kind, j, i)] = (cols, wls(F, y))
r2 = []
for i in range(D):
    F = np.concatenate([FSS[k][i] for k in range(NN)], 0)
    y = np.concatenate([YS[k][i] for k in range(NN)], 0)
    b = wls(F, y); r = y - F@b
    r2.append(1.0 - (r*r).sum()/max(((y-y.mean())**2).sum(), 1e-300))
log("R² in-sample (довідка, НЕ вирок): " + " ".join("%.3f" % v for v in r2))

PERM = [RNG.permutation(NW) for _ in range(D)]

def run(mode, lam, j):
    N = NETS[j]
    if mode == "orac":
        f = lambda i, F, s: lam*BT[j][i]
    elif mode == "shuf":
        def f(i, F, s, j=j):
            cols, b = BET[("feat8", j, i)]
            v = F[:, cols] @ b
            return lam*s*(v[0]*0.0 + v[PERM[i]])
    else:
        def f(i, F, s, j=j, mode=mode):
            cols, b = BET[(mode, j, i)]
            return lam*s*(F[:, cols] @ b)
    MS, _, _ = closure_corr(N["W"], corr=f)
    return cerr(MS[15], N["mB"][15])

MODES = ["const", "feat8", "shuf", "orac"]
print("\nλ      " + "".join("%-12s" % m for m in MODES) + "   ×проти бази")
RES = {}
for lam in LAMS:
    row = []
    for mode in MODES:
        e = [run(mode, lam, j) for j in range(NN)]
        RES[(mode, lam)] = e; row.append(float(np.median(e)))
    print("%.2f   " % lam + "".join("%-12s" % ("%.4f%%" % (100*v)) for v in row)
          + "   " + " / ".join("×%.3f" % (B0/v) for v in row), flush=True)

print("")
for mode in MODES:
    bl = max(LAMS, key=lambda l: B0/np.median(RES[(mode, l)]))
    w = sum(1 for j in range(NN) if RES[(mode, bl)][j] < BASE[j])
    print("%-6s найкраща λ=%.2f  ×%.4f  знаковий %d/%d"
          % (mode, bl, B0/np.median(RES[(mode, bl)]), w, NN))
    # парабола по трьох точках навколо оптимуму (закон С104)
    k = LAMS.index(bl)
    if 0 < k < len(LAMS)-1:
        e0 = np.median(RES[(mode, LAMS[k-1])]); e1 = np.median(RES[(mode, bl)])
        e2 = np.median(RES[(mode, LAMS[k+1])]); h = LAMS[k]-LAMS[k-1]
        den = e0 - 2*e1 + e2
        if abs(den) > 1e-30:
            print("       вершина параболи λ* = %.3f" % (bl + h*(e0-e2)/(2*den)))
log("ГОТОВО")
