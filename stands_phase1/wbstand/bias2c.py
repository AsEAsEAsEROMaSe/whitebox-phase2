"""С115-н: 8-ОЗНАКОВА МОДЕЛЬ ЗМІЩЕННЯ ЗАМИКАННЯ У ФАЗІ 2. ЗАМІР ПІДСТАНОВКОЮ.

Аудит (С70/С75/С79): `b` пояснюється 8 безрозмірними ознаками (R²_oos 0.792),
підстановка дає ×1.085 — але той вирок винесено при вазі гілки 7.6% MSE.
У фазі 2 гілка замикання важить 70-82% ⟹ важіль на порядок більший.

🚨 ПОМИЛКА ПРИЛАДУ, СПІЙМАНА ДИМОВИМ ПРОГОНОМ (bias2b): ціль `b_i = m_наше,i −
m_істина,i` є НАКОПИЧЕНОЮ похибкою. Віднімати її НА КОЖНОМУ шарі = знімати те
саме 16 разів ⟹ ×0.33 у ВСІХ режимів, включно з оракулом. Оракул мусив би
допомагати ⟹ прилад, не явище.
⟹ ТУТ: поправка застосовується ОДИН раз, до фінального `m` (це і є питання
«чи універсальна модель зміщення покращує середнє»), а окремим режимом стоїть
ДИНАМІЧНИЙ якір (m ← істина на КОЖНОМУ шарі) як зміряна стеля ceil1 0.0516%.

РЕЖИМИ (усі — косинусна похибка проти половини B):
  base    наша рекурсія
  outc    − λ·s₁₅·β₀            константа на шар, LOO по мережах
  out8    − λ·s₁₅·(F₁₅@β)       8 ознак, LOO
  shuf    те саме, ФОРМА переставлена між нейронами   🔒контроль
  oout    − λ·b₁₅ (істина з половини A)               🔒стеля виходу (λ=1 ⟹ шум A/B)
  anch    m ← mA[i] на КОЖНОМУ шарі                   🔒мусить дати ≈0.0516%

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 anch відтворить 0.045..0.060%  [ceil1 0.0516%]              75%
  P2 oout при λ=1 = шум половин (~0.02%), тавтологія             90%
  P3 out8 LOO дає ×1.02..×1.20                                   50%
  P4 оптимум λ для out8 НИЖЧЕ 1.0 («поправка стиснена»)          75%
  P5 shuf має оптимум ближче до нуля, ніж out8                   65%
  P6 const бере >60% від out8                                    60%
  P7 база 0.118..0.130%                                          80%
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, sys, glob, time
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
import pyarrow.parquet as pq

NNET  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LAMS  = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
RIDGE = 1e-3
D, NW = 16, 1024
RNG   = np.random.default_rng(20260830)

def feats(t, Ph, ph, q):
    o = np.ones_like(t)
    return np.stack([o, t, Ph, ph, t*ph, t*t, q, q*q], 1)

def closure(W, anchor=None):
    """anchor[i] (якщо є) ЗАМІНЮЄ m на шарі i. Повертає MS, FS, SS."""
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
        MS.append(m.copy()); FS.append(feats(t, Ph, ph, q)); SS.append(s.copy())
        if anchor is not None:
            m = anchor[i].copy()
            dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
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
    return np.linalg.solve(G/np.outer(dg, dg) + ridge*np.eye(F.shape[1]), (F.T @ y)/dg)/dg

tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:NNET]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    if nm not in NAMES: continue
    cache = "/home/trim/wblogs/w2_%s.npy" % nm
    if os.path.exists(cache): W = np.load(cache).astype(np.float64)
    else:
        W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float32).reshape(D, NW, NW)
        np.save(cache, W); W = W.astype(np.float64)
    NETS.append(dict(name=nm, W=W, mA=np.asarray(d["m_half"])[:, 0, :],
                     mB=np.asarray(d["m_half"])[:, 1, :]))
NN = len(NETS); log("мереж: %d" % NN)

BASE, ANCH, Y15, F15, S15, B15, MB = [], [], [], [], [], [], []
for N in NETS:
    MS, FS, SS = closure(N["W"])
    BASE.append(cerr(MS[15], N["mB"][15]))
    MSa, _, _ = closure(N["W"], anchor=N["mA"])
    ANCH.append(cerr(MSa[15], N["mB"][15]))
    B15.append(MS[15] - N["mA"][15]); Y15.append((MS[15]-N["mA"][15])/SS[15])
    F15.append(FS[15]); S15.append(SS[15]); MB.append(MS[15])
    log("%-18s база %.4f%%  🔒якір %.4f%%  🔒шум A/B %.4f%%"
        % (N["name"], 100*BASE[-1], 100*ANCH[-1], 100*cerr(N["mA"][15], N["mB"][15])))
B0 = float(np.median(BASE))
log("🔒 P7 БАЗА %.4f%%  ·  🔒 P1 ЯКІР %.4f%% [ceil1 0.0516%%]  ⟹ ×%.3f"
    % (100*B0, 100*np.median(ANCH), B0/np.median(ANCH)))

BET = {}
for kind, cols in (("outc", [0]), ("out8", list(range(8)))):
    for j in range(NN):
        F = np.concatenate([F15[k][:, cols] for k in range(NN) if k != j], 0)
        y = np.concatenate([Y15[k] for k in range(NN) if k != j], 0)
        BET[(kind, j)] = (cols, wls(F, y))
Fa = np.concatenate(F15, 0); ya = np.concatenate(Y15, 0)
ba = wls(Fa, ya); ra = ya - Fa@ba
log("R² in-sample шар15 (довідка, НЕ вирок): %.3f" % (1 - (ra*ra).sum()/((ya-ya.mean())**2).sum()))
PERM = RNG.permutation(NW)

def run(mode, lam, j):
    m = MB[j].copy()
    if mode == "oout": m = m - lam*B15[j]
    elif mode == "shuf":
        cols, b = BET[("out8", j)]; v = F15[j][:, cols] @ b
        m = m - lam*S15[j]*v[PERM]
    else:
        cols, b = BET[(mode, j)]
        m = m - lam*S15[j]*(F15[j][:, cols] @ b)
    return cerr(m, NETS[j]["mB"][15])

MODES = ["outc", "out8", "shuf", "oout"]
print("\nλ      " + "".join("%-11s" % m for m in MODES) + "  ×проти бази")
RES = {}
for lam in LAMS:
    row = []
    for mode in MODES:
        e = [run(mode, lam, j) for j in range(NN)]
        RES[(mode, lam)] = e; row.append(float(np.median(e)))
    print("%.2f   " % lam + "".join("%-11s" % ("%.4f%%" % (100*v)) for v in row)
          + "  " + " / ".join("×%.3f" % (B0/v) for v in row), flush=True)
print("")
for mode in MODES:
    bl = max(LAMS, key=lambda l: B0/np.median(RES[(mode, l)]))
    w = sum(1 for j in range(NN) if RES[(mode, bl)][j] < BASE[j])
    line = "%-5s λ*=%.2f  ×%.4f  знаковий %d/%d" % (mode, bl, B0/np.median(RES[(mode, bl)]), w, NN)
    k = LAMS.index(bl)
    if 0 < k < len(LAMS)-1:
        e0, e1, e2 = (np.median(RES[(mode, LAMS[k+d])]) for d in (-1, 0, 1))
        den = e0 - 2*e1 + e2
        if abs(den) > 1e-30: line += "   вершина %.3f" % (bl + (LAMS[k]-LAMS[k-1])*(e0-e2)/(2*den))
    print(line)
log("ГОТОВО")
