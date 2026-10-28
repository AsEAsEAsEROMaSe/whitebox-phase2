"""С115-н: 8-ОЗНАКОВА МОДЕЛЬ ЗМІЩЕННЯ ЗАМИКАННЯ, ПЕРЕНЕСЕНА У ФАЗУ 2.

Питання ЮЗЕРА: «те, що є шумом, насправді несе всі вектори і моменти».
Аудит показав: у фазі 1 це ЗМІРЯНО — `b` пояснюється на R²_oos=0.792 вісьмома
безрозмірними ознаками (С70), але дає лише ×1.085 (дві пломби: драбина
lowrank_rungs + biassplit). ⚠️ АЛЕ той вирок винесено, коли зміщення важило
7.6% MSE. У фазі 2 гілка замикання важить 70-82% ⟹ важіль на порядок більший.

🚨 ЗАКОН С70: `R²` НЕ характеризує залишок (два залишки з ІДЕНТИЧНИМ R² дали
×3.265 і ×1.382) ⟹ міряти ЛИШЕ ПІДСТАНОВКОЮ в рекурсію.

СХЕМА: ознаки `[1,t,Φ,φ,tφ,t²,q,q²]`, q=√dd/s (фазо-2 замінник `rd/s`).
Ціль `y=b/s`, `b = m_наше − m_істина[половина A]`. Фіт LOO по мережах, пошарово.
Підстановка: `m ← m − λ·s·(F@β)` НА КОЖНОМУ шарі, далі рекурсія йде як є.
Вирок — косинусна похибка фінального `m` проти половини B (незалежної).

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 оракул (точне b з половини A) дає ×3..×10 у косинусній похибці      70%
  P2 LOO-фіт FEAT8 дає ×1.05..×1.3                                       55%
  P3 оптимум λ ЛЕЖИТЬ НИЖЧЕ 1.0 («поправка стиснена», 7-й раз)           75%
  P4 чужа β (перемішані мережі) дає оптимум λ≈0                          80%
  P5 база відтвориться як 0.118..0.125% (записане 0.1202%)               80%
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

def feats(t, Ph, ph, q):
    o = np.ones_like(t)
    return np.stack([o, t, Ph, ph, t*ph, t*t, q, q*q], 1)

def closure_corr(W, corr=None):
    """рекурсія оцінювача; corr(i, t, Ph, ph, q, s) -> поправка до m (або None).
    Повертає (MS, FS) де FS[i] = матриця ознак шару i."""
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
            m = m - corr(i, F, s)
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
    rhs = (F.T @ y)/dg
    return np.linalg.solve(Gn + ridge*np.eye(F.shape[1]), rhs)/dg

# ─────────── дані ───────────
tb = pq.read_table(PQ)
NAMES = {tb["mlp_name"][k].as_py(): k for k in range(tb.num_rows)}
banks = sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:NNET]
log("банків: %d" % len(banks))

NETS = []
for f in banks:
    d = np.load(f, allow_pickle=True)
    nm = str(d["name"])
    if nm not in NAMES: log("  пропуск %s" % nm); continue
    W = np.asarray(tb["weights"][NAMES[nm]].as_py(), dtype=np.float64).reshape(D, NW, NW)
    NETS.append(dict(name=nm, W=W, mA=np.asarray(d["m_half"])[:, 0, :],
                     mB=np.asarray(d["m_half"])[:, 1, :]))
    log("  %s" % nm)

# ─────────── база + збір ознак і цілей ───────────
BASE, YS, FSS = [], [], []
for k, N in enumerate(NETS):
    MS, FS, SS = closure_corr(N["W"])
    BASE.append(cerr(MS[15], N["mB"][15]))
    YS.append([(MS[i] - N["mA"][i])/SS[i] for i in range(D)])
    FSS.append(FS)
    log("%-18s база %.4f%%  (шум цілі половини: %.4f%%)"
        % (N["name"], 100*BASE[-1], 100*cerr(N["mA"][15], N["mB"][15])))
log("🔒 P5 БАЗА (медіана) %.4f%%   [записане 0.1202%%]" % (100*np.median(BASE)))

# ─────────── LOO-фіт по мережах, пошарово ───────────
NN = len(NETS)
BET = {}   # BET[(j,i)] = коефіцієнти для мережі j, шару i (навчені БЕЗ j)
for j in range(NN):
    for i in range(D):
        F = np.concatenate([FSS[k][i] for k in range(NN) if k != j], 0)
        y = np.concatenate([YS[k][i]  for k in range(NN) if k != j], 0)
        BET[(j, i)] = wls(F, y)

# in-sample R² для довідки (НЕ для вироку — закон С70)
r2 = []
for i in range(D):
    F = np.concatenate([FSS[k][i] for k in range(NN)], 0)
    y = np.concatenate([YS[k][i]  for k in range(NN)], 0)
    b = wls(F, y); r = y - F@b
    r2.append(1.0 - (r*r).sum()/max(((y-y.mean())**2).sum(), 1e-300))
log("R² in-sample по шарах: " + " ".join("%.3f" % v for v in r2))

# ─────────── ПІДСТАНОВКА ───────────
def run(mode, lam, j):
    N = NETS[j]
    if mode == "orac":
        MS0, _, SS0 = closure_corr(N["W"])
        bt = [MS0[i] - N["mA"][i] for i in range(D)]
        f = lambda i, F, s: lam*bt[i]
    elif mode == "feat":
        f = lambda i, F, s: lam*s*(F @ BET[(j, i)])
    elif mode == "foreign":
        jf = (j+1) % NN
        f = lambda i, F, s: lam*s*(F @ BET[(jf, i)]) if jf != j else None
    MS, _, _ = closure_corr(N["W"], corr=f)
    return cerr(MS[15], N["mB"][15])

print("\nλ      %-10s %-10s %-10s" % ("FEAT8", "оракул", "🔒чужа"))
RES = {}
for lam in LAMS:
    row = []
    for mode in ("feat", "orac", "foreign"):
        e = [run(mode, lam, j) for j in range(NN)]
        row.append(np.median(e)); RES[(mode, lam)] = e
    print("%.2f   %.4f%%    %.4f%%    %.4f%%   ×%.3f / ×%.3f / ×%.3f"
          % (lam, 100*row[0], 100*row[1], 100*row[2],
             np.median(BASE)/row[0], np.median(BASE)/row[1], np.median(BASE)/row[2]), flush=True)

# знаковий тест на найкращій λ для FEAT8
bl = max([l for l in LAMS if l > 0], key=lambda l: np.median(BASE)/np.median(RES[("feat", l)]))
w = sum(1 for j in range(NN) if RES[("feat", bl)][j] < BASE[j])
print("\nзнаковий FEAT8 при λ=%.2f: %d/%d" % (bl, w, NN))
log("ГОТОВО")
