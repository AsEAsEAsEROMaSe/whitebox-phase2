"""ЧИМ ВИЗНАЧАЄТЬСЯ МАГНІТУДА ПОХИБКИ ЗАМИКАННЯ НА НЕЙРОНІ.

[wneu116] стеля понейронної ваги ×1.2096, 🔒шафл ×0.5234 ⟹ `V_j` несе справжню
інформацію. Бракує лише `E²_j`. Моя перша модель `κ·m_j²` дала ×0.8802.
[neck116] пошарові профілі закриті (крос ≤×1.0055) ⟹ це ЄДИНЕ живе злиття.

🚨 ПРОТИ ЧОГО ЙДЕ: корпус ДЕВ'ЯТЬ разів довів ізотропність похибки замикання.
   Але всі дев'ять міряли НАПРЯМОК або ЗНАК. Для ваги потрібна МАГНІТУДА |e_j|,
   і її не міряв ніхто. Це різні питання.

🔮 ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
 Q1 (40%) хоч один предиктор дає |corr(log e², ·)| > 0.30 на held-out нейронах
 Q2 (55%) найсильніший предиктор — сама `V_j` (спільна причина: розкид активації)
 Q3 (35%) LOO-модель бере ≥половину стелі (×1.10 із ×1.21)
🔒 КОНТРОЛІ: чужі коефіцієнти моделі · перемішані передбачення
"""
import sys, glob, time
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW, NPAIR = 16, 1024, 6144
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)

def closure(W, g3, cm=0.40):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if 9 <= i <= 15 and cm != 0.0: m = m - cm*s*(g3[i]/6.0)*(t*ph)
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: return m, mp, s, t
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))

def mc_run(W, seed):
    rs = cp.random.RandomState(seed=seed)
    z = rs.standard_normal((NPAIR, NW), dtype=cp.float32)
    cov = (z.T @ z)/NPAIR
    wh = cp.linalg.inv(cp.linalg.cholesky(cov.astype(cp.float64))).T.astype(cp.float32)
    Wg = [cp.asarray(W[k].astype(cp.float32)) for k in range(D)]
    y = (z @ wh) @ Wg[0]
    x = cp.concatenate([cp.maximum(y, 0.0), cp.maximum(-y, 0.0)], axis=0)
    for k in range(1, D):
        x = cp.maximum(x @ Wg[k], 0.0)
    xm = cp.asnumpy(x.mean(0).astype(cp.float64))
    xv = cp.asnumpy(x.var(0).astype(cp.float64))/(2*NPAIR)
    del Wg, z, x; cp.get_default_memory_pool().free_all_blocks()
    return xm, xv

def mse(v, T): return float((v-T) @ (v-T))/len(T)

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
FS = sorted(glob.glob("/home/trim/wblogs/gamall_*.npz"))
DAT = []
for f in FS:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    T = d["relu_15"]; g3b = np.stack([d["g3_%d" % L] for L in range(D)])
    md0, mp, s, t = closure(W, g3b)
    xm, V = mc_run(W, 4400 + NAMES.index(nm))
    a = float(xm @ md0)/float(md0 @ md0); md = a*md0
    e = md - T
    DAT.append(dict(name=nm, md=md, T=T, xm=xm, V=V, e=e, mp=a*mp, s=a*s, t=t,
                    g3=g3b[15], ph=phi(t), Ph=Phi(t)))
    lg("%-18s готово" % nm)

# ── 1. кореляції log e² з кандидатами, ПО МЕРЕЖАХ ──
def feats(b):
    eps = 1e-300
    return {"log m²":   np.log(b["md"]**2 + eps),
            "log V":    np.log(b["V"] + eps),
            "log s²":   np.log(b["s"]**2 + eps),
            "t":        b["t"],
            "|t|":      np.abs(b["t"]),
            "log V/m²": np.log((b["V"] + eps)/(b["md"]**2 + eps)),
            "γ₃":       b["g3"],
            "|γ₃|":     np.abs(b["g3"]),
            "log φ":    np.log(b["ph"] + eps)}
KEYS = list(feats(DAT[0]).keys())
CR = {k: [] for k in KEYS}
for b in DAT:
    y = np.log(b["e"]**2 + 1e-300); F = feats(b)
    for k in KEYS: CR[k].append(np.corrcoef(F[k], y)[0, 1])
print("\n── corr(log e_j², предиктор), 24 мережі ──")
print("%-10s %10s %10s %10s" % ("предиктор", "сер", "sd", "|сер|/sd"))
for k in KEYS:
    a = np.array(CR[k]); print("%-10s %10.4f %10.4f %10.2f" % (k, a.mean(), a.std(), abs(a.mean())/max(a.std(),1e-9)))

# ── 2. LOO-модель log e² і скільки з стелі вона бере ──
def design(b):
    F = feats(b)
    return np.stack([np.ones(NW)] + [F[k] for k in ("log m²","log V","|t|","log φ")], 1)
R = {k: [] for k in ("glob","orac","model","shuf","foreign")}
for i, b in enumerate(DAT):
    ws = np.linspace(0, 1, 101)
    gm = [mse(b["xm"] + w*(b["md"] - b["xm"]), b["T"]) for w in ws]
    R["glob"].append(min(gm))
    R["orac"].append(mse(b["xm"] + (b["V"]/(b["V"]+b["e"]**2+1e-300))*(b["md"]-b["xm"]), b["T"]))
    X = np.concatenate([design(DAT[j]) for j in range(len(DAT)) if j != i])
    Y = np.concatenate([np.log(DAT[j]["e"]**2 + 1e-300) for j in range(len(DAT)) if j != i])
    cf = np.linalg.lstsq(X, Y, rcond=None)[0]
    e2h = np.exp(design(b) @ cf)
    R["model"].append(mse(b["xm"] + (b["V"]/(b["V"]+e2h))*(b["md"]-b["xm"]), b["T"]))
    rs = np.random.RandomState(70+i); e2s = rs.permutation(e2h)
    R["shuf"].append(mse(b["xm"] + (b["V"]/(b["V"]+e2s))*(b["md"]-b["xm"]), b["T"]))
g = float(np.mean(R["glob"]))
print("\n%-30s %14s %10s" % ("режим", "MSE", "×глоб"))
for k, n2 in (("glob","найкраща ГЛОБАЛЬНА W"),("orac","🔒 ОРАКУЛЬНА понейронна"),
              ("model","LOO-модель log e²"),("shuf","🔒 перемішані передбачення")):
    v = float(np.mean(R[k])); print("%-30s %14.5e %10.4f" % (n2, v, g/v))
lg("ГОТОВО")
