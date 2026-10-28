"""С117 — ЧИ ДОДАЄ γ₄ ПОНАД γ₃, І ЧИ ВИЖИВАЄ ЦЕ ПРИ РОЗДІЛЕНИХ ВИБІРКАХ.

[res117] у рекурсії шари 9..15: g3 ×1.1924 (c=0.40) · g3+g4 ×1.2288 (c=0.60).
🚨 АЛЕ γ₄ і `relu_15` у банку `gamall` рахувались З ОДНИХ СЕМПЛІВ ⟹ спільний
шум. Контроль на шарі 0 у res117 показав, що артефакт РЕАЛЬНИЙ: там `p` гаусів
точно (γ₃=γ₄=0), а R²_edg = 0.585 на чистому шумі.

Тут поправка береться з ПОЛОВИНИ A, істина — з ПОЛОВИНИ B (`gamhalf_*`,
12 мереж × 2 × 4.19e6 семплів) ⟹ спільного шуму НЕМАЄ за побудовою.

🔮 X1 (55%) виграш γ₄ понад γ₃ виживає, але слабшає (×1.20 замість ×1.229)
🔮 X2 (35%) при N=6144 γ₄ ще щось дає (бо sd(ĝ₄)=0.0625 при розкиді ~0.19)
🔮 X3 (70%) контроль «чужа» ламає обидві
"""
import sys, glob, time
import numpy as np, pyarrow.parquet as pq
sys.argv = ["x"]
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
V_MC, ANCH_E2, MULT = 3.3478454e-06, 1.4355576e-06, 0.1
LAY = tuple(range(9, 16))
T0 = time.time()
def lg(m): print("[%6.1fs] %s" % (time.time()-T0, m), flush=True)
def cerr(x, T):
    a = float(x @ T)/float(x @ x); r = a*x - T; return float(r @ r)/len(T)

def closure(W, dl=None, c=0.0):
    m = np.zeros(NW); C = np.eye(NW)
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if dl is not None and i in LAY and c != 0.0: m = m + c*s*dl[i]
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        if i == D-1: break
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return m

tb = pq.read_table("/home/trim/wb2data/dev24/data/mini-00000-of-00001.parquet")
NAMES = [tb["mlp_name"][i].as_py() for i in range(tb.num_rows)]
B = []
for f in sorted(glob.glob("/home/trim/wblogs/gamhalf_*.npz")):
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    W = np.asarray(tb["weights"][NAMES.index(nm)].as_py(), np.float64).reshape(D, NW, NW)
    b = dict(name=nm, W=W, TA=d["h0_relu_%d" % (D-1)], TB=d["h1_relu_%d" % (D-1)])
    for h in ("h0", "h1"):
        b[h] = {L: (d["%s_mp_%d" % (h, L)], d["%s_sd_%d" % (h, L)],
                    d["%s_g3_%d" % (h, L)], d["%s_g4_%d" % (h, L)]) for L in range(D)}
    B.append(b)
lg("мереж %d (половини по %d семплів)" % (len(B), int(np.load(
    sorted(glob.glob("/home/trim/wblogs/gamhalf_*.npz"))[0])["n"])//2))

def terms(b, h, use4, noiseN=None, shrink=None, rs=None):
    """δ = −(γ₃/6)tφ + use4·(γ₄/24)(t²−1)φ, оцінене на половині h"""
    out = {}
    for L in range(D):
        mp, sd, g3, g4 = b[h][L]
        t = mp/np.maximum(sd, 1e-300); ph = phi(t)
        if noiseN is not None:
            for g, k in ((g3, 6.0), (g4, 24.0)):
                pass
            s3, s4 = np.sqrt(6.0/noiseN), np.sqrt(24.0/noiseN)
            g3 = g3 + s3*rs.standard_normal(NW); g4 = g4 + s4*rs.standard_normal(NW)
            for g, sg in ((g3, s3), (g4, s4)):
                mu = g.mean(); vb = max(g.var() - sg*sg, 0.0)
                S = vb/(vb + sg*sg) if shrink is None else shrink
                g[:] = mu + S*(g - mu)
        out[L] = -(g3/6.0)*t*ph + (use4*(g4/24.0)*(t*t - 1.0)*ph)
    return out

BASE_B = float(np.mean([cerr(closure(b["W"]), b["TB"]) for b in B]))
def sc(v): return (1.0/(1.0/V_MC + 1.0/(v*(ANCH_E2/BASE_B))))*MULT
lg("база (ціль=половина B) %.6e · adj %.4e" % (BASE_B, sc(BASE_B)))

CS = [0.20, 0.40, 0.60, 0.80, 1.00]
rs = np.random.RandomState(11711)
ROWS = [
    ("A→B  g3",      lambda b: terms(b, "h0", 0.0), "TB"),
    ("A→B  g3+g4",   lambda b: terms(b, "h0", 1.0), "TB"),
    ("A→A  g3+g4",   lambda b: terms(b, "h0", 1.0), "TA"),   # спільний шум (для щілини)
    ("N6144 g3",     lambda b: terms(b, "h0", 0.0, 6144, None, rs), "TB"),
    ("N6144 g3+g4",  lambda b: terms(b, "h0", 1.0, 6144, None, rs), "TB"),
    ("🔒чужа g3+g4",  None, "TB"),
    ("🔒переміш",     None, "TB"),
    ("🔒стала/шар",   None, "TB"),
]
print("\n── шари 9..15, ×база у косинусній похибці ──")
print("%14s | %s" % ("режим", " ".join("c=%.2f" % c for c in CS)))
DL = [terms(b, "h0", 1.0) for b in B]
BASE_A = float(np.mean([cerr(closure(b["W"]), b["TA"]) for b in B]))
for nm, fn, tgt in ROWS:
    if fn is not None: dls = [fn(b) for b in B]
    elif "чужа" in nm: dls = [DL[(i+5) % len(B)] for i in range(len(B))]
    elif "переміш" in nm:
        dls = [{L: DL[i][L][rs.permutation(NW)] for L in range(D)} for i in range(len(B))]
    else:
        dls = [{L: np.full(NW, DL[i][L].mean()) for L in range(D)} for i in range(len(B))]
    base = BASE_A if tgt == "TA" else BASE_B
    row = [base/float(np.mean([cerr(closure(b["W"], dl, c), b[tgt])
                               for b, dl in zip(B, dls)])) for c in CS]
    print("%14s | %s" % (nm, " ".join("%6.4f" % x for x in row)))
    sys.stdout.flush()
lg("ГОТОВО")
print("DONE_G4X")
