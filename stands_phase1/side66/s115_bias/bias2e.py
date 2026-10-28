"""С115-н: ВИПРАВЛЕНА ПЛОМБА «ЯКІР». Мусить відтворити ceil1 0.0516% (×2.33).

🚨 ПОМИЛКА В bias2c/d: після підміни `m` істинним я ПЕРЕРАХОВУВАВ
   dd = (mp²+s²)Φ + mp·s·φ − m²,  беручи E[relu²] із ЗАМИКАННЯ, а `m` — ІСТИННЕ.
   Змішані величини ⟹ дисперсія неузгоджена (може стати від'ємною) ⟹ ×0.65.
   Правильно (як у gacc): підміняти ЛИШЕ канал `m`, `dd` лишати замикальну.
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, glob
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024

def closure(W, anchor=None, upto=None, keepdd=True):
    m = np.zeros(NW); C = np.eye(NW); MS = []
    for i in range(D):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy())
        if anchor is not None and (upto is None or i <= upto):
            m = anchor[i].copy()
            if not keepdd: dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if i == D-1: break
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))

NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:8]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    c = "/home/trim/wblogs/w2_%s.npy" % nm
    if os.path.exists(c):
        NETS.append(dict(name=nm, W=np.load(c).astype(np.float64),
                         mA=np.asarray(d["m_half"])[:,0,:], mB=np.asarray(d["m_half"])[:,1,:]))
log("мереж: %d" % len(NETS))
base = [cerr(closure(N["W"])[15], N["mB"][15]) for N in NETS]
B0 = float(np.median(base)); log("база %.4f%%" % (100*B0))

print("\nякір `m` з половини A (dd ЗАМИКАЛЬНА — виправлено):")
print("до шару   похибка      ×база")
for up in [0,2,4,6,8,10,12,13,14,None]:
    e = [cerr(closure(N["W"], anchor=N["mA"], upto=up)[15], N["mB"][15]) for N in NETS]
    v = float(np.median(e))
    print("  %-8s %.5f%%    ×%.3f" % ("усі" if up is None else up, 100*v, B0/v), flush=True)
print("\n🔒 контроль — СТАРА (хибна) схема, dd перераховувалась:")
e = [cerr(closure(N["W"], anchor=N["mA"], keepdd=False)[15], N["mB"][15]) for N in NETS]
print("  усі шари  %.5f%%    ×%.3f" % (100*np.median(e), B0/np.median(e)))
log("ГОТОВО")
