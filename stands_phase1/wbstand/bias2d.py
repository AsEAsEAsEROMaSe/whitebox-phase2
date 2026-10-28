"""С115-н доважок: (1) дрібна сітка λ біля нуля (закон С104 про крок сітки),
(2) РОЗДІЛЕННЯ `b` на МАСШТАБ (вздовж m) і ФОРМУ (⊥ m) — бо косинусна метрика
масштаб уже знімає даром, і саме тут може ховатись різниця з фазою 1.
(3) 🚨 діагностика проваленої пломби «якір»: чи винні наші калібровані константи
    (KAPPA/LAM_FEAT), підібрані під НЕВИПРАВЛЕНУ траєкторію.
"""
import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, sys, glob
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])
D, NW = 16, 1024
RNG = np.random.default_rng(20260830)

def feats(t, Ph, ph, q):
    o = np.ones_like(t)
    return np.stack([o, t, Ph, ph, t*ph, t*t, q, q*q], 1)

def closure(W, anchor=None, kap=None, lamf=None):
    KA = KAPPA if kap is None else kap
    LF = LAM_FEAT if lamf is None else lamf
    m = np.zeros(NW); C = np.eye(NW); MS, FS, SS = [], [], []
    for i in range(D):
        w = W[i]; cp = w.T @ C @ w
        if i > KSIG: cp = cp*KA
        s = np.sqrt(np.maximum(np.diag(cp), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LF != 0.0: m = m - LF*shape_corr(cp, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); FS.append(feats(t, Ph, ph, np.sqrt(np.maximum(dd,0))/s)); SS.append(s.copy())
        if anchor is not None:
            m = anchor[i].copy(); dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        if i == D-1: break
        C = cp*np.outer(Ph, Ph) + 0.5*(cp*cp)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS, FS, SS

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))
def wls(F, y, r=1e-3):
    G = F.T@F; dg = np.sqrt(np.maximum(np.diag(G),1e-300))
    return np.linalg.solve(G/np.outer(dg,dg)+r*np.eye(F.shape[1]), (F.T@y)/dg)/dg

NETS = []
for f in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:8]:
    d = np.load(f, allow_pickle=True); nm = str(d["name"])
    c = "/home/trim/wblogs/w2_%s.npy" % nm
    if not os.path.exists(c): continue
    NETS.append(dict(name=nm, W=np.load(c).astype(np.float64),
                     mA=np.asarray(d["m_half"])[:,0,:], mB=np.asarray(d["m_half"])[:,1,:]))
NN = len(NETS); log("мереж: %d" % NN)

BASE,F15,S15,B15,MB,YS,YP = [],[],[],[],[],[],[]
for N in NETS:
    MS,FS,SS = closure(N["W"]); m = MS[15]
    BASE.append(cerr(m, N["mB"][15])); b = m - N["mA"][15]
    # розклад b на масштаб (вздовж m) і форму (⊥ m)
    a = (b@m)/(m@m); bpar = a*m; bperp = b - bpar
    YS.append(b/SS[15]); YP.append(bperp/SS[15])
    F15.append(FS[15]); S15.append(SS[15]); B15.append(b); MB.append(m)
    log("%-18s ‖b∥‖/‖b‖ %.3f   ‖b⊥‖/‖b‖ %.3f" %
        (N["name"], np.linalg.norm(bpar)/np.linalg.norm(b), np.linalg.norm(bperp)/np.linalg.norm(b)))
B0 = float(np.median(BASE))

Fa = np.concatenate(F15,0)
for nm_, Y in (("b/s ПОВНА", YS), ("b⊥/s ФОРМА", YP)):
    ya = np.concatenate(Y,0); ba = wls(Fa,ya); ra = ya-Fa@ba
    log("R² in-sample %-12s %.3f" % (nm_, 1-(ra*ra).sum()/((ya-ya.mean())**2).sum()))

BET = {}
for j in range(NN):
    F = np.concatenate([F15[k] for k in range(NN) if k!=j],0)
    for kind, Y in (("full",YS), ("perp",YP)):
        y = np.concatenate([Y[k] for k in range(NN) if k!=j],0)
        BET[(kind,j)] = wls(F,y)

print("\nдрібна сітка λ (закон С104):")
print("λ        out8-full    out8-perp    🔒oracle-perp")
for lam in [0.0,0.02,0.05,0.10,0.15,0.25,0.5,1.0]:
    row=[]
    for kind in ("full","perp","orac"):
        e=[]
        for j in range(NN):
            m = MB[j].copy()
            if kind=="orac":
                b=B15[j]; a=(b@MB[j])/(MB[j]@MB[j]); m = m - lam*(b - a*MB[j])
            else: m = m - lam*S15[j]*(F15[j]@BET[(kind,j)])
            e.append(cerr(m, NETS[j]["mB"][15]))
        row.append(float(np.median(e)))
    print("%.2f   " % lam + "".join("%-13s" % ("%.5f%%"%(100*v)) for v in row)
          + "  " + " / ".join("×%.4f"%(B0/v) for v in row), flush=True)

print("\n🚨 діагностика проваленої пломби «якір» — чи винні калібровані константи:")
for kap,lamf,tag in ((None,None,"наші KAPPA+LAM_FEAT"),(1.0,None,"KAPPA=1"),
                     (None,0.0,"LAM_FEAT=0"),(1.0,0.0,"обидві вимкнені = ЧИСТЕ замикання")):
    bb,aa=[],[]
    for N in NETS:
        MS,_,_ = closure(N["W"],kap=kap,lamf=lamf); bb.append(cerr(MS[15],N["mB"][15]))
        MSa,_,_ = closure(N["W"],anchor=N["mA"],kap=kap,lamf=lamf); aa.append(cerr(MSa[15],N["mB"][15]))
    print("  %-36s база %.4f%%  якір %.4f%%  ⟹ ×%.3f"
          % (tag,100*np.median(bb),100*np.median(aa),np.median(bb)/np.median(aa)), flush=True)
log("ГОТОВО")
