"""С115-о стадія 2: ЧИ ПЕРЕДБАЧАЄ МІЖШАРОВА СТРУКТУРА ФОРМУ ЗМІЩЕННЯ `b⊥`?

Стадія 1 (xcov) зміряла: відхилення від Стайна РЕАЛЬНЕ — 4.3% на сусідніх парах
при шумі 0.3-1.7%, і 22.3% на 0→15. PR(J_{0→15}) = 32.0 проти 2.74 у фазі 1.

ВИРІШАЛЬНЕ ПИТАННЯ: сьогодні зміряно, що ФОРМА зміщення `b⊥` варта ×6.15 і
НЕ Є функцією локальних (t,Φ,φ,q) — R² рівно 0.000. Чи є вона функцією
НЕЛОКАЛЬНИХ ознак, здобутих із Δ = Cov(h_l,h_l') − Стайн?

🔒 МІРЯЄМО ОРАКУЛЬНОЮ Δ (з істинних семплів) ⟹ це ВЕРХНЯ МЕЖА класу:
   якщо точна Δ не передбачає b⊥, ніяке її наближення не передбачить.
🔒 ціль з ПОЛОВИНИ A банку, вирок проти половини B (незалежні за побудовою)
🔒 контроль: ті самі ознаки з ЧУЖОЇ мережі
🔒 виправлено: центрування z у пломбі (у xcov його бракувало)

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 пломба (z,h0) після центрування впаде до ~0.05 (лишиться чистий шум)  70%
  P2 R²(b⊥ на ознаках з Δ) LOO > 0.05                                     25%
  P3 якщо R² є, він живе на ОСТАННІХ шарах (13-15), не на мілких          75%
  P4 чужа мережа дасть 0                                                  85%
"""
import os, sys, time, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp

NNET  = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LOGN  = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED  = 4242
CHUNK = 1 << 13
D, NW = 16, 1024
PAIRS = [11, 12, 13, 14]          # пари (l, l+1), з яких беремо ознаки
N = 1 << LOGN
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)

exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def cl_np(W):
    m = np.zeros(NW); C = np.eye(NW); MS = []
    for i in range(D):
        w = W[i]; cp_ = w.T @ C @ w
        if i > KSIG: cp_ = cp_*KAPPA
        s = np.sqrt(np.maximum(np.diag(cp_), 1e-20))
        mp = m @ w; t = mp/s; ph = phi(t); Ph = Phi(t)
        m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(cp_, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy())
        if i == D-1: break
        C = cp_*np.outer(Ph, Ph) + 0.5*(cp_*cp_)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS

def cerr(x, tr):
    a = (x@tr)/max((x*x).sum(),1e-300); e = a*x - tr
    return float(np.sqrt((e*e).sum())/np.sqrt((tr*tr).sum()))
def wls(F, y, r=1e-3):
    G = F.T@F; dg = np.sqrt(np.maximum(np.diag(G),1e-300))
    return np.linalg.solve(G/np.outer(dg,dg)+r*np.eye(F.shape[1]), (F.T@y)/dg)/dg

def gpu_pass(Wg, n, sd):
    sm=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]
    mk=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]
    Sd={l:cp.zeros((NW,NW),dtype=cp.float64) for l in PAIRS}
    Sa={l:cp.zeros((NW,NW),dtype=cp.float64) for l in PAIRS}
    Sz=cp.zeros((NW,NW),dtype=cp.float64); zs=cp.zeros(NW,dtype=cp.float64)
    done=0;c=0
    while done<n:
        b=min(CHUNK,n-done)
        z=cp.random.RandomState(seed=sd+c).standard_normal((b,NW),dtype=cp.float32)
        H=[];x=z
        for i in range(D):
            p=x@Wg[i]; mkk=(p>0.0); x=cp.maximum(p,0.0)
            H.append(x); mk[i]+=mkk.sum(0,dtype=cp.float64); sm[i]+=x.sum(0,dtype=cp.float64)
        for l in PAIRS:
            Sd[l]+=(H[l].T@H[l]).astype(cp.float64); Sa[l]+=(H[l].T@H[l+1]).astype(cp.float64)
        Sz+=(z.T@H[0]).astype(cp.float64); zs+=z.sum(0,dtype=cp.float64)
        done+=b;c+=1; del H,z,x,p,mkk
    inv=1.0/float(n)
    M=[s*inv for s in sm]; PH=[s*inv for s in mk]
    DEL={}
    for l in PAIRS:
        Cd=Sd[l]*inv-cp.outer(M[l],M[l]); Ca=Sa[l]*inv-cp.outer(M[l],M[l+1])
        pred=(Cd@Wg[l+1].astype(cp.float64))*PH[l+1][None,:]
        DEL[l]=Ca-pred
    Cz=Sz*inv-cp.outer(zs*inv,M[0])                       # 🔒 З ЦЕНТРУВАННЯМ
    seal=float(cp.sqrt(((Cz-Wg[0]*PH[0][None,:].astype(cp.float32))**2).sum())/cp.sqrt((Cz*Cz).sum()))
    return M,PH,DEL,seal

BANKS=sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:NNET]
FEAT=[];YP=[];MEM=[];BASE=[];NAMES=[]
for bank in BANKS:
    d=np.load(bank,allow_pickle=True); nm=str(d["name"])
    f="/home/trim/wblogs/w2_%s.npy"%nm
    if not os.path.exists(f): log("нема ваг: %s"%nm); continue
    mA=np.asarray(d["m_half"])[:,0,:]; mB=np.asarray(d["m_half"])[:,1,:]
    W=np.load(f); Wg=[cp.asarray(W[i]) for i in range(D)]
    M,PH,DEL,seal=gpu_pass(Wg,N,SEED)
    MS=cl_np(W.astype(np.float64)); m15=MS[15]
    b=m15-mA[15]; a=(b@m15)/(m15@m15); bperp=b-a*m15
    # ознаки з Δ: для КОЖНОГО вихідного нейрона j беремо зведення стовпця Δ[:,j]
    # 🚨 ознаки пари (l,l+1) індексовані нейронами шару l+1 ⟹ ПЕРЕНЕСТИ на 15
    cols=[]
    for l in PAIRS:
        Dl=DEL[l]; ml=M[l]
        v1=cp.sqrt((Dl*Dl).sum(0))                      # норма стовпця
        v2=ml@Dl                                        # проєкція на m_l
        v3=(Wg[l+1].astype(cp.float64)*Dl).sum(0)       # відхилення E[relu(p_j)²]
        for v in (v1,v2,v3):
            u=v.copy()
            for k in range(l+2,D):
                u=(u@Wg[k].astype(cp.float64))*PH[k]
            cols.append(u)
    Fx=cp.stack(cols,1).get().astype(np.float64)
    Fx=Fx[:, np.abs(Fx).max(0) > 0]
    Fx=np.concatenate([np.ones((NW,1)), (Fx-Fx.mean(0))/ (Fx.std(0)+1e-300)],1)
    FEAT.append(Fx); YP.append(bperp/np.linalg.norm(bperp)); MEM.append((m15,mA[15],mB[15]))
    BASE.append(cerr(m15,mB[15])); NAMES.append(nm)
    log("%-18s 🔒пломба(z,h0) %.4f · база %.4f%% · ознак %d" % (nm,seal,100*BASE[-1],Fx.shape[1]))
    del Wg,DEL; cp.get_default_memory_pool().free_all_blocks()

NN=len(FEAT); log("мереж: %d"%NN)
Fa=np.concatenate(FEAT,0); ya=np.concatenate(YP,0)
ba=wls(Fa,ya); ra=ya-Fa@ba
log("R² in-sample (усі мережі): %.4f"%(1-(ra*ra).sum()/((ya-ya.mean())**2).sum()))
r2loo=[]
for j in range(NN):
    F=np.concatenate([FEAT[k] for k in range(NN) if k!=j],0)
    y=np.concatenate([YP[k] for k in range(NN) if k!=j],0)
    bb=wls(F,y); pr=FEAT[j]@bb; rr=YP[j]-pr
    r2loo.append(1-(rr*rr).sum()/((YP[j]-YP[j].mean())**2).sum())
print("\nR² LOO по мережах: "+" ".join("%.4f"%v for v in r2loo))
print("медіана LOO R² = %.4f   [локальні 8 ознак давали 0.000]"%np.median(r2loo))
# 🔒 чужа: ознаки іншої мережі
r2f=[]
for j in range(NN):
    k=(j+1)%NN
    F=np.concatenate([FEAT[q] for q in range(NN) if q!=j],0)
    y=np.concatenate([YP[q] for q in range(NN) if q!=j],0)
    bb=wls(F,y); pr=FEAT[k]@bb; rr=YP[j]-pr
    r2f.append(1-(rr*rr).sum()/((YP[j]-YP[j].mean())**2).sum())
print("🔒 ЧУЖА мережа:    "+" ".join("%.4f"%v for v in r2f))
print("   медіана = %.4f"%np.median(r2f))
log("ГОТОВО")
