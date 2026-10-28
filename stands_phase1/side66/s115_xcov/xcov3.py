"""С115-о стадія 3 (виправлення методології, зауваження ЮЗЕРА).

🚨 МОЯ ПОМИЛКА В xcov2: я стиснув Δ (1024×1024) у 3 скаляри на нейрон, бо між
незалежними жеребами W індекс нейрона не переносний. Але це обмеження
РЕАЛІЗОВНОСТІ, а закон каже спершу міряти СТЕЛЮ вкиданням ТОЧНОЇ величини.
⟹ негативний R²=0.0014 закривав мої зведення, а не об'єкт.

ТЕСТ A — СТЕЛЯ, БЕЗ СТИСНЕННЯ. Усередині ОДНІЄЇ мережі індекс має сенс ⟹
  b⊥ ≈ Δᵀu,  u ∈ R^1024 (або R^3072 зі стеком Δ₁₂,Δ₁₃,Δ₁₄).
  Фіт на ПОЛОВИНІ нейронів шару 15, вирок на ДРУГІЙ половині. Це оракульний
  тест (u вимагає знати b⊥), тобто ВЕРХНЯ МЕЖА класу.
  🔒 контролі: перемішана ціль · Δ ЧУЖОЇ мережі · випадкова матриця тієї ж норми

ТЕСТ B — 32×32, БІНУВАННЯ ЗА `t` (пропозиція ЮЗЕРА у переносній формі).
  Клітинка (a,b) = середнє Δ по нейронах шару 14 з t-біну a і шару 15 з біну b.
  ⟹ 32 числа на нейрон, ОДНАКОВОГО СЕНСУ в кожній мережі ⟹ LOO між мережами.

ПЕРЕДБАЧЕННЯ ДО ПРОГОНУ:
  P1 тест A (стеля, повна Δ) дасть held-out R² > 0.10                40%
  P2 якщо A дасть >0.3, то B дасть хоч 0.05                          50%
  P3 контроль «чужа Δ» дасть ~0                                      85%
  P4 стек трьох пар кращий за саму Δ₁₄                               55%
"""
import os, sys, time, glob
sys.path.insert(0, "/home/trim/gpupkg")
import numpy as np, cupy as cp

NNET = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LOGN = int(sys.argv[2]) if len(sys.argv) > 2 else 20
SEED, CHUNK, D, NW = 4242, 1 << 13, 16, 1024
PAIRS = [12, 13, 14]
NBIN = 32
N = 1 << LOGN
T0 = time.time()
def log(m): print("[%7.1fs] %s" % (time.time()-T0, m), flush=True)
exec(open("/home/trim/wbstand/neuro1.py").read().split("# ─────────── дані")[0])

def cl_np(W):
    m = np.zeros(NW); C = np.eye(NW); MS = []; TS = []
    for i in range(D):
        w = W[i]; q = w.T @ C @ w
        if i > KSIG: q = q*KAPPA
        s = np.sqrt(np.maximum(np.diag(q), 1e-20)); mp = m @ w
        t = mp/s; ph = phi(t); Ph = Phi(t); m = mp*Ph + s*ph
        if LAM_FEAT != 0.0: m = m - LAM_FEAT*shape_corr(q, s, t, Ph, ph, m, i/float(D))
        dd = (mp*mp + s*s)*Ph + (mp*s)*ph - m*m
        MS.append(m.copy()); TS.append(t.copy())
        if i == D-1: break
        C = q*np.outer(Ph, Ph) + 0.5*(q*q)*np.outer(ph/s, ph/s)
        C = C + np.diag(dd - np.diag(C))
    return MS, TS

def gpu_delta(Wg, n, sd):
    sm=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]
    mk=[cp.zeros(NW,dtype=cp.float64) for _ in range(D)]
    Sd={l:cp.zeros((NW,NW),dtype=cp.float64) for l in PAIRS}
    Sa={l:cp.zeros((NW,NW),dtype=cp.float64) for l in PAIRS}
    done=0;c=0
    while done<n:
        b=min(CHUNK,n-done)
        z=cp.random.RandomState(seed=sd+c).standard_normal((b,NW),dtype=cp.float32)
        H=[];x=z
        for i in range(D):
            p=x@Wg[i]; mm=(p>0.0); x=cp.maximum(p,0.0)
            H.append(x); mk[i]+=mm.sum(0,dtype=cp.float64); sm[i]+=x.sum(0,dtype=cp.float64)
        for l in PAIRS:
            Sd[l]+=(H[l].T@H[l]).astype(cp.float64); Sa[l]+=(H[l].T@H[l+1]).astype(cp.float64)
        done+=b;c+=1; del H,z,x,p,mm
    inv=1.0/float(n); M=[s*inv for s in sm]; PH=[s*inv for s in mk]
    out={}
    for l in PAIRS:
        Cd=Sd[l]*inv-cp.outer(M[l],M[l]); Ca=Sa[l]*inv-cp.outer(M[l],M[l+1])
        Dl=Ca-(Cd@Wg[l+1].astype(cp.float64))*PH[l+1][None,:]
        for k in range(l+2,D):                       # перенести стовпці на шар 15
            Dl=(Dl@Wg[k].astype(cp.float64))*PH[k][None,:]
        out[l]=Dl
    return out, M, PH

def ridge_fit(A, y, lam):
    G = A.T@A; tr = np.trace(G)/G.shape[0]
    return np.linalg.solve(G + lam*tr*np.eye(G.shape[0]), A.T@y)
def r2(y, p): 
    return 1.0 - ((y-p)**2).sum()/max(((y-y.mean())**2).sum(), 1e-300)

DAT=[]
for bank in sorted(glob.glob("/home/trim/wblogs/truthhi_*.npz"))[:NNET]:
    d=np.load(bank,allow_pickle=True); nm=str(d["name"])
    f="/home/trim/wblogs/w2_%s.npy"%nm
    if not os.path.exists(f): continue
    mA=np.asarray(d["m_half"])[:,0,:]
    W=np.load(f); Wg=[cp.asarray(W[i]) for i in range(D)]
    DEL,M,PH=gpu_delta(Wg,N,SEED)
    MS,TS=cl_np(W.astype(np.float64)); m15=MS[15]
    b=m15-mA[15]; a=(b@m15)/(m15@m15); yp=b-a*m15
    DAT.append(dict(name=nm, DEL={l:DEL[l].get().astype(np.float64) for l in PAIRS},
                    y=yp/np.linalg.norm(yp), t14=TS[14], t15=TS[15]))
    log("%-18s ‖b⊥‖/‖b‖ %.3f"%(nm, np.linalg.norm(yp)/np.linalg.norm(b)))
    del Wg,DEL; cp.get_default_memory_pool().free_all_blocks()
NN=len(DAT); log("мереж: %d"%NN)

# ─────────── ТЕСТ A: СТЕЛЯ, повна Δ, розділ по НЕЙРОНАХ ───────────
rng=np.random.default_rng(7)
print("\nТЕСТ A — стеля класу (оракульний u, held-out половина нейронів шару 15)")
print("варіант          ridge   held-out R²      🔒переміш   🔒чужа Δ   🔒випадк.")
for tag, use in (("лише Δ₁₄",[14]), ("стек Δ₁₂₋₁₄",PAIRS)):
    for lam in (1e-3, 1e-1, 1.0):
        own,shf,frn,rnd=[],[],[],[]
        for j,dd in enumerate(DAT):
            A=np.concatenate([dd["DEL"][l] for l in use],0).T      # 1024 × (1024*|use|)
            y=dd["y"]; idx=rng.permutation(NW); S1,S2=idx[:NW//2],idx[NW//2:]
            own.append(r2(y[S2], A[S2]@ridge_fit(A[S1],y[S1],lam)))
            ys=y[rng.permutation(NW)]
            shf.append(r2(ys[S2], A[S2]@ridge_fit(A[S1],ys[S1],lam)))
            k=(j+1)%NN
            Af=np.concatenate([DAT[k]["DEL"][l] for l in use],0).T
            frn.append(r2(y[S2], Af[S2]@ridge_fit(Af[S1],y[S1],lam)))
            Ar=rng.standard_normal(A.shape)*A.std()
            rnd.append(r2(y[S2], Ar[S2]@ridge_fit(Ar[S1],y[S1],lam)))
        print("%-16s %.0e   %+.4f          %+.4f     %+.4f    %+.4f"
              % (tag, lam, np.median(own), np.median(shf), np.median(frn), np.median(rnd)), flush=True)

# ─────────── ТЕСТ B: 32×32, біни за t (переносно) ───────────
def binned(dd):
    q14=np.quantile(dd["t14"],np.linspace(0,1,NBIN+1)); q15=np.quantile(dd["t15"],np.linspace(0,1,NBIN+1))
    i14=np.clip(np.digitize(dd["t14"],q14[1:-1]),0,NBIN-1); i15=np.clip(np.digitize(dd["t15"],q15[1:-1]),0,NBIN-1)
    Dl=dd["DEL"][14]; B=np.zeros((NBIN,NBIN))
    for a in range(NBIN):
        ra=(i14==a)
        if ra.sum()==0: continue
        v=Dl[ra].mean(0)
        for bq in range(NBIN):
            rb=(i15==bq)
            if rb.sum(): B[a,bq]=v[rb].mean()
    return B[:,i15].T                                   # 1024 × 32, ознака нейрона j
print("\nТЕСТ B — 32×32, біни за t (переносна форма), LOO між мережами")
FB=[binned(dd) for dd in DAT]
for lam in (1e-3,1e-1,1.0):
    own,frn=[],[]
    for j in range(NN):
        A=np.concatenate([FB[k] for k in range(NN) if k!=j],0)
        y=np.concatenate([DAT[k]["y"] for k in range(NN) if k!=j],0)
        u=ridge_fit(A,y,lam); own.append(r2(DAT[j]["y"], FB[j]@u))
        k=(j+1)%NN; frn.append(r2(DAT[j]["y"], FB[k]@u))
    print("  ridge %.0e   LOO R² %+.4f    🔒чужа %+.4f"%(lam,np.median(own),np.median(frn)), flush=True)
log("ГОТОВО")
