"""cv90 — CONTROL VARIATES: б'ємо в 91.2% балу (дисперсію МК), а не в замикання.

🚨 ПЕРША ПЕРЕВІРКА, ЯКА МОЖЕ ВБИТИ ВСЕ: у сабміті стоїть антитетика `[z,−z]`.
Для БУДЬ-ЯКОЇ НЕПАРНОЇ `g` вибіркове середнє РІВНО нуль ⟹ `mean(f−βg)=mean(f)`,
тобто лінійна CV (`g=Jz`) не дає НІЧОГО за побудовою. Кандидати мусять бути ПАРНІ.

КАНДИДАТИ з ТОЧНО відомим середнім (це головна вимога — інакше зміщення):
  A `|h₁| = |zW₁|`      E = σ√(2/π), σ²=Σ W₁², точно (z гаусів)
  B `‖z‖²`              E = n, точно
  C `‖Jz‖²`             E = tr(JᵀJ), точно; J = ΠW_k diag(p_k) з ЗАМИКАННЯ
  D `h₁²`               E = σ², точно
Виграш = 1/(1−R²) від багатовимірної регресії f на набір CV.

ПЛОМБА: непарна CV (`g=Jz`) мусить дати R²≈0 під антитетикою. Якщо ні —
я не розумію власну схему.
"""
import os, sys, time
os.environ.setdefault("CUDA_DEVICE_ORDER","PCI_BUS_ID"); os.environ.setdefault("OMP_NUM_THREADS","2")
import numpy as np, torch
DEV=sys.argv[1] if len(sys.argv)>1 else "cuda:1"
NNET=int(sys.argv[2]) if len(sys.argv)>2 else 16
MREP=int(sys.argv[3]) if len(sys.argv)>3 else 24
NMC=5432; DT=torch.float32; SQ2=1.4142135623730951; ISP=0.3989422804014327
W=np.load("/home/trim/wbdata/fs_w.npy",mmap_mode="r"); n,L=W.shape[2],W.shape[1]
print(f"{DEV} · мереж {NNET} · повторів {MREP}\n")

def probs(ws):
    m=torch.zeros(n,dtype=torch.float64,device=DEV); C=torch.eye(n,dtype=torch.float64,device=DEV); P=[]
    for i in range(L):
        w=ws[i].double(); mh=m@w; Ch=w.T@(C@w); d=torch.diagonal(Ch).clamp(min=1e-30); s=d.sqrt()
        t=mh/s; p=0.5*(1.0+torch.erf(t/SQ2)); f=torch.exp(-0.5*t*t)*ISP; ma=s*(t*p+f); P.append(p)
        if i==L-1: break
        va=((mh*mh+d)*p+mh*s*f-ma*ma).clamp(min=1e-30); Ca=Ch*torch.outer(p,p)
        C=Ca+torch.diag(va-torch.diagonal(Ca)); m=ma
    return P

def jac(ws,P):
    J=torch.eye(n,dtype=DT,device=DEV)
    for i in range(L-1): J=(J@ws[i])*P[i].to(DT).unsqueeze(0)
    return J@ws[L-1]

def sample(ws,J,sg1,seed):
    """повертає f (вихід) і CV-и, усі УСЕРЕДНЕНІ по антитетичній парі."""
    g=torch.Generator(device=DEV).manual_seed(seed)
    z=torch.randn(NMC//2,n,generator=g,device=DEV,dtype=DT); z=torch.cat([z,-z],0)
    cov=(z.T@z).double()/z.shape[0]; z=z@torch.linalg.inv(torch.linalg.cholesky(cov)).T.to(DT)
    x=z
    for i in range(L):
        h=x@ws[i]
        if i==0: h1=h
        x=torch.relu(h) if i<L-1 else h
    f=torch.relu(h).mean(0).double()
    A=(h1.abs().mean(0)-sg1*ISP*SQ2/1.0).double()      # |h₁| − E|h₁|
    D=((h1*h1).mean(0)-sg1*sg1).double()               # h₁² − σ²
    zn=((z*z).sum(1).mean()-float(n))                  # ‖z‖² − n  (скаляр)
    Jz=z@J; C_=((Jz*Jz).mean(0)-(J*J).sum(0)).double() # ‖Jz‖² покомпонентно
    odd=(z@J).mean(0).double()                         # ПЛОМБА: непарна
    return f,torch.stack([A,D,C_]),float(zn),odd

R2=[];R2o=[];GA=[]
t0=time.time()
for q in range(NNET):
    ws=[torch.as_tensor(np.ascontiguousarray(W[q,i]).astype(np.float32),device=DEV) for i in range(L)]
    P=probs(ws); J=jac(ws,P)
    sg1=torch.sqrt(torch.clamp((ws[0]*ws[0]).sum(0),min=1e-30))
    F=[];G=[];O=[]
    for r in range(MREP):
        f,cv,zn,odd=sample(ws,J,sg1,808000+977*q+r); F.append(f);G.append(cv);O.append(odd)
    F=torch.stack(F); G=torch.stack(G); O=torch.stack(O)          # (MREP,n),(MREP,3,n),(MREP,n)
    Fc=F-F.mean(0,keepdim=True)
    def r2of(y,X):
        """R² через нормальні рівняння з регуляризацією; nan-безпечно."""
        sd=X.std(0); ok=sd>1e-20
        if not bool(ok.any()): return 0.0
        X=X[:,ok]/sd[ok]
        A=X.T@X+1e-8*torch.eye(X.shape[1],dtype=X.dtype,device=X.device)
        b=torch.linalg.solve(A,X.T@y)
        res=y-X@b; v=float((y*y).sum())
        if not np.isfinite(v) or v<1e-300: return 0.0
        r=1.0-float((res*res).sum())/v
        return r if np.isfinite(r) else 0.0
    r2s=[];r2o=[];per=[[],[],[]]
    for j in range(0,n,8):
        y=Fc[:,j]
        X=torch.stack([G[:,k,j]-G[:,k,j].mean() for k in range(3)],1)
        r2s.append(r2of(y,X))
        for k in range(3):
            per[k].append(r2of(y,X[:,k:k+1]))
        r2o.append(r2of(y,(O[:,j]-O[:,j].mean()).unsqueeze(1)))
    R2.append(float(np.mean(r2s))); R2o.append(float(np.mean(r2o)))
    GA.append([float(np.mean(per[k])) for k in range(3)])
    if q%5==0: print(f"  net{q:>3d} R²(парні)={R2[-1]:.4f} R²(непарна)={R2o[-1]:.4f} ({time.time()-t0:.0f} с)",flush=True)

r2=float(np.mean(R2)); ro=float(np.mean(R2o))
k=1.0/max(1.0-r2,1e-9)
print(f"\n  час {time.time()-t0:.0f} с\n"+"="*70)
print(f"CONTROL VARIATES · {NNET} мереж × {MREP} повторів")
print("="*70)
print(f"  R² парних CV (|h₁|, h₁², ‖Jz‖²) = {r2:.4f}  ± {np.std(R2,ddof=1)/np.sqrt(NNET):.4f}")
print(f"  ⛔ПЛОМБА R² непарної (Jz)        = {ro:.4f}  "
      f"{'✅ ~0, антитетика її вбиває, як і виведено' if abs(ro)<0.08 else '⛔ НЕ РОЗУМІЮ СХЕМУ'}")
pa=np.array(GA).mean(0)
print(f"  R² окремо: |h₁| {pa[0]:.4f} · h₁² {pa[1]:.4f} · ‖Jz‖² {pa[2]:.4f}")
print(f"  🚨 фон від фітингу шуму при {MREP} точках і 1 регресорі ≈ {1.0/MREP:.4f},"
      f" при 3 ≈ {3.0/MREP:.4f}")
print(f"\n📣 виграш дисперсії ×{k:.4f}  ⟹ бал ≈ {1.86762e-07/(0.912*k+0.088):.4e}")
print(f"   {'✅ Є ЩО КОПАТИ' if k>1.15 else '⛔ СЛАБКО'}")
