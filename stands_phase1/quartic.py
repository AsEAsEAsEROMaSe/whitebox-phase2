"""ЧИ ЗОСЕРЕДЖЕНИЙ 4-й ПОРЯДОК? (заміна spectrum.py, який завалився по дисперсії оцінки)

spectrum.py був правильний за формулою, але оцінювач має дисперсію ~d^k ⟹ на k=4 шум
більший за сигнал (контроль: непарні порядки 0.0005 → 0.10 → 11.4 → 917). Не рахується.

ПИТАННЯ, ЯКЕ НАСПРАВДІ ВАЖИТЬ: 4-й порядок ЗОСЕРЕДЖЕНИЙ чи розмазаний?
  зосереджений у ~десятках напрямків ⟹ знімається контрольною змінною (якір БЕЗКОШТОВНИЙ:
      середнє ермітових ознак = 0 за побудовою, купувати семплами не треба)
  розмазаний по 1.8e8 ⟹ мертво, і дорога до 1e-8 закрита з цього боку

СХЕМА:
  1. y_even = (a(x)+a(−x))/2                      — після антитетики
  2. C = E[y_even·(xxᵀ−I)];  прибираємо оптимальну квадратику xᵀ(C/2)x − tr(C)/2
     ⟹ y_res = те, що лишається ПІСЛЯ v2 (2-й порядок знято точно)
  3. V = топ-K власних напрямків Σ_i C_i²  — де живе квадратична кривизна
  4. t = x@V (K-вим, стандартний гаус бо V ортонормальна)
     ознаки = ВСІ ермітові квартики He_α(t), |α|=4  ⟹ C(K+3,4) штук
  5. β підганяємо на половині A, дисперсію міряємо на половині B  ⟵ ПОЗА ВИБІРКОЮ

Гіпотеза, що перевіряється: кривизна вищого порядку живе в тих самих напрямках, що й
квадратична. Якщо так — крива «виграш(K)» швидко насититься на малому K.
"""
import torch, time, math, itertools, numpy as np
LOG='/mnt/win_d/llm/White-Box/quartic.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def even(W,x):
    a=x
    for w in W: a=torch.relu(a@w)
    b=-x
    for w in W: b=torch.relu(b@w)
    return (a+b)*0.5

def draw(W,seed0,nch,CH=2000):
    XS=[];YS=[]
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        XS.append(z); YS.append(even(W,z))
    return torch.cat(XS,0), torch.cat(YS,0)

def quad_proj(X,Y,CH=2000):
    """C_i = E[y_i (xxᵀ−I)] для всіх координат"""
    n=X.shape[0]; T=torch.zeros(D,D*D,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev)
    for s in range(0,n,CH):
        x=X[s:s+CH]; y=Y[s:s+CH].double()
        P=(x.unsqueeze(2)*x.unsqueeze(1)).reshape(x.shape[0],-1).double()
        T+=y.t()@P; SY+=y.sum(0); del P
    T/=n; SY/=n
    eye=torch.eye(D,dtype=torch.float64,device=dev).reshape(-1)
    return (T-SY.unsqueeze(1)*eye.unsqueeze(0)).reshape(D,D,D), SY

def remove_quad(X,Y,C,SY,CH=2000):
    """y_res = y − [xᵀ(C/2)x − tr(C)/2]  — оптимальна квадратична поправка"""
    n=X.shape[0]; out=torch.empty(n,D,dtype=torch.float64,device=dev)
    tr=0.5*torch.diagonal(C,dim1=1,dim2=2).sum(1)                 # tr(C)/2 по координатах
    for s in range(0,n,CH):
        x=X[s:s+CH].double(); m=x.shape[0]
        q=torch.einsum('sa,iab,sb->si',x,C*0.5,x)-tr.unsqueeze(0)
        out[s:s+m]=Y[s:s+m].double()-SY.unsqueeze(0)-q
    return out

# ---- ермітові квартики на K вимірах ----
def he(t,k):
    if k==0: return torch.ones_like(t)
    if k==1: return t
    if k==2: return t*t-1
    if k==3: return t*(t*t-3)
    return t*t*(t*t-6)+3
def quartic_feats(t,K):
    """всі He_α(t), |α|=4, нормовані на sqrt(α!) ⟹ приблизно ортонормальні"""
    idx=[a for a in itertools.combinations_with_replacement(range(K),4)]
    F=torch.empty(t.shape[0],len(idx),dtype=torch.float64,device=dev)
    for j,a in enumerate(idx):
        cnt={}
        for v in a: cnt[v]=cnt.get(v,0)+1
        col=torch.ones(t.shape[0],dtype=torch.float64,device=dev); nrm=1.0
        for v,c in cnt.items():
            col=col*he(t[:,v].double(),c); nrm*=math.factorial(c)
        F[:,j]=col/math.sqrt(nrm)
    return F

for net in (0,):
    W=load(net); log('='*76); log('МЕРЕЖА %d'%net)
    XA,YA=draw(W,110000,20); XB,YB=draw(W,150000,10)     # 40000 / 20000
    log('  підгонка на {:d}, перевірка на {:d} семплах'.format(XA.shape[0],XB.shape[0]))
    C,SY=quad_proj(XA,YA)
    RA=remove_quad(XA,YA,C,SY); RB=remove_quad(XB,YB,C,SY)
    vA=float((RA*RA).mean(0).sum()); vB=float((RB*RB).mean(0).sum())
    v0=float(((YB.double()-SY)**2).mean(0).sum())
    log('  парна Var (поза вибіркою) {:.4e} → після зняття 2-го порядку {:.4e}  ⟹ {:.2f}×'
        .format(v0,vB,v0/vB))
    M=torch.einsum('iab,icb->ac',C,C)                     # де живе квадратична кривизна
    ev,V=torch.linalg.eigh(M.cpu()); V=V.flip(1).to(dev)
    log('')
    log('  {:>4s} {:>8s} {:>14s} {:>14s} {:>10s}'.format('K','ознак','Var поза виб.','виграш 4-го','сукупно'))
    for K in (4,6,8,10,12,14,16):
        Vk=V[:,:K].double()
        FA=quartic_feats(XA.double()@Vk,K); FB=quartic_feats(XB.double()@Vk,K)
        P=FA.shape[1]
        if P>XA.shape[0]//4: log('  K={:2d}: ознак {:d} забагато — пропуск'.format(K,P)); continue
        G=FA.t()@FA/XA.shape[0]+torch.eye(P,dtype=torch.float64,device=dev)*1e-4
        beta=torch.linalg.solve(G, FA.t()@RA/XA.shape[0])
        res=RB-FB@beta                                    # ПОЗА ВИБІРКОЮ
        vq=float((res*res).mean(0).sum())
        log('  {:4d} {:8d} {:14.4e} {:14.2f}× {:9.2f}×'.format(K,P,vq,vB/vq,v0/vq))
        del FA,FB,G,beta,res
log('done.')
