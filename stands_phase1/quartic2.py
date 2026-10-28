"""ЧИ ЗОСЕРЕДЖЕНИЙ 4-й ПОРЯДОК — версія 2 (виправлення quartic.py).

ДВІ ПОМИЛКИ ПЕРШОЇ ВЕРСІЇ:
 1. C рахувалась БЕЗ центрування y. Математично байдуже (E[xxᵀ−I]=0), але шум оцінки
    визначається E[y²], а не Var(y). Активації додатні: середнє ~0.6 при Var ~0.016
    ⟹ E[y²] у ~20× більший ⟹ шум у 65536 елементах C перекрив сигнал (Var 4.16 → 117).
 2. Проміжне зняття квадратики ВЗАГАЛІ НЕ ПОТРІБНЕ: ермітові многочлени різних порядків
    ОРТОГОНАЛЬНІ під гаусовою мірою ⟹ регресія на квартики дає ті самі коефіцієнти
    незалежно від наявності квадратичної частини. Будував шумний крок на порожньому місці.

ЩО МІРЯЄМО: частку Var(y_even), пояснену ермітовими КВАРТИКАМИ у K-вимірному підпросторі,
ПОЗА ВИБІРКОЮ. Порівнюємо з відомими 51-57% другого порядку.

КОНТРОЛЬ (критичний): той самий вимір на ВИПАДКОВОМУ підпросторі тієї ж розмірності.
Без нього не відрізнити «квартики зосереджені тут» від «3876 ознак пояснять будь-що».
"""
import torch, time, math, itertools, numpy as np
LOG='/mnt/win_d/llm/White-Box/quartic2.log'; T0=time.time()
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

def quad_dirs(W,seed0,nch,CH=2000):
    """напрямки, де живе квадратична кривизна: топ власних Σ_i C_i C_iᵀ.
       y ЦЕНТРУЄТЬСЯ (інакше шум × E[y²]/Var(y) ≈ 20)."""
    # прохід 1 — середнє
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        S+=even(W,z).double().sum(0); n+=CH
    mu=S/n
    # прохід 2 — C на ЦЕНТРОВАНОМУ y
    T=torch.zeros(D,D*D,dtype=torch.float64,device=dev)
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        y=even(W,z).double()-mu
        P=(z.unsqueeze(2)*z.unsqueeze(1)).reshape(CH,-1).double()
        T+=y.t()@P; del P
    C=(T/n).reshape(D,D,D)
    M=torch.einsum('iab,icb->ac',C,C)
    ev,V=torch.linalg.eigh(M.cpu())
    return V.flip(1).to(dev).double(), ev.flip(0)

def he(t,k):
    if k==0: return torch.ones_like(t)
    if k==1: return t
    if k==2: return t*t-1
    if k==3: return t*(t*t-3)
    return t*t*(t*t-6)+3
def quartic_feats(t,K):
    idx=list(itertools.combinations_with_replacement(range(K),4))
    F=torch.empty(t.shape[0],len(idx),dtype=torch.float64,device=dev)
    for j,a in enumerate(idx):
        cnt={}
        for v in a: cnt[v]=cnt.get(v,0)+1
        col=torch.ones(t.shape[0],dtype=torch.float64,device=dev); nrm=1.0
        for v,c in cnt.items():
            col=col*he(t[:,v],c); nrm*=math.factorial(c)
        F[:,j]=col/math.sqrt(nrm)
    return F

def fit_eval(XA,RA,XB,RB,Vk,K,vB):
    FA=quartic_feats(XA@Vk,K); FB=quartic_feats(XB@Vk,K); P=FA.shape[1]
    G=FA.t()@FA/XA.shape[0]+torch.eye(P,dtype=torch.float64,device=dev)*1e-6
    beta=torch.linalg.solve(G, FA.t()@RA/XA.shape[0])
    res=RB-FB@beta
    vq=float((res*res).mean(0).sum())
    del FA,FB,G,beta,res
    return P, max(0.0,(vB-vq)/vB)

for net in (0,1):
    W=load(net); log('='*78); log('МЕРЕЖА %d'%net)
    V,ev=quad_dirs(W,200000,40)
    log('  спектр Σ_i C_iC_iᵀ: топ-5 {}  хвіст {:.3e}'
        .format(' '.join('{:.2e}'.format(float(e)) for e in ev[:5]),float(ev[-1])))
    XA,YA=draw(W,110000,30); XB,YB=draw(W,150000,15)      # 60000 / 30000
    mu=YA.double().mean(0)
    RA=YA.double()-mu; RB=YB.double()-mu
    XA=XA.double(); XB=XB.double()
    vB=float((RB*RB).mean(0).sum())
    log('  підгонка {:d} / перевірка {:d};  парна Var поза вибіркою {:.4e}'
        .format(XA.shape[0],XB.shape[0],vB))
    gr=torch.Generator(device=dev).manual_seed(31337)
    Vr,_=torch.linalg.qr(torch.randn(D,20,device=dev,generator=gr).double())
    log('')
    log('  {:>4s} {:>8s} {:>16s} {:>16s} {:>10s}'
        .format('K','ознак','пояснено (дані)','пояснено (випадк)','сигнал/шум'))
    for K in (4,6,8,10,12,14,16):
        P,f_d=fit_eval(XA,RA,XB,RB,V[:,:K],K,vB)
        _,f_r=fit_eval(XA,RA,XB,RB,Vr[:,:K],K,vB)
        log('  {:4d} {:8d} {:15.4f}  {:15.4f}  {:9.2f}×'
            .format(K,P,f_d,f_r,f_d/max(f_r,1e-9)))
    log('  для порівняння: 2-й порядок = 51-57% (finding-v2-is-exact-order-two)')
log('done.')
