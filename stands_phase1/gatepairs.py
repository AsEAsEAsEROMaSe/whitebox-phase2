"""АРКСИНУСНИЙ ЯКІР — остання названа БЕЗКОШТОВНА сім'я ([[finding-arcsine-gate-anchor]]).
s_i=sign((x·W₀)_i); E[s_i s_j]=(2/π)arcsin(ρ_ij) ТОЧНО й безкоштовно для ВСІХ пар одразу.
sign не поліном ⟹ поза ZCA; s_i s_j ПАРНА ⟹ антитетика не вбиває. Сім'я законно жива.
ЦІЛЬ: скаляр α=a^L·v* (парна частина). ЧЕСНО: коефіцієнти на 1-й половині, R² на 2-й.
ПОТОКОВО: тримаємо лише GᵀG (NP×NP), ознаки не зберігаємо."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NS=300000; NP=4000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
class Acc:
    def __init__(s,p):
        s.G=torch.zeros(p,p,dtype=torch.float64,device=dev); s.r=torch.zeros(p,dtype=torch.float64,device=dev)
        s.Sf=torch.zeros(p,dtype=torch.float64,device=dev); s.Sy=0.0; s.yy=0.0; s.n=0
    def add(s,F,y):
        s.G+=(F.t()@F).double(); s.r+=(F.t()@y).double(); s.Sf+=F.sum(0).double()
        s.Sy+=float(y.sum()); s.yy+=float((y*y).sum()); s.n+=F.shape[0]
    def c(s):
        n=s.n
        return (s.G-torch.outer(s.Sf,s.Sf)/n, s.r-s.Sf*s.Sy/n, s.yy-s.Sy*s.Sy/n)
def oos(a1,a2,lam=1e-4):
    G1,r1,_=a1.c(); G2,r2,y2=a2.c()
    G1=G1/a1.n; r1=r1/a1.n
    b=torch.linalg.solve(G1+lam*float(G1.diagonal().mean())*torch.eye(G1.shape[0],dtype=torch.float64,device=dev),r1)
    sse=y2-2*float(b@r2)+float(b@G2@b)
    return 1-sse/y2

for net in (0,1):
    W=load(net); log('='*72); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g); X=X/X.norm(dim=1,keepdim=True)
    for _ in range(3):
        Y=fwd(W,X); X=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-30)
    vs=(X[0]/X[0].norm())
    gg=torch.Generator(device=dev).manual_seed(11)
    I=torch.randint(0,D,(NP,),device=dev,generator=gg); J=torch.randint(0,D,(NP,),device=dev,generator=gg)
    m=I!=J; I,J=I[m],J[m]
    WR,_=torch.linalg.qr(torch.randn(D,D,device=dev,generator=gg))
    wi=torch.randn(I.shape[0],device=dev,generator=gg)                    # для позит.контролю
    A=[Acc(I.shape[0]) for _ in range(6)]                                 # 0,1=W₀ ; 2,3=випадк ; 4,5=позит
    nch=NS//CH; va=[]
    for c in range(nch):
        g2=torch.Generator(device=dev).manual_seed(60000+c)
        x=torch.randn(CH,D,device=dev,generator=g2)
        with torch.no_grad():
            r=0.5*(fwd(W,x)+fwd(W,-x))
            al=(r@vs)
            F=torch.sign(x@W[0]); F=F[:,I]*F[:,J]
            FRr=torch.sign(x@WR); FRr=FRr[:,I]*FRr[:,J]
        va.append(al)
        k=0 if c<nch//2 else 1
        A[k].add(F,al); A[2+k].add(FRr,al)
        A[4+k].add(F,al+ (F@wi)/math.sqrt(I.shape[0])*0.0)                # заповнимо після оцінки σ
        del F,FRr
    al_all=torch.cat(va); sd=float(al_all.std())
    log('  Var(α_even)={:.4e}; пар={:d}; семплів={:d}'.format(sd*sd,I.shape[0],nch*CH))
    log('  R² out-of-sample, затвори W₀      : {:8.4f}'.format(oos(A[0],A[1])))
    log('  R² out-of-sample, ВИПАДКОВИЙ базис: {:8.4f}   (контроль)'.format(oos(A[2],A[3])))
    # позитивний контроль окремим проходом
    P=[Acc(I.shape[0]) for _ in range(2)]
    for c in range(nch):
        g2=torch.Generator(device=dev).manual_seed(60000+c)
        x=torch.randn(CH,D,device=dev,generator=g2)
        with torch.no_grad():
            r=0.5*(fwd(W,x)+fwd(W,-x)); al=(r@vs)
            F=torch.sign(x@W[0]); F=F[:,I]*F[:,J]
            inj=F@wi
        inj=inj/inj.std()*sd*math.sqrt(0.05)
        P[0 if c<nch//2 else 1].add(F,al+inj); del F
    log('  ПОЗ.КОНТРОЛЬ підсадка 5% тими ж ознаками ⟹ R²={:8.4f}  (очік ≳0.05)'.format(oos(P[0],P[1])))
