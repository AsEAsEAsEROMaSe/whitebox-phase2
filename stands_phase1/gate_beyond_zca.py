"""РІШУЧИЙ КОНТРОЛЬ: чи лишається арксинусний CV живим ПОНАД ZCA?
v2 знімає 2-й порядок ТОЧНО. Тому міряти треба на ЗАЛИШКУ після зняття квадратики —
і з ЦІЛІ, і з ОЗНАК (бо (s·v)² сама має квадратичну складову).
Оптимальна квадратика для y: q(x)=xᵀAx−tr(A), A=C/2, C=E[y(xxᵀ−I)]  (див. orders2.py).
ЧЕСНО: C і коефіцієнти — на 1-й половині, R² — на 2-й."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NS=300000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
def deq(Xh,Y):
    """прибрати найкращу квадратику: A=C/2, C=E[y(xxᵀ−I)]; повертає функцію-застосовувач"""
    n=Xh.shape[0]; Y=Y-Y.mean(0,keepdim=True)
    C=torch.einsum('ni,nj,nk->ijk',Xh,Xh,Y)/n if False else None
    # ефективніше: для кожної цілі окремо  C_c = Xᵀ diag(y_c) X /n − I*mean(y_c)
    out=[]
    for c in range(Y.shape[1]):
        yc=Y[:,c]
        Cc=(Xh.t()@(Xh*yc.unsqueeze(1)))/n
        Cc=Cc-torch.eye(D,device=dev,dtype=Xh.dtype)*float(yc.mean())
        out.append(Cc*0.5)
    return torch.stack(out)          # ncoord × D × D  (це A)
def applyq(Xh,A):
    """q_c(x) = xᵀA_c x − tr(A_c)"""
    import torch as _t
    out=_t.empty(Xh.shape[0],A.shape[0],dtype=Xh.dtype,device=Xh.device)
    for c in range(A.shape[0]):
        out[:,c]=((Xh@A[c])*Xh).sum(1)-A[c].diagonal().sum()
    return out

for net in (0,1):
    W=load(net); log('='*74); log('МЕРЕЖА %d'%net)
    Sl=[];Rl=[];Xl=[]
    for c in range(NS//CH):
        g2=torch.Generator(device=dev).manual_seed(60000+c)
        x=torch.randn(CH,D,device=dev,generator=g2)
        with torch.no_grad(): Rl.append(0.5*(fwd(W,x)+fwd(W,-x)))
        Sl.append(torch.sign(x@W[0]).to(torch.int8)); Xl.append(x)
    S=torch.cat(Sl).float(); R=torch.cat(Rl).double(); Xh=torch.cat(Xl); n=S.shape[0]; h=n//2
    R=R-R.mean(0,keepdim=True)
    # звужуємось до 24 випадкових координат виходу (щоб A влізла), зважимо дисперсією
    gg=torch.Generator(device=dev).manual_seed(7)
    cid=torch.randperm(D,device=dev,generator=gg)[:24]
    Y=R[:,cid]
    K0=(W[0].t()@W[0]).double(); dd=K0.diagonal().sqrt()
    Am=(2/math.pi)*torch.arcsin((K0/torch.outer(dd,dd)).clamp(-1,1))
    # напрямки V (як у gate_lowrank)
    M=torch.zeros(D,D,dtype=torch.float64,device=dev)
    for t in range(24):
        u=torch.randn(D,device=dev,generator=gg); u=u/u.norm()
        y=(R@u.double()).float(); Cm=((S*y.unsqueeze(1)).t()@S)/n; Cm=0.5*(Cm+Cm.t()).double(); M+=Cm@Cm
    ev,V=torch.linalg.eigh(M); V=V.flip(1)
    def r2(Ytr,Yte,Ptr,Pte):
        G=(Ptr.t()@Ptr)/Ptr.shape[0]; r=(Ptr.t()@Ytr)/Ptr.shape[0]
        B=torch.linalg.solve(G+1e-6*float(G.diagonal().mean())*torch.eye(G.shape[0],dtype=torch.float64,device=dev),r)
        res=Yte-Pte@B
        return 1-float((res*res).mean(0).sum())/float((Yte*Yte).mean(0).sum())
    for k in (8,32,128):
        Vk=V[:,:k].float()
        P=((S@Vk)**2).double()
        mu=torch.stack([(Vk[:,j].double()@Am@Vk[:,j].double()) for j in range(k)])
        P=P-mu.unsqueeze(0)
        a=r2(Y[:h],Y[h:],P[:h],P[h:])
        # --- ПОНАД ZCA: зняти квадратику з ЦІЛІ і з ОЗНАК (коефіцієнти з 1-ї половини) ---
        Aq=deq(Xh[:h].double(),Y[:h]); Yr=Y-applyq(Xh.double(),Aq)
        Aqp=deq(Xh[:h].double(),P[:h]); Pr=P-applyq(Xh.double(),Aqp)
        b=r2(Yr[:h],Yr[h:],Pr[:h],Pr[h:])
        log('  ранг {:3d}:  R² сирий = {:7.4f}   R² ПОНАД ZCA = {:7.4f}'.format(k,a,b))
        del P,Pr,Aq,Aqp,Yr
        torch.cuda.empty_cache()
