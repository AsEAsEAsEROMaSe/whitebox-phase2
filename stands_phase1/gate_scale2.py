"""Потокова версія: масштабування R² по числу пар + НИЗЬКИЙ РАНГ.
Ознаки НЕ матеріалізуються — накопичуємо лише GᵀG і Gᵀy почанково."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time(); NS=300000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
class Acc:
    def __init__(s,p): s.G=torch.zeros(p,p,dtype=torch.float64,device=dev);s.r=torch.zeros(p,dtype=torch.float64,device=dev);s.Sf=torch.zeros(p,dtype=torch.float64,device=dev);s.Sy=0.;s.yy=0.;s.n=0
    def add(s,F,y):
        s.G+=(F.t()@F).double(); s.r+=(F.t()@y).double(); s.Sf+=F.sum(0).double()
        s.Sy+=float(y.sum()); s.yy+=float((y*y).sum()); s.n+=F.shape[0]
    def c(s): n=s.n; return (s.G-torch.outer(s.Sf,s.Sf)/n)/n,(s.r-s.Sf*s.Sy/n)/n,(s.yy-s.Sy*s.Sy/n)/n
def oos(a1,a2,lam=1e-4):
    G1,r1,_=a1.c(); G2,r2,y2=a2.c()
    b=torch.linalg.solve(G1+lam*float(G1.diagonal().mean())*torch.eye(G1.shape[0],dtype=torch.float64,device=dev),r1)
    return 1-(y2-2*float(b@r2)+float(b@G2@b))/y2

for net in (0,1):
    W=load(net); log('='*70); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g); X=X/X.norm(dim=1,keepdim=True)
    for _ in range(3):
        Y=fwd(W,X); X=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-30)
    vs=(X[0]/X[0].norm())
    Sl=[];Al=[]
    for c in range(NS//CH):
        g2=torch.Generator(device=dev).manual_seed(60000+c)
        x=torch.randn(CH,D,device=dev,generator=g2)
        with torch.no_grad():
            r=0.5*(fwd(W,x)+fwd(W,-x))
            Sl.append(torch.sign(x@W[0]).to(torch.int8)); Al.append(r@vs)
    S=torch.cat(Sl); al=torch.cat(Al).double(); n=S.shape[0]; h=n//2
    al=al-al.mean()
    log('  Var(α_even)={:.4e}, семплів={:d}'.format(float((al*al).mean()),n))
    log('  --- R² out-of-sample vs число пар (уся сім\'я = 32640) ---')
    for NP in (2000,4000,8000,16000,32640):
        gg=torch.Generator(device=dev).manual_seed(11)
        if NP>=32640:
            iu=torch.triu_indices(D,D,offset=1,device=dev); I,J=iu[0],iu[1]
        else:
            I=torch.randint(0,D,(NP,),device=dev,generator=gg); J=torch.randint(0,D,(NP,),device=dev,generator=gg)
            m=I!=J; I,J=I[m],J[m]
        A=[Acc(I.shape[0]),Acc(I.shape[0])]
        for c0 in range(0,n,CH):
            sl=slice(c0,min(c0+CH,n))
            F=(S[sl][:,I]*S[sl][:,J]).float()
            A[0 if c0<h else 1].add(F,al[sl].float()); del F
        try:
            r2=oos(A[0],A[1])
            log('    пар {:6d} ({:5.1f}% сім\'ї)   R² = {:7.4f}'.format(I.shape[0],100*I.shape[0]/32640,r2))
        except Exception as e: log('    пар {:6d}: {}'.format(I.shape[0],e))
        del A
        torch.cuda.empty_cache()
    # НИЗЬКИЙ РАНГ
    Sf=S.float()
    C=((Sf*al.float().unsqueeze(1)).t()@Sf)/n; C=0.5*(C+C.t())
    ev,V=torch.linalg.eigh(C.double()); o=torch.argsort(ev.abs(),descending=True); ev=ev[o];V=V[:,o]
    log('  --- НИЗЬКИЙ РАНГ: ознака (s·v_k)², середнє vᵀAv з арксинуса = БЕЗКОШТОВНЕ ---')
    log('    |λ| топ-6: {}'.format(' '.join('{:.2e}'.format(float(e)) for e in ev[:6].abs())))
    for k in (1,2,4,8,16,32,64,128,256):
        P=(Sf@V[:,:k].float())**2
        A=[Acc(k),Acc(k)]
        A[0].add(P[:h],al[:h].float()); A[1].add(P[h:],al[h:].float())
        r2=oos(A[0],A[1])
        log('    ранг {:3d}   R² = {:7.4f}   ⟹ загальний виграш ≈ {:.4f}×'.format(k,r2,1.0/(1.0-0.6*max(r2,0))))
        del P,A
