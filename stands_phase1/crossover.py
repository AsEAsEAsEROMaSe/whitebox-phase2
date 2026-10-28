"""ГЛИБИНА ПЕРЕТИНУ: де аналітика програє Монте-Карло.

relu_ladder.py показав: відносна MSE методу v2 ПЛАСКА по глибині (2e-7..7e-7),
причому k=1 відносно НАЙГІРШИЙ, k=32 у 10× кращий. Глибина не карає MC.
А похибку замикання глибина карає як √k (виміряно тричі).

⟹ при k=1 аналітика ТОЧНА (один шар ReLU з гаусовим входом = закрита формула,
   NNGP там не наближення а тотожність, похибка рівно 0)
⟹ існує ГЛИБИНА ПЕРЕТИНУ, до якої аналітика виграє. Ніколи не міряли.

МІРЯЄМО для мережі з ReLU на ПЕРШИХ k шарах (решта лінійні), відносну MSE:
   - v2 (ZCA + антитетика), N=200k
   - NNGP-аналітика (пропагація m,K крізь k шарів, далі лінійний хвіст ТОЧНО)
Обидва проти істини з 4М семплів. Відносна MSE = MSE / mean(E[a^L]²).
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/crossover.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=20
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi); DT=torch.float64
relu=lambda x: torch.clamp(x,min=0.0)
from numpy.polynomial.hermite import hermgauss
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=DT,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=DT,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))

def fwd(W,x,k):
    a=x
    for li in range(L):
        a=a@W[li]
        if li<k: a=relu(a)
    return a

def truth(W,k,N=4_000_000,CH=200_000,seed=770):
    s=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=g); x=torch.cat([z,-z],0)
        s+=fwd(W,x,k).double().sum(0); n+=x.shape[0]
    return s/n

def v2(W,k,npair,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g); x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    return fwd(W,x@Wh,k).double().mean(0)

def nngp(Wd,k):
    """m,K крізь k шарів ReLU, далі лінійний хвіст ТОЧНО (E[xM]=E[x]M)"""
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev)
    for ell in range(k):
        mu=Wd[ell].t()@m; C=Wd[ell].t()@(K-torch.outer(m,m))@Wd[ell]
        sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t=mu/sd; Ph=Phi(t); ph=phi(t)
        m=mu*Ph+sd*ph
        rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt((1-rho*rho).clamp(min=1e-14))
        Kg=torch.zeros(D,D,dtype=DT,device=dev); sdsq=sd[None,:]*sq
        for a in range(H):
            Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
            Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
            Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
        Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg); K=Kg
    for ell in range(k,L): m=Wd[ell].t()@m          # лінійний хвіст: точно
    return m

NPAIR=100_000; REPS=8
KS=[1,2,3,4,5,6,8,10,12,16,20,24,28,32]
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    Wd=[w.double() for w in W]
    log(''); log('######## НЕТ {} ########'.format(net))
    log('   k    відн.MSE v2      відн.MSE NNGP     хто виграє   у скільки разів')
    for k in KS:
        TH=truth(W,k,seed=770+net*97+k)
        p=float((TH*TH).mean())
        errs=[float(((v2(W,k,NPAIR,1200+net*53+k*7+r)-TH)**2).mean()) for r in range(REPS)]
        mv=float(np.mean(errs))/p
        mn=float((((nngp(Wd,k)-TH)**2).mean()))/p
        who='NNGP' if mn<mv else 'v2'
        log('  {:2d}    {:.4e}      {:.4e}      {:6s}     {:8.2f}×'.format(
            k, mv, mn, who, max(mv,mn)/max(min(mv,mn),1e-300)))
log('done.')
