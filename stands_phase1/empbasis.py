"""Чи можна дістати підпростір похибки замикання з ЕМПІРИЧНОГО залишку?

K_hat (з N семплів) - K_гаус = dK + шум.
Шум на елементах ~ 1/sqrt(N) (при N=200k це 2.2e-3), сигнал dK ~1e-4 -> шум у 20x більший.
АЛЕ сигнал сидить у ~20 вимірах, шум розмазаний по 256. Хто виграє — вимір, не суперечка.

Міряємо: частку <G_s,dK_s>, схоплену топ-16 власних векторів ЕМПІРИЧНОГО залишку,
проти оракульного базису (стеля) і випадкового (контроль).
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
LOG='/mnt/win_d/llm/White-Box/empbasis.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=24
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))

def readout(m,K,Wl):
    mu=Wl.t()@m; C=Wl.t()@(K-torch.outer(m,m))@Wl
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12)); t=mu/sd; Ph=Phi(t); ph=phi(t)
    mn=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
    Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
    for a in range(H):
        Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
    Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg)
    return mn,Kg

def samples_K(W,upto,N,seed0,CH=50_000):
    """емпірична K_hat на шарі upto з N семплів (антитетичні пари)"""
    S=torch.zeros(D,D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(seed0+c))
        x=torch.cat([z,-z],0)
        for li in range(upto+1): x=relu(x@W[li])
        xd=x.double(); S+=xd.t()@xd; n+=x.shape[0]
    return S/n

def true_state(W,N=1_200_000,CH=60_000):
    s1=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    SK=[torch.zeros(D,D,dtype=torch.float64,device=dev) for _ in range(L)]
    n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(12000+c))
        x=torch.cat([z,-z],0)
        for li in range(L):
            x=relu(x@W[li]); xd=x.double()
            s1[li]+=xd.sum(0); SK[li]+=xd.t()@xd
        n+=x.shape[0]
    return [s/n for s in s1],[s/n for s in SK]

def topvecs(M,r):
    M=0.5*(M+M.t()); ev,U=torch.linalg.eigh(M)
    o=torch.argsort(-ev.abs()); return U[:,o[:r]]

R=16
NS=[50_000,200_000,800_000]
agg={'оракул dK (стеля)':[], 'випадковий':[]}
for N in NS: agg['емпіричний N={}'.format(N)]=[]
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,KT=true_state(W)
    trL=float(torch.diagonal(KT[L-1]).sum())
    P=[torch.zeros(D,D,device=dev,requires_grad=True) for _ in range(L)]
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev)
    for l in range(L):
        m,K=checkpoint(readout,m,K,W[l],use_reentrant=False); K=K+P[l]
    torch.diagonal(K).sum().backward()
    G=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in P]
    log(''); log('######## NET {} ########'.format(net))
    with torch.no_grad():
        dKs=[];KGs=[]
        for s in range(L-1):
            m0=torch.zeros(D,device=dev) if s==0 else mt[s-1].float()
            K0=torch.eye(D,device=dev) if s==0 else KT[s-1].float()
            _,Kg=readout(m0,K0,W[s]); KGs.append(Kg.double().cpu()); dKs.append((KT[s].float()-Kg).double().cpu())
        Gc=[g.double().cpu() for g in G]
        c=np.array([-float((Gc[s]*dKs[s]).sum())/trL for s in range(L-1)])
        top=np.argsort(-np.abs(c))[:3]
        gen=torch.Generator(device=dev).manual_seed(11)
        for s in top:
            full=float((Gc[s]*dKs[s]).sum())
            def cap(U):
                Pr=U@U.t(); return float(((Pr@Gc[s]@Pr)*dKs[s]).sum())/full
            line=[]
            v=cap(topvecs(dKs[s],R)); agg['оракул dK (стеля)'].append(v); line.append('оракул {:+.2f}'.format(v))
            for N in NS:
                Kh=samples_K(W,int(s),N,int(900_000+net*1000+s*10))
                resid=Kh.cpu()-KGs[s]
                # чистота залишку: скільки в ньому сигналу проти шуму
                sn=float((resid*dKs[s]).sum()/(resid.norm()*dKs[s].norm()))
                v=cap(topvecs(resid,R)); agg['емпіричний N={}'.format(N)].append(v)
                line.append('N={}k {:+.2f} (cos={:+.3f})'.format(N//1000,v,sn))
            v=cap(topvecs(torch.randn(D,D,generator=torch.Generator().manual_seed(int(s)),dtype=torch.float64),R))
            agg['випадковий'].append(v); line.append('вип {:+.2f}'.format(v))
            log('  шар {:2d} (внесок {:+.2e}): {}'.format(s,c[s],' | '.join(line)))
log('')
log('СЕРЕДНЄ — частка схопленого <G,dK> при ранзі {}:'.format(R))
for k,v in agg.items():
    if v: log('   {:22s} {:+.3f}'.format(k,np.mean(v)))
log('done.')
