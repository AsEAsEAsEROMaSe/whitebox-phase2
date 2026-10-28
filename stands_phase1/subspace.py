"""Чи СТАБІЛЬНИЙ підпростір похибки замикання по глибині?

Якщо базис dK_s можна взяти з dK_{s-k} (перенесеного вперед вагами), то один оракульний
вимір на мілкому шарі дає базис для всіх наступних, і дешева проєкція повертається.

Метрика операційна: яку частку <G_s, dK_s> схоплює проєкція на базис із ПОПЕРЕДНЬОГО шару.
Плюс геометричне перекриття підпросторів.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
LOG='/mnt/win_d/llm/White-Box/subspace.log'
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
res={'той самий шар (стеля)':[], 'з s-1 сирий':[], 'з s-1 перенесений':[],
     'з s-2 перенесений':[], 'з s-4 перенесений':[]}
ovl={'s-1 сирий':[], 's-1 перенесений':[], 's-2 перенесений':[], 's-4 перенесений':[]}
for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,KT=true_state(W)
    trL=float(torch.diagonal(KT[L-1]).sum())
    P=[torch.zeros(D,D,device=dev,requires_grad=True) for _ in range(L)]
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev)
    for l in range(L):
        m,K=checkpoint(readout,m,K,W[l],use_reentrant=False); K=K+P[l]
    torch.diagonal(K).sum().backward()
    G=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in P]
    with torch.no_grad():
        dKs=[]
        for s in range(L-1):
            m0=torch.zeros(D,device=dev) if s==0 else mt[s-1].float()
            K0=torch.eye(D,device=dev) if s==0 else KT[s-1].float()
            _,Kg=readout(m0,K0,W[s]); dKs.append((KT[s].float()-Kg).double().cpu())
        Gc=[g.double().cpu() for g in G]
        Wc=[w.double().cpu() for w in W]
        c=np.array([-float((Gc[s]*dKs[s]).sum())/trL for s in range(L-1)])
        top=np.argsort(-np.abs(c))[:5]
        log(''); log('######## NET {} ########  (ранг r={}, топ-5 шарів за внеском: {})'.format(net,R,sorted(top)))
        for s in top:
            if s<4: continue
            full=float((Gc[s]*dKs[s]).sum())
            def cap(U):
                Pr=U@U.t(); return float(((Pr@Gc[s]@Pr)*dKs[s]).sum())/full
            Us=topvecs(dKs[s],R)
            bases={'той самий шар (стеля)':Us,
                   'з s-1 сирий':topvecs(dKs[s-1],R),
                   'з s-1 перенесений':topvecs(Wc[s].t()@dKs[s-1]@Wc[s],R),
                   'з s-2 перенесений':topvecs(Wc[s].t()@Wc[s-1].t()@dKs[s-2]@Wc[s-1]@Wc[s],R),
                   'з s-4 перенесений':topvecs(Wc[s].t()@Wc[s-1].t()@Wc[s-2].t()@Wc[s-3].t()@dKs[s-4]
                                               @Wc[s-3]@Wc[s-2]@Wc[s-1]@Wc[s],R)}
            line=[]
            for bn,U in bases.items():
                v=cap(U); res[bn].append(v); line.append('{} {:+.2f}'.format(bn,v))
                if bn!='той самий шар (стеля)':
                    ovl[bn.replace('з ','')].append(float((Us.t()@U).norm()**2)/R)
            log('  шар {:2d} (внесок {:+.2e}): {}'.format(s,c[s],' | '.join(line)))
log('')
log('СЕРЕДНЄ — частка схопленого <G,dK> при ранзі {}:'.format(R))
for k,v in res.items():
    if v: log('   {:24s} {:+.3f}   (n={})'.format(k,np.mean(v),len(v)))
log('ГЕОМЕТРИЧНЕ перекриття підпросторів (1.0 = збіг, {:.3f} = випадкові):'.format(R/D))
for k,v in ovl.items():
    if v: log('   {:24s} {:.3f}'.format(k,np.mean(v)))
log('done.')
