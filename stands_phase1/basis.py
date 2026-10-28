"""Правильний базис для <G_s, dK_s>.

Гіпотеза: dK_s (похибка замикання) низькорангова І лежить у провідних власних векторах
ВІДОМОЇ нам K_gauss_s. Тоді проєкція обох на топ-r власних векторів K_gauss схопить скаляр,
і дешевий шлях (r*d на семпл замість d^2) повертається.

Перевіряємо чотири базиси: власні K_gauss (ВІДОМИЙ), власні dK (оракул), власні G (оракул), випадковий.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
LOG='/mnt/win_d/llm/White-Box/basis.log'
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

RS=[1,2,4,8,16,32,64,128]
acc={b:{r:[] for r in RS} for b in ['K_гаус (ВІДОМИЙ)','dK (оракул)','G (оракул)','випадковий']}
prd={r:[] for r in RS}
for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,KT=true_state(W)
    trL=float(torch.diagonal(KT[L-1]).sum())
    P=[torch.zeros(D,D,device=dev,requires_grad=True) for _ in range(L)]
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); KG=[]
    for l in range(L):
        m,K=checkpoint(readout,m,K,W[l],use_reentrant=False); KG.append(K.detach().clone()); K=K+P[l]
    torch.diagonal(K).sum().backward()
    G=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in P]
    log(''); log('######## NET {} ########'.format(net))
    with torch.no_grad():
        # PR похибки замикання по шарах
        prs=[]
        dKs=[]
        for s in range(L-1):
            m0=torch.zeros(D,device=dev) if s==0 else mt[s-1].float()
            K0=torch.eye(D,device=dev) if s==0 else KT[s-1].float()
            _,Kg=readout(m0,K0,W[s]); dK=(KT[s].float()-Kg); dKs.append(dK)
            ev=torch.linalg.eigvalsh(0.5*(dK+dK.t()).double().cpu())
            prs.append(float((ev.abs().sum()**2/(ev**2).sum())/D))
        log('PR(dK) по шарах: сер {:.3f}  мін {:.3f}  макс {:.3f}   (1.0 = повний ранг)'.format(
            np.mean(prs),np.min(prs),np.max(prs)))
        # для трьох найважливіших шарів — точність проєкції в кожному базисі
        c=np.array([-float((G[s]*dKs[s]).sum())/trL for s in range(L-1)])
        top=np.argsort(-np.abs(c))[:3]
        gen=torch.Generator(device=dev).manual_seed(7)
        for s in top:
            full=float((G[s]*dKs[s]).sum())
            bases={'K_гаус (ВІДОМИЙ)':KG[s],'dK (оракул)':dKs[s],'G (оракул)':G[s],
                   'випадковий':torch.randn(D,D,device=dev,generator=gen)}
            line={}
            for bn,M in bases.items():
                for r in RS:
                    U=topvecs(M.double().cpu(),r)
                    Pr=U@U.t()
                    v=float((( Pr@G[s].double().cpu()@Pr)*dKs[s].double().cpu()).sum())/full
                    acc[bn][r].append(v); line.setdefault(bn,[]).append(v)
            log(' шар {:2d} (внесок {:+.2e}):'.format(s,c[s]))
            for bn in bases: log('   {:18s} {}'.format(bn,'  '.join('r{}={:+.2f}'.format(r,v) for r,v in zip(RS,line[bn]))))
log('')
log('СЕРЕДНЄ по 3 нетах x 3 шари — частка схопленого <G,dK>:')
for bn in acc:
    log('  {:18s} {}'.format(bn,'  '.join('r{}={:+.2f}'.format(r,np.mean(acc[bn][r])) for r in RS)))
log('done.')
