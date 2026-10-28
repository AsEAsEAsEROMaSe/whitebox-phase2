"""ПЕРЕХРЕСНА ПЕРЕВІРКА картини стиснення.

lyap.py: відображення по глибині стискає з множником c ~ 0.87-0.88 за шар.
Якщо це правда, то чутливість фіналу до стану на шарі s мусить спадати НАЗАД
по глибині рівно як  ||G_s|| ~ c^(L-s).

G_s = d Tr(K^L)/d K_s -- та сама величина, що в adjoint.py (corr 0.99 з істиною),
тобто перевірка йде через ІНШУ машинерію (зворотна хода) на ту саму фізику.

Якщо нахил збігається -> картина замкнена трьома незалежними вимірами.
Якщо ні -> щось у ній не так, і краще дізнатись зараз.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
LOG='/mnt/win_d/llm/White-Box/gdecay.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=20; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=DT,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=DT,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))

def readout(m,K,Wl):
    mu=Wl.t()@m; C=Wl.t()@(K-torch.outer(m,m))@Wl
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t=mu/sd; Ph=Phi(t); ph=phi(t)
    mn=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt((1-rho*rho).clamp(min=1e-14))
    Kg=torch.zeros(D,D,dtype=DT,device=dev); sdsq=sd[None,:]*sq
    for a in range(H):
        Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
    Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg)
    return mn,Kg

gz=lambda p: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float64),device=dev,dtype=DT) for i in range(L)]
    P =[torch.zeros(D,D,dtype=DT,device=dev,requires_grad=True) for _ in range(L)]
    Pm=[torch.zeros(D,  dtype=DT,device=dev,requires_grad=True) for _ in range(L)]
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev)
    for l in range(L):
        m=m+Pm[l]; K=K+P[l]
        m,K=checkpoint(readout,m,K,W[l],use_reentrant=False)
    torch.diagonal(K).sum().backward()
    gK=np.array([float(gz(P[l]).norm())  for l in range(L)])
    gm=np.array([float(gz(Pm[l]).norm()) for l in range(L)])
    log(''); log('#### НЕТ {} ####'.format(net))
    log('  ||G_s|| (K-частина) на шарах 0,7,15,23,29,31:')
    log('     ' + ' '.join('{:.3e}'.format(gK[i]) for i in (0,7,15,23,29,31)))
    # нахил у ЛОГАРИФМІ по глибині -> множник за шар
    sel=np.arange(2,L-1)                      # без країв
    y=np.log(gK[sel].clamp if False else np.log(np.maximum(gK[sel],1e-300)))
    A=np.polyfit(sel,y,1)
    log('  множник за шар з нахилу ||G||:  c = {:.4f}'.format(math.exp(A[0])))
    log('  (lyap.py дав медіану 0.87-0.88)')
    # чесніше: локальні відношення
    r=gK[1:]/np.maximum(gK[:-1],1e-300)
    log('  локальні відношення G_{s+1}/G_s: мед {:.4f}   міжкварт {:.4f}..{:.4f}'.format(
        float(np.median(r)),float(np.percentile(r,25)),float(np.percentile(r,75))))
    log('  1/c => ефективна пам\'ять {:.1f} шарів; частка бюджету похибки в останніх 10 шарах:'.format(
        1/(1-math.exp(A[0]))))
    w=gK/gK.sum()
    log('     останні 10 шарів несуть {:.1%} суми ||G||,   перші 16 -> {:.1%}'.format(
        float(w[-10:].sum()), float(w[:16].sum())))
log('done.')
