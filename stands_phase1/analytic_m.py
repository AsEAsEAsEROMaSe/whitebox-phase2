"""ВИРІШАЛЬНИЙ ВИМІР: чи дає АНАЛІТИЧНЕ m достатню точність на глибині?

Абляція (lowrank_ablate.py) показала: K ні до чого, вся стіна — m.
Аналітика (кумулянтна пропагація) дає m БЕЗ ЖОДНОГО СЕМПЛА — єдине джерело m, яке не
платить ([[finding-anchor-cost-identity]]). Питання лише в точності.

Пропагація з analytic_K2.py (квадратура ТОЧНА: внутрішній інтеграл замкнено, зовнішній
розрізано на зламі; самоперевірка проти ядра арккосинуса 1e-10).

ТЕСТ: синтетичний старт на шарі 31 з
  A) аналітичне m + оцінена K (пілот 1000)      ← БЕЗКОШТОВНЕ m
  B) істинне m + оцінена K                       ← стеля
  C) оцінене m (пілот 1000) + оцінена K          ← те, що маємо зараз
"""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; NG=64; T0=time.time(); M=200000; CH=20000
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def pre_at(W,x,l):
    a=x
    for i in range(l): a=torch.relu(a@W[i])
    return a@W[l]
def from_pre(W,h,l):
    a=torch.relu(h)
    for i in range(l+1,L): a=torch.relu(a@W[i])
    return a
_p,_w=np.polynomial.legendre.leggauss(NG)
GP=torch.tensor((_p+1)/2,dtype=torch.float64,device=dev); GW=torch.tensor(_w/2,dtype=torch.float64,device=dev)
SQ2PI=math.sqrt(2*math.pi); RANGE=14.0
def _Phi(x): return 0.5*(1+torch.erf(x/math.sqrt(2)))
def _phi(x): return torch.exp(-0.5*x*x)/SQ2PI
def relu_moments(m,C):
    s=C.diagonal().clamp(min=1e-20).sqrt(); z=m/s
    mu=m*_Phi(z)+s*_phi(z)
    rho=(C/torch.outer(s,s)).clamp(-1+1e-10,1-1e-10)
    E2=torch.empty(D,D,dtype=torch.float64,device=dev)
    for a0 in range(0,D,8):
        mi=m[a0:a0+8].view(-1,1,1); si=s[a0:a0+8].view(-1,1,1)
        u=(-mi/si)+RANGE*GP.view(1,1,NG)
        r=rho[a0:a0+8].view(-1,D,1); mj=m.view(1,D,1); sj=s.view(1,D,1)
        a=mj+sj*r*u; b=(sj*torch.sqrt(1-r*r)).clamp(min=1e-12)
        E2[a0:a0+8]=(( (mi+si*u)*_phi(u)*(a*_Phi(a/b)+b*_phi(a/b)) )*GW.view(1,1,NG)).sum(2)*RANGE
    return mu,E2-torch.outer(mu,mu)
l=31
for net in (0,1):
    W=load(net); log('='*82); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(20):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(50000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=from_pre(W,pre_at(W,xx,0),0).double().sum(0); n+=xx.shape[0]
    TH=S/n
    Sm=torch.zeros(D,dtype=torch.float64,device=dev); m0=0
    for c in range(20):
        g=torch.Generator(device=dev).manual_seed(660000+c)
        z=torch.randn(50000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        Sm+=pre_at(W,xx,l).double().sum(0); m0+=xx.shape[0]
    mT=Sm/m0
    # аналітичний прохід
    ta=time.time(); W0=W[0].double()
    mA=torch.zeros(D,dtype=torch.float64,device=dev); CA=W0.t()@W0
    for i in range(0,L-1):
        mu,Sg=relu_moments(mA,CA); Wi=W[i+1].double()
        mA=mu@Wi; CA=Wi.t()@Sg@Wi
    log('  аналітичний прохід {:.1f}с; ‖m_анал − m_іст‖/‖m_іст‖ = {:.3f}%'
        .format(time.time()-ta,100*float((mA-mT).norm()/mT.norm())))
    # пілот
    P=1000; hs=[];done=0
    while done<P:
        b=min(CH,P-done)
        g=torch.Generator(device=dev).manual_seed(70000+done)
        z=torch.randn(b//2,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        hs.append(pre_at(W,xx,l).double()); done+=b
    H=torch.cat(hs); mh=H.mean(0); Hc=H-mh; Kh=(Hc.t()@Hc)/H.shape[0]
    log('    ‖m_пілот − m_іст‖/‖m_іст‖ = {:.3f}%  (для порівняння)'.format(100*float((mh-mT).norm()/mT.norm())))
    ev,V=torch.linalg.eigh(Kh); ev=ev.flip(0).clamp(min=0); V=V.flip(1)
    lam=ev[:32]; U=V[:,:32]; resid=max(float(ev.sum())-float(lam.sum()),0.)/D
    def run(mm,tag):
        Ss=torch.zeros(D,dtype=torch.float64,device=dev); k=0
        for c in range(M//CH):
            g=torch.Generator(device=dev).manual_seed(31000+c)
            u=torch.randn(CH//2,32,device=dev,generator=g,dtype=torch.float64); uu=torch.cat([u,-u],0)
            h=mm.unsqueeze(0)+(uu*lam.sqrt().unsqueeze(0))@U.t()
            e=torch.randn(CH//2,D,device=dev,generator=g,dtype=torch.float64)
            h=h+torch.cat([e,-e],0)*math.sqrt(resid)
            Ss+=from_pre(W,h.float(),l).double().sum(0); k+=uu.shape[0]
        b=float(((Ss/k-TH)**2).mean())
        N=(6485.0-P)*32.0
        log('    {:34s} зсув {:.3e}   score {:.3e}'.format(tag,b,0.10*(b+0.035/N)))
    run(mA,'A) АНАЛІТИЧНЕ m (безкоштовне)')
    run(mT,'B) істинне m (стеля)')
    run(mh,'C) оцінене m, пілот 1000')
