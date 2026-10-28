"""НАСКІЛЬКИ ТОЧНИЙ БЕЗКОШТОВНИЙ ЯКІР ДЛЯ ЗАТВОРІВ?

Затвори дали held-out R²=0.444 (gatespec.py) — найбільший сигнал після слідів (0.70).
Я закрив їх АРГУМЕНТОМ («256 ортантних імовірностей не легші за задачу»), а не виміром.
Урок С44: перевіряти числом навіть «очевидні» теоретичні аргументи.

ЯКІР: p_j = P(h_j^l > 0) = Φ(μ_j/σ_j) з NNGP — БЕЗКОШТОВНО, тим самим проходом,
що вже рахує сліди Tr(K^l).

МІРЯЄМО:
  1. p_NNGP проти p_MC (істина) — відносна похибка по шарах
  2. зсув CV:  bias = Σ_j β_j (p_NNGP − p_MC),  β з регресії a^L на затвори
  3. усадково-оптимальна MSE:  MSE_opt = V₀ − G²/(G+Bs)
     (та сама машинерія, що shrink_cv.py; порогу немає, крива гладка)
⚠️ V₀ = MSE при продакшн-N, НЕ score (пастка С45).
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/gate_anchor.log'
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
TAPS=[7,15,23,27,29,31]

def nngp(W):
    """траєкторія NNGP; повертає p_l = Φ(μ/σ) на TAPS (безкоштовний якір)"""
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev); P={}
    for ell in range(L):
        mu=W[ell].t()@m; C=W[ell].t()@(K-torch.outer(m,m))@W[ell]
        sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t=mu/sd; Ph=Phi(t); ph=phi(t)
        if ell in TAPS: P[ell]=Ph.clone()
        m=mu*Ph+sd*ph
        rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt((1-rho*rho).clamp(min=1e-14))
        Kg=torch.zeros(D,D,dtype=DT,device=dev); sdsq=sd[None,:]*sq
        for a in range(H):
            Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
            Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
            Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
        Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg); K=Kg
    return P

NPROD=200_000; CH=25_000
for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    Wd=[w.double() for w in W]
    log(''); log('######## НЕТ {} ########'.format(net))
    P=nngp(Wd)

    # істина: p_MC на великій вибірці + статистики для CV
    NT=2_000_000
    sg={l:torch.zeros(D,dtype=DT,device=dev) for l in TAPS}; nt=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(5500+net*31+c)
        z=torch.randn(CH//2,D,device=dev,generator=g); x=torch.cat([z,-z],0)
        a=x
        for li in range(L):
            h=a@W[li]
            if li in TAPS: sg[li]+=(h>0).double().sum(0)
            a=relu(h)
        nt+=x.shape[0]
    log('  1) ТОЧНІСТЬ БЕЗКОШТОВНОГО ЯКОРЯ Φ(μ/σ):')
    log('     шар   ||p_NNGP−p_MC||∞   RMS відн.похибка   (сліди для порівняння ~1e-3)')
    pmc={}
    for l in TAPS:
        pmc[l]=sg[l]/nt
        d=P[l]-pmc[l]
        log('     {:2d}      {:.4e}         {:.4e}'.format(
            l, float(d.abs().max()), float((d/pmc[l].clamp(min=1e-9)).pow(2).mean().sqrt())))
log('done.')
