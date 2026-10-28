"""ВИРІШАЛЬНИЙ тест: чи б'є скоригований NNGP просте MC на ТИХ САМИХ семплах?

Корекція: Tr_випр = Tr_nngp - sum_s <G_s, dK_s>,  dK_s = K_hat_s - f_замикання(K_hat_{s-1})
          усе з N семплів, які й так крутяться.
Порівняння на ТІЙ САМІЙ вибірці:
   Tr_nngp   — аналітика, зсув ~1e-3, дисперсія 0
   Tr_mc     — просте середнє ||a||^2, зсуву 0, дисперсія ~1/N   (це і є "MC-якір")
   Tr_випр   — гібрид
Плюс ранг-16 ескіз замість повного рангу (єдина версія, що влазить у бюджет).
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
LOG='/mnt/win_d/llm/White-Box/correct.log'
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
TAPS=[3,7,11,15,19,23,27,30]
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

def empirical(W,N,seed0,CH=50_000):
    """K_hat і m_hat на ВСІХ шарах з N семплів (антитетичні пари)"""
    S=[torch.zeros(D,D,dtype=torch.float64,device=dev) for _ in range(L)]
    s1=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(seed0+c))
        x=torch.cat([z,-z],0)
        for li in range(L):
            x=relu(x@W[li]); xd=x.double()
            S[li]+=xd.t()@xd; s1[li]+=xd.sum(0)
        n+=x.shape[0]
    return [s/n for s in s1],[s/n for s in S]

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    log(''); log('######## NET {} ########'.format(net))
    mT,KT=empirical(W,12_000_000,500_000)          # ІСТИНА (12M)
    truth=np.array([float(torch.diagonal(KT[t]).sum()) for t in TAPS])

    # адьоїнти для КОЖНОГО з 8 відводів
    ADJ={}
    for t in TAPS:
        P=[torch.zeros(D,D,device=dev,requires_grad=True) for _ in range(t+1)]
        Pm=[torch.zeros(D,device=dev,requires_grad=True) for _ in range(t+1)]
        m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); KG=[]
        for l in range(t+1):
            m,K=checkpoint(readout,m,K,W[l],use_reentrant=False); KG.append(K.detach().clone())
            m=m+Pm[l]; K=K+P[l]
        torch.diagonal(K).sum().backward()
        gz=lambda p: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
        ADJ[t]=([gz(p) for p in P],KG,[gz(p) for p in Pm])
    nngp=np.array([float(torch.diagonal(ADJ[t][1][t]).sum()) for t in TAPS])

    for N in [200_000,800_000]:
        mh,Kh=empirical(W,N,777_000+net*100)
        mc=np.array([float(torch.diagonal(Kh[t]).sum()) for t in TAPS])
        with torch.no_grad():
            # одношагова похибка замикання з ЕМПІРИЧНОГО стану
            dK=[];dm=[]
            for s in range(L):
                m0=torch.zeros(D,device=dev) if s==0 else mh[s-1].float()
                K0=torch.eye(D,device=dev) if s==0 else Kh[s-1].float()
                mg,Kg=readout(m0,K0,W[s])
                dK.append(Kh[s].float()-Kg); dm.append(mh[s].float()-mg)
            corr_full=[];corr_r16=[]
            gen=torch.Generator(device=dev).manual_seed(3)
            for t in TAPS:
                G,_,Gm=ADJ[t]
                cf=sum(float((G[s]*dK[s]).sum()+(Gm[s]*dm[s]).sum()) for s in range(t+1))
                # ранг-16 ескіз залишку (виробнича версія)
                cr=0.0
                for s in range(t+1):
                    Om=torch.randn(D,16,device=dev,generator=gen)
                    Y=dK[s]@Om; Q,_=torch.linalg.qr(Y.double().cpu())
                    Q=Q.float().to(dev); Pr=Q@Q.t()
                    cr+=float(((Pr@G[s]@Pr)*dK[s]).sum()+(Gm[s]*dm[s]).sum())
                corr_full.append(cf); corr_r16.append(cr)
        cf=nngp+np.array(corr_full); cr=nngp+np.array(corr_r16)
        r=lambda v: np.sqrt((((v-truth)/truth)**2).mean())
        log('  N={:>7}:  NNGP {:.3e} | MC {:.3e} | випр.повний {:.3e} | випр.ранг16 {:.3e}'.format(
            N,r(nngp),r(mc),r(cf),r(cr)))
        log('           пошарово MC : {}'.format(' '.join('{:+.1e}'.format(v) for v in (mc-truth)/truth)))
        log('           пошарово випр: {}'.format(' '.join('{:+.1e}'.format(v) for v in (cf-truth)/truth)))
log('done.')
