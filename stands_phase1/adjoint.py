"""АДЬОЇНТ: чутливість фінального сліду до стану на кожному шарі.

Ідея: повна похибка = sum_s <G_s, dK_s>, де
   G_s  = d Tr(K^L) / d K_s      (спряжений градієнт, БЕЗКОШТОВНО з нашого ж пропагатора)
   dK_s = одношагова похибка замикання на шарі s (матриця 256x256)
Якщо ця розкладка відтворює виміряні внески — маємо ТОЧНУ карту важелів,
і тоді потрібні не повні матриці dK_s, а лише СКАЛЯРИ <G_s,dK_s> на кількох шарах.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
LOG='/mnt/win_d/llm/White-Box/adjoint.log'
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

for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,KT=true_state(W)
    trtrue=np.array([float(torch.diagonal(k).sum()) for k in KT])
    log(''); log('############ NET {} ############'.format(net))

    # ---- градієнт фінального сліду по стану кожного шару (одна зворотна хода) ----
    P=[torch.zeros(D,D,device=dev,requires_grad=True) for _ in range(L)]
    Pm=[torch.zeros(D,device=dev,requires_grad=True) for _ in range(L)]
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev)
    for l in range(L):
        m,K=checkpoint(readout,m,K,W[l],use_reentrant=False)
        m=m+Pm[l]; K=K+P[l]          # точка вставки збурення = стан ПІСЛЯ шару l
    torch.diagonal(K).sum().backward()
    gz=lambda p: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
    G=[gz(p) for p in P]; Gm=[gz(p) for p in Pm]

    # ---- одношагова похибка замикання dK_s (старт з ІСТИННОГО стану) ----
    pred=[]; meas=[]
    with torch.no_grad():
        for s in range(L-1):
            m0=torch.zeros(D,device=dev) if s==0 else mt[s-1].float()
            K0=torch.eye(D,device=dev) if s==0 else KT[s-1].float()
            mg,Kg=readout(m0,K0,W[s])
            dK=(KT[s].float()-Kg); dm=(mt[s].float()-mg)
            # передбачений внесок = -<G,dK> (бо dK = істина - гаус)
            pred.append(-float((G[s]*dK).sum()+(Gm[s]*dm).sum())/trtrue[L-1])
            # виміряний: якір в істину на s проти якоря на s-1
            def prop(st):
                if st==0: mm,KK=torch.zeros(D,device=dev),torch.eye(D,device=dev)
                else:     mm,KK=mt[st-1].float(),KT[st-1].float()
                for l in range(st,L): mm,KK=readout(mm,KK,W[l])
                return float(torch.diagonal(KK).sum())
            meas.append((prop(s)-prop(s+1))/trtrue[L-1])
    pred=np.array(pred); meas=np.array(meas)
    log('шар | адьоїнт-передбач. |  виміряно  | відношення')
    for s in range(L-1):
        log('  {:2d} |    {:+.3e}     | {:+.3e} |  {:6.2f}'.format(s,pred[s],meas[s],
            pred[s]/meas[s] if abs(meas[s])>1e-12 else float('nan')))
    log('corr(передбач,виміряно) = {:.4f}'.format(np.corrcoef(pred,meas)[0,1]))
    log('сума: передбач {:+.3e} | виміряно {:+.3e}'.format(pred.sum(),meas.sum()))
    o=np.argsort(-np.abs(meas))
    cum=np.cumsum(meas[o]**2)/ (meas**2).sum()
    log('КОНЦЕНТРАЦІЯ важеля: топ-1 {:.1%}  топ-3 {:.1%}  топ-5 {:.1%}  топ-10 {:.1%} дисперсії внесків'.format(
        cum[0],cum[2],cum[4],cum[9]))
    log('топ-5 шарів за |внеском|: {}'.format(list(o[:5])))
    ot=np.argsort(-np.abs(pred))
    log('топ-5 шарів за АДЬОЇНТОМ:  {}   (перетин з істинним топ-5: {})'.format(
        list(ot[:5]),len(set(ot[:5].tolist())&set(o[:5].tolist()))))
log('done.')
