"""ОПТИМАЛЬНИЙ ВИБІР ВІДВОДІВ: сигнал ПРОТИ точності якоря.

ЗВУЖЕНА ЗАДАЧА (С46): одношагова похибка замикання 1.55e-4 вже на рівні,
уся біда — накопичення √L з випадковими знаками.
⟹ слід Tr(K^ℓ) на РАННЬОМУ шарі має якір у √(31/ℓ) разів точніший!
   але й сигналу в ньому менше (gatespec: шар 0 дає 0.125, шар 31 — 0.444)

НІКОЛИ НЕ МІРЯЛИ цей компроміс. Відводи [3,7,11,15,19,23,27,30] бралися «розкидано»,
а не оптимізувалися за (виграш R²)/(похибка якоря)².

МІРЯЄМО по КОЖНОМУ шару ℓ:
  1. похибка якоря: |Tr_NNGP(ℓ) − Tr_MC(ℓ)| / Tr_MC(ℓ)     — чи справді росте як √ℓ
  2. сигнал: held-out R² від ознаки ||a^ℓ||² (посемплово)
  3. усадково-оптимальна MSE для РІЗНИХ наборів відводів (ранні / пізні / усі)
     MSE_opt = V₀ − G²/(G+Bs),  Bs = зсув² від похибки якоря
⚠️ V₀ = MSE при продакшн-N (НЕ score). held-out обов'язково.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/tap_tradeoff.log'
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

def nngp_tr(W):
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev); tr=[]
    for ell in range(L):
        mu=W[ell].t()@m; C=W[ell].t()@(K-torch.outer(m,m))@W[ell]
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
        tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

NPROD=200_000; CH=25_000; NT=3_000_000
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    Wd=[w.double() for w in W]
    log(''); log('######## НЕТ {} ########'.format(net))
    trn=nngp_tr(Wd)

    # істина слідів
    st=torch.zeros(L,dtype=DT,device=dev); nt=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(3300+net*29+c)
        z=torch.randn(CH//2,D,device=dev,generator=g); x=torch.cat([z,-z],0)
        a=x
        for li in range(L):
            a=relu(a@W[li]); st[li]+=(a*a).sum()
        nt+=x.shape[0]
    trt=(st/nt).cpu().numpy()
    rel=np.abs(trn-trt)/np.abs(trt)
    log('  1) ПОХИБКА ЯКОРЯ по глибині (чи росте як √ℓ):')
    log('     шар:  ' + ' '.join('{:8d}'.format(l) for l in (0,3,7,11,15,19,23,27,31)))
    log('     відн: ' + ' '.join('{:8.2e}'.format(rel[l]) for l in (0,3,7,11,15,19,23,27,31)))
    nz=rel[1:]/np.sqrt(np.arange(1,L))
    log('     rel/√ℓ (мав би бути ~сталий): мед {:.2e}  розкид {:.2e}..{:.2e}'.format(
        float(np.median(nz)),float(nz.min()),float(nz.max())))

    # 2) сигнал: held-out R² від ||a^ℓ||², покомпонентно
    A=[];T=[]
    for c in range(NPROD//CH):
        g=torch.Generator(device=dev).manual_seed(7700+net*23+c)
        z=torch.randn(CH//2,D,device=dev,generator=g); x=torch.cat([z,-z],0)
        a=x; ts=[]
        for li in range(L):
            a=relu(a@W[li]); ts.append((a*a).sum(1))
        A.append(a.double()); T.append(torch.stack(ts,1).double())
    A=torch.cat(A); T=torch.cat(T); n=A.shape[0]; h=n//2
    mA=A[:h].mean(0); mT=T[:h].mean(0)
    vA=((A[:h]-mA)**2).mean(0); alive=vA>1e-6*float(vA.max())
    def r2_set(idx):
        f1=T[:h][:,idx]-mT[idx]; a1=A[:h]-mA
        S=(f1.t()@f1)/h; b=(f1.t()@a1)/h
        sd=torch.diagonal(S).clamp(min=1e-300).sqrt()
        Sn=S/torch.outer(sd,sd)+1e-8*torch.eye(len(idx),dtype=DT,device=dev)
        beta=(torch.linalg.solve(Sn.cpu(),(b/sd[:,None]).cpu()).to(dev))/sd[:,None]
        f2=T[h:][:,idx]-mT[idx]; a2=A[h:]-mA
        r=a2-f2@beta
        r2=1-(r.var(0)/a2.var(0).clamp(min=1e-30))
        return float(r2[alive].mean()), beta
    log('  2) СИГНАЛ окремого шару (held-out R² від ||a^ℓ||²):')
    for l in (0,3,7,11,15,19,23,27,31):
        r2,_=r2_set([l]); log('     шар {:2d}:  R² = {:+.4f}   похибка якоря {:.2e}'.format(l,r2,rel[l]))

    # 3) набори відводів: усадково-оптимальна MSE
    log('  3) НАБОРИ ВІДВОДІВ (усадка, V₀ = MSE при N={}):'.format(n))
    V0=(vA[alive]/n).mean()
    sets={'ранні 0-7':[0,1,2,3,4,5,6,7], 'середні 12-19':[12,13,14,15,16,17,18,19],
          'пізні 24-31':[24,25,26,27,28,29,30,31], 'чинні 8':[3,7,11,15,19,23,27,30],
          'усі 32':list(range(L))}
    for nm,idx in sets.items():
        r2,beta=r2_set(idx)
        derr=torch.tensor((trn-trt)[idx],dtype=DT,device=dev)
        bias=(beta*derr[:,None]).sum(0)                    # зсув на нейрон
        Bs=float((bias[alive]**2).mean())
        G=float(V0)*r2
        lam=G/max(G+Bs,1e-300)
        mse=float(V0)-G*G/max(G+Bs,1e-300)
        log('     {:14s} R²={:+.4f}  зсув²={:.3e}  λ={:.3f}  MSE {:.4e}  ({:+.2f}%)'.format(
            nm,r2,Bs,lam,mse,100*(mse/float(V0)-1)))
    del A,T; torch.cuda.empty_cache()
log('done.')
