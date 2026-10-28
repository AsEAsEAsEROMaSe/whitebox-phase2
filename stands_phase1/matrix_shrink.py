"""МАТРИЧНА УСАДКА замість СКАЛЯРНОЇ (нове: похибка якоря — майже ОДИН режим).

drift.py: похибки якоря на різних відводах корельовані 0.60..0.96 (сер 0.74)
   -> у 8-вимірному просторі відводів це фактично ОДИН спільний напрямок, не 8 незалежних

ЗАРАЗ ми боремося скалярною усадкою: тиснемо ВСЮ поправку одним λ (С45).
АЛЕ якщо зсув сидить в одному напрямку u, правильніше вибрати β ⟂ u:
   зсув βᵀδ ≈ 0, а з сигналу губиться лише одна вісь із восьми.
Напрямок u — БЕЗКОШТОВНИЙ: рахується офлайн по банку 200 нетів (не по наших семплах!).

Це «матрична усадка»: замість λ·I застосовуємо проєктор P = I − Σ uᵤuᵤᵀ.
Скалярна усадка ЦЬОГО НЕ ВМІЄ в принципі — вона стискає всі напрямки однаково.

⚠️ база = v2 (ZCA+антитетика). ⚠️ V₀ = MSE при продакшн-N, не score.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/matrix_shrink.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=20; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
from numpy.polynomial.hermite import hermgauss
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=DT,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=DT,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))
TAPS=[3,7,11,15,19,23,27,30]; K=len(TAPS)

# ---- напрямки похибки якоря з БАНКУ (офлайн, безкоштовно) ----
c0=torch.load('/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt',weights_only=False)
TRN=np.asarray(c0['TRN'],dtype=np.float64); TRT=np.asarray(c0['TRT'],dtype=np.float64)
REL=(TRN-TRT)/TRT
SIG=(REL.T@REL)/len(REL)
ev,U=np.linalg.eigh(SIG); ev=ev[::-1]; U=U[:,::-1]
log('спектр коваріації похибки якоря (по 200 нетах):')
log('  власні числа: ' + ' '.join('{:.3e}'.format(v) for v in ev))
log('  частка у топ-1 / топ-2 / топ-3: {:.4f} / {:.4f} / {:.4f}'.format(
    ev[0]/ev.sum(), ev[:2].sum()/ev.sum(), ev[:3].sum()/ev.sum()))
Ut=torch.tensor(U.copy(),dtype=DT,device=dev)
SIGt=torch.tensor(SIG,dtype=DT,device=dev)

def nngp_tr(Wd):
    m=torch.zeros(D,dtype=DT,device=dev); Kk=torch.eye(D,dtype=DT,device=dev); tr=[]
    for ell in range(L):
        mu=Wd[ell].t()@m; C=Wd[ell].t()@(Kk-torch.outer(m,m))@Wd[ell]
        sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t=mu/sd; Ph=Phi(t); ph=phi(t)
        m=mu*Ph+sd*ph
        rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt((1-rho*rho).clamp(min=1e-14))
        Kg=torch.zeros(D,D,dtype=DT,device=dev); sdsq=sd[None,:]*sq
        for a in range(H):
            Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
            Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
            Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
        Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg); Kk=Kg
        if ell in TAPS: tr.append(float(torch.diagonal(Kk).sum()))
    return np.array(tr)

def truth(W,N=6_000_000,CH=200_000):
    s=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(880+c))
        x=torch.cat([z,-z],0)
        for li in range(L): x=relu(x@W[li])
        s+=x.double().sum(0); n+=x.shape[0]
    return s/n

def v2_batch(W,npair,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g); x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    e_,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(e_.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    a=x@Wh; T=[]
    for li in range(L):
        a=relu(a@W[li])
        if li in TAPS: T.append((a*a).sum(1))
    return a.double(), torch.stack(T,1).double()

NPAIR=100_000; REPS=10
names=('v2','скаляр λ','проєкт-1','проєкт-2','проєкт-3','проєкт+λ')
res={n:[] for n in names}
for net in [0,1,2,3]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    TH=truth(W); trn=torch.tensor(nngp_tr([w.double() for w in W]),dtype=DT,device=dev)
    for r in range(REPS):
        a,T=v2_batch(W,NPAIR,4400+net*61+r); N=a.shape[0]
        base=a.mean(0); vA=a.var(0); alive=vA>1e-6*float(vA.max())
        Tm=T.mean(0); Tc=T-Tm
        S=(Tc.t()@Tc)/N+1e-9*torch.eye(K,dtype=DT,device=dev)
        b=(Tc.t()@(a-base))/N
        beta=torch.linalg.solve(S.cpu(),b.cpu()).to(dev)              # (K,D)
        d=Tm-trn
        R2=torch.where(alive,((beta*b).sum(0)/vA.clamp(min=1e-300)),torch.zeros_like(vA)).clamp(0,0.999)
        G=(vA/N)*R2
        SIGa=SIGt*torch.outer(Tm,Tm)
        b2=torch.einsum('kj,kl,lj->j',beta,SIGa,beta)
        lam=torch.where(alive,G/(G+b2).clamp(min=1e-300),torch.zeros_like(G)).clamp(0,1)
        est={'v2':base,'скаляр λ':base-lam*(beta*d[:,None]).sum(0)}
        # ПРОЄКЦІЯ: викидаємо m головних напрямків похибки якоря (у масштабі Tm)
        for m_ in (1,2,3):
            Um=Ut[:,:m_]*Tm[:,None]                                    # напрямки в абсолютних одиницях
            Q,_=torch.linalg.qr(Um.cpu()); Q=Q.to(dev)
            P=torch.eye(K,dtype=DT,device=dev)-Q@Q.t()
            bp=P@beta                                                  # β ⟂ напрямкам похибки
            est['проєкт-%d'%m_]=base-(bp*d[:,None]).sum(0)
            if m_==2:
                bb=torch.einsum('kj,kl,lj->j',bp,SIGa,bp)
                Rp=torch.where(alive,((bp*b).sum(0)/vA.clamp(min=1e-300)),torch.zeros_like(vA)).clamp(0,0.999)
                Gp=(vA/N)*Rp
                lp=torch.where(alive,Gp/(Gp+bb).clamp(min=1e-300),torch.zeros_like(Gp)).clamp(0,1)
                est['проєкт+λ']=base-lp*(bp*d[:,None]).sum(0)
        for k_,v_ in est.items(): res[k_].append(float(((v_-TH)**2).mean()))
    log('нет {} готовий'.format(net))
log('')
log('=== РЕЗУЛЬТАТ (4 нети × {} повторів, N={}, база=v2) ==='.format(REPS,2*NPAIR))
mv=np.mean(res['v2'])
for n_ in names:
    m=np.mean(res[n_]); se=np.std(res[n_])/math.sqrt(len(res[n_]))
    log('  {:10s} MSE {:.4e} ± {:.1e}   {:+.2f}%'.format(n_,m,se,100*(m/mv-1)))
log('done.')
