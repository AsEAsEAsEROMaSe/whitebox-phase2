"""АДАПТИВНА УСАДКА: λ під КОНКРЕТНИЙ нет, безкоштовно.

ПРОБЛЕМА (виявлено tap_tradeoff.py): похибка якоря NNGP різниться МІЖ НЕТАМИ вдесятеро
(нет0 ~1e-3, нет1 ~1e-2). А λ ми брали з ФІКСОВАНОГО пріору по банку 200 нетів (С45).
⟹ систематично ПЕРЕ-усаджуємо точні нети й НЕДО-усаджуємо неточні.

ЗАЧІПКА: спостережувана різниця d = Tr_MC − Tr_NNGP = (шум MC) + δ,  δ = похибка якоря.
   обидві частини БЕЗКОШТОВНІ: Tr_MC і Tr_NNGP уже пораховані,
   дисперсія шуму MC = Var(||a^ℓ||²)/N теж уже порахована.
⟹ δ̂² = max(0, d² − Var_MC/N)  — оцінка похибки якоря ЦЬОГО нета, на льоту.
⟹ λ = G/(G + β²δ̂²) замість λ з пріору.

БАЗА = v2 (ZCA-вибілювання + антитетика), НЕ простий MC.
(С45: міряння від простого MC дало фейкові −1.37%, бо не та робоча точка.)
⚠️ істина з 6М семплів; кілька повторів; held-out β.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/adaptive_lambda.log'
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
TAPS=[3,7,11,15,19,23,27,30]; K=len(TAPS)

def nngp_tr(W):
    m=torch.zeros(D,dtype=DT,device=dev); Kk=torch.eye(D,dtype=DT,device=dev); tr=[]
    for ell in range(L):
        mu=W[ell].t()@m; C=W[ell].t()@(Kk-torch.outer(m,m))@W[ell]
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
    """v2: антитетика + ZCA-вибілювання. Повертає a^L і сліди ||a^ℓ||² посемплово."""
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g); x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    x=x@Wh
    a=x; T=[]
    for li in range(L):
        a=relu(a@W[li])
        if li in TAPS: T.append((a*a).sum(1))
    return a.double(), torch.stack(T,1).double()

# пріор на похибку слідів (як у С45) — для порівняння з адаптивною
c0=torch.load('/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt',weights_only=False)
REL=(np.asarray(c0['TRN'])-np.asarray(c0['TRT']))/np.asarray(c0['TRT'])
SIG_REL=torch.tensor((REL.T@REL)/len(REL),dtype=DT,device=dev)

NPAIR=100_000; REPS=10
res={n:[] for n in ('v2','CV пріор λ','CV АДАПТ λ','CV λ=1')}
for net in [0,1,2,3]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    TH=truth(W); trn=torch.tensor(nngp_tr([w.double() for w in W]),dtype=DT,device=dev)
    for r in range(REPS):
        a,T=v2_batch(W,NPAIR,4400+net*61+r); N=a.shape[0]
        base=a.mean(0); vA=a.var(0); alive=vA>1e-6*float(vA.max())
        Tm=T.mean(0); Tc=T-Tm
        S=(Tc.t()@Tc)/N+1e-9*torch.eye(K,dtype=DT,device=dev)
        b=(Tc.t()@(a-base))/N
        beta=torch.linalg.solve(S.cpu(),b.cpu()).to(dev)
        R2=torch.where(alive,((beta*b).sum(0)/vA.clamp(min=1e-300)),torch.zeros_like(vA)).clamp(0,0.999)
        G=(vA/N)*R2
        d=Tm-trn                                   # спостережувана різниця (БЕЗКОШТОВНА)
        varMC=T.var(0)/N                           # шум MC на слід (БЕЗКОШТОВНИЙ)
        # --- пріорна усадка (С45) ---
        SIGp=SIG_REL*torch.outer(Tm,Tm)
        b2p=torch.einsum('kj,kl,lj->j',beta,SIGp,beta)
        lam_p=torch.where(alive,G/(G+b2p).clamp(min=1e-300),torch.zeros_like(G)).clamp(0,1)
        corr=(beta*d[:,None]).sum(0)               # = βᵀd, сама поправка CV
        # --- АДАПТИВНА усадка ---
        # ⚠️ похибки якоря на РІЗНИХ відводах СИЛЬНО корельовані (спільна накопичена
        # похибка замикання) -> діагональна оцінка НЕДООЦІНЮЄ зсув. Правильно: увесь
        # βᵀd є оцінкою зсуву, віднімаємо лише ШУМОВУ частину βᵀdiag(varMC)β.
        noise=((beta*beta)*varMC[:,None]).sum(0)
        b2a=(corr*corr-noise).clamp(min=0.0)
        lam_a=torch.where(alive,G/(G+b2a).clamp(min=1e-300),torch.zeros_like(G)).clamp(0,1)
        for nm,est in (('v2',base),('CV пріор λ',base-lam_p*corr),
                       ('CV АДАПТ λ',base-lam_a*corr),('CV λ=1',base-corr)):
            res[nm].append(float(((est-TH)**2).mean()))
        if r==0:
            log('нет {}: ||d||={:.3e}  ||√varMC||={:.3e}  λ_пріор сер {:.3f}  λ_адапт сер {:.3f}'.format(
                net,float(d.norm()),float(varMC.sqrt().norm()),
                float(lam_p[alive].mean()),float(lam_a[alive].mean())))
log('')
log('=== РЕЗУЛЬТАТ (4 нети × {} повторів, N={}, база=v2) ==='.format(REPS,2*NPAIR))
mv=np.mean(res['v2'])
for nm in ('v2','CV пріор λ','CV АДАПТ λ','CV λ=1'):
    m=np.mean(res[nm]); se=np.std(res[nm])/math.sqrt(len(res[nm]))
    log('  {:12s} MSE {:.4e} ± {:.1e}   {:+.2f}%'.format(nm,m,se,100*(m/mv-1)))
log('done.')
