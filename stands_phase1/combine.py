"""ЗБІРКА: усе, що сьогодні виміряно як робоче, в один оцінювач проти v2.

база   = v2 (ZCA-вибілювання + антитетика)                     [чинний прод]
+CV_T  = глибокі норми ||a^l||^2, якір NNGP, УСАДКА lambda      [С45: -1.1…-1.37%]
+CV_H  = He4/He6 у напрямках мережі, середнє ТОЧНО нуль        [С45: +4.1% при продакшн-N]
обидва разом

Усе з ОДНОГО проходу: норми безкоштовні, ермітові ознаки = один матмул x@U.
N = продакшн (200k), бо знак результату залежить від N ([[feedback-match-experiment-to-production]]).
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/combine.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=24
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev)
wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[3,7,11,15,19,23,27,30]; k=len(TAPS)
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))

def nngp_traces(W):
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); tr=[]
    for ell in range(L):
        mu=W[ell].t()@m; C=W[ell].t()@(K-torch.outer(m,m))@W[ell]
        sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12)); t=mu/sd; Ph=Phi(t); ph=phi(t)
        m=mu*Ph+sd*ph
        rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
        Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
        for a in range(H):
            Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
            Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
            Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
        Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg); K=Kg
        if ell in TAPS: tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

def he(t,p):
    h0=torch.ones_like(t); h1=t
    for kk in range(2,p+1): h0,h1=h1,t*h1-(kk-1)*h0
    return h1 if p>=1 else h0

def truth(W,N=6_000_000,CH=200_000):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(880+c))
        x=torch.cat([z,-z],0)
        for li in range(L): x=relu(x@W[li])
        s+=x.double().sum(0); n+=x.shape[0]
    return s/n

def batch(W,npair,seed,U):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g)
    x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    x=x@Wh
    s=x@U
    F=torch.cat([he(s,4)/math.sqrt(24.0), he(s,6)/math.sqrt(720.0)],1)   # E=0 ТОЧНО
    a=x; T=[]
    for li in range(L):
        a=relu(a@W[li])
        if li in TAPS: T.append((a*a).sum(1))
    return a, torch.stack(T,1), F

# пріор на похибку слідів NNGP (200 нетів) -> усадка
c0=torch.load('/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt',weights_only=False)
REL=(np.asarray(c0['TRN'])-np.asarray(c0['TRT']))/np.asarray(c0['TRT'])
SIG_REL=(REL.T@REL)/len(REL)

NPAIR=100_000; REPS=12; KU=64
res={n:[] for n in ('v2','+CV_T','+CV_H','обидва','CV_T(пілот β)')}
for net in [0,1,2,3]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    TH=truth(W); U=(W[0]/W[0].norm(dim=0,keepdim=True))[:,:KU]
    trn=nngp_traces(W)
    # ПІЛОТНА вибірка для β (окрема, не входить в оцінювач) — тест гіпотези
    ap,Tp,_=batch(W,NPAIR,777000+net,U)
    apd=ap.double(); Tpd=Tp.double(); Np=apd.shape[0]
    bp=apd.mean(0); Tpm=Tpd.mean(0); Tpc=Tpd-Tpm
    Sp=(Tpc.t()@Tpc)/Np+1e-9*torch.eye(k,dtype=torch.float64,device=dev)
    betaP=torch.linalg.solve(Sp.cpu(),((Tpc.t()@(apd-bp))/Np).cpu()).to(dev)
    vAp=apd.var(0); alivep=vAp>1e-9*float(vAp.max())
    R2p=torch.where(alivep,((betaP*((Tpc.t()@(apd-bp))/Np)).sum(0)/vAp.clamp(min=1e-300)),
                    torch.zeros_like(vAp)).clamp(0,0.999)
    SIGp=torch.tensor(SIG_REL*np.outer(Tpm.cpu().numpy(),Tpm.cpu().numpy()),
                      dtype=torch.float64,device=dev)
    b2p=torch.einsum('kj,kl,lj->j',betaP,SIGp,betaP)
    del ap,Tp,apd,Tpd
    for r in range(REPS):
        a,T,F=batch(W,NPAIR,13000+net*97+r,U)
        N=a.shape[0]
        ad=a.double(); Td=T.double(); Fd=F.double()
        base=ad.mean(0)
        vA=ad.var(0); alive=vA>1e-9*float(vA.max())      # мертві нейрони: var=0 -> NaN
        # --- CV на глибоких нормах, якір NNGP + усадка ---
        Tm=Td.mean(0); Tc=Td-Tm
        S=(Tc.t()@Tc)/N+1e-9*torch.eye(k,dtype=torch.float64,device=dev)
        b=(Tc.t()@(ad-base))/N
        betaT=torch.linalg.solve(S.cpu(),b.cpu()).to(dev)
        Etr=Tm.cpu().numpy()
        SIG=torch.tensor(SIG_REL*np.outer(Etr,Etr),dtype=torch.float64,device=dev)
        bias2=torch.einsum('kj,kl,lj->j',betaT,SIG,betaT)
        vN=(vA/N).clamp(min=1e-300)
        # R2 = пояснена ПОСЕМПЛОВА дисперсія / ПОСЕМПЛОВА дисперсія (не дисперсія середнього!)
        R2v=torch.where(alive,((betaT*b).sum(0)/vA.clamp(min=1e-300)),
                        torch.zeros_like(vA)).clamp(0,0.999)
        G=vN*R2v
        lam=torch.where(alive,G/(G+bias2).clamp(min=1e-300),torch.zeros_like(G)).clamp(0,1)
        anchor=torch.tensor(trn,dtype=torch.float64,device=dev)
        cvT=lam*(betaT*(Tm-anchor)[:,None]).sum(0)
        # --- CV на ермітових ознаках (середнє точно 0) ---
        Fm=Fd.mean(0); Fc=Fd-Fm
        SF=(Fc.t()@Fc)/N+1e-9*torch.eye(Fd.shape[1],dtype=torch.float64,device=dev)
        bF=(Fc.t()@(ad-base))/N
        betaF=torch.linalg.solve(SF.cpu(),bF.cpu()).to(dev)
        cvH=(betaF*Fm[:,None]).sum(0)
        Gp=(vA/N)*R2p
        lamP=torch.where(alive,Gp/(Gp+b2p).clamp(min=1e-300),torch.zeros_like(Gp)).clamp(0,1)
        cvTp=lamP*(betaP*(Tm-anchor)[:,None]).sum(0)
        for nm,est in (('v2',base),('+CV_T',base-cvT),('+CV_H',base-cvH),
                       ('обидва',base-cvT-cvH),('CV_T(пілот β)',base-cvTp)):
            res[nm].append(float(((est-TH)**2).mean()))
    log('нет {} готовий'.format(net))
log('')
log('=== РЕЗУЛЬТАТ ({} нетів × {} повторів, N={}) ==='.format(4,REPS,2*NPAIR))
mv=np.mean(res['v2'])
for nm in ('v2','+CV_T','CV_T(пілот β)','+CV_H','обидва'):
    m=np.mean(res[nm]); se=np.std(res[nm])/math.sqrt(len(res[nm]))
    log('  {:8s} MSE {:.4e} ± {:.1e}   {:+.2f}%'.format(nm,m,se,100*(m/mv-1)))
log('done.')
