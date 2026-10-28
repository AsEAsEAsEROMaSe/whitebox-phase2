"""Як саме накопичується похибка? Когерентно (однаковий знак, систематичний дрейф)
чи випадково (складається як sqrt)? І чи передбачувана одношагова похибка З САМОГО СТАНУ?

Три виміри:
 A. одношагова знакова похибка eps_l для КОЖНОГО з 32 шарів (старт з ІСТИННОГО стану l-1)
 B. внесок кожного шару У ФІНАЛЬНУ похибку (якір в істину на шарі s, далі замикання до кінця)
 C. чи корелює eps_l з величинами, які видно З ПРОПАГОВАНОГО стану (тобто чи можна лікувати на льоту)
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/compound.log'
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

def readout(m,K,Wl,obs=False):
    mu=Wl.t()@m; C=Wl.t()@(K-torch.outer(m,m))@Wl
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12)); t=mu/sd; Ph=Phi(t); ph=phi(t)
    mn=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
    Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
    for a in range(H):
        Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        Kg+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    Kg[range(D),range(D)]=(mu*mu+sd*sd)*Ph+mu*sd*ph
    if not obs: return mn,Kg,None
    off=rho-torch.diag(torch.diagonal(rho))
    ev=torch.linalg.eigvalsh(C.double().cpu()).clamp(min=1e-20)
    o={'mean_absrho':float(off.abs().mean()),'mean_rho2':float((off**2).mean()),
       'mean_t':float(t.mean()),'std_t':float(t.std()),
       'cv_sd':float(sd.std()/sd.mean()),
       'PR':float((ev.sum()**2/(ev*ev).sum())/D),
       'skew_diag':float(((torch.diagonal(Kg)-torch.diagonal(Kg).mean())**3).mean()
                         /torch.diagonal(Kg).std()**3)}
    return mn,Kg,o

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

NETS=[0,1,2]
ALL_EPS=[]; ALL_OBS=[]
for net in NETS:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,KT=true_state(W)
    trtrue=np.array([float(torch.diagonal(k).sum()) for k in KT])
    log('')
    log('################ NET {} ################'.format(net))

    # --- A: одношагова знакова похибка, старт з ІСТИННОГО стану ---
    eps=[]; obs=[]
    for l in range(L):
        m0=torch.zeros(D,device=dev) if l==0 else mt[l-1].float()
        K0=torch.eye(D,device=dev) if l==0 else KT[l-1].float()
        _,Kp,o=readout(m0,K0,W[l],obs=True)
        e=(float(torch.diagonal(Kp).sum())-trtrue[l])/trtrue[l]
        eps.append(e); obs.append(o)
    eps=np.array(eps); ALL_EPS.append(eps); ALL_OBS.append(obs)
    pos=int((eps>0).sum())
    log('A. ОДНОШАГОВА похибка (істинний старт) по 32 шарах:')
    log('   {}'.format(' '.join('{:+.1e}'.format(v) for v in eps[:16])))
    log('   {}'.format(' '.join('{:+.1e}'.format(v) for v in eps[16:])))
    log('   RMS {:.3e} | сер {:+.3e} | додатніх {}/32 | сума {:+.3e} | RMS*sqrt(32) {:.3e}'.format(
        np.sqrt((eps**2).mean()),eps.mean(),pos,eps.sum(),np.sqrt((eps**2).mean())*math.sqrt(32)))

    # --- B: внесок шару у ФІНАЛЬНУ похибку ---
    def prop_from(s):
        if s==0: m,K=torch.zeros(D,device=dev),torch.eye(D,device=dev)
        else:    m,K=mt[s-1].float(),KT[s-1].float()
        for l in range(s,L): m,K,_=readout(m,K,W[l])
        return (float(torch.diagonal(K).sum())-trtrue[L-1])/trtrue[L-1]
    E=np.array([prop_from(s) for s in range(L+1)])   # E[0]=чистий NNGP, E[L]=0
    contrib=E[:-1]-E[1:]
    log('B. фінальна похибка при якорі на шарі s: E[0]={:+.3e} (чистий NNGP) -> E[32]={:+.3e}'.format(E[0],E[L]))
    log('   внесок шару s у фінал:')
    log('   {}'.format(' '.join('{:+.1e}'.format(v) for v in contrib[:16])))
    log('   {}'.format(' '.join('{:+.1e}'.format(v) for v in contrib[16:])))
    log('   сума внесків {:+.3e} (має = E[0] {:+.3e})  додатніх {}/32'.format(
        contrib.sum(),E[0],int((contrib>0).sum())))
    amp=np.abs(contrib)/np.abs(eps)
    log('   ПІДСИЛЕННЯ |внесок|/|одношагова| по шарах (медіана {:.2f}):'.format(np.median(amp)))
    log('   {}'.format(' '.join('{:.1f}'.format(v) for v in amp[:16])))
    log('   {}'.format(' '.join('{:.1f}'.format(v) for v in amp[16:])))

# --- C: чи передбачувана eps_l зі СТАНУ (пул по нетах) ---
log('')
log('C. КОРЕЛЯЦІЯ одношагової похибки з видимими зі стану величинами (пул {} нетів x 32 шари):'.format(len(NETS)))
keys=list(ALL_OBS[0][0].keys())
Y=np.concatenate(ALL_EPS)
for k in keys:
    X=np.array([o[k] for ob in ALL_OBS for o in ob])
    ok=np.isfinite(X)&np.isfinite(Y)
    log('   {:14s} corr = {:+.4f}'.format(k,np.corrcoef(X[ok],Y[ok])[0,1]))
Xall=np.array([[o[k] for k in keys] for ob in ALL_OBS for o in ob])
ok=np.isfinite(Xall).all(1)&np.isfinite(Y)
Xa=Xall[ok]; Ya=Y[ok]
Xa=(Xa-Xa.mean(0))/(Xa.std(0)+1e-12)
A=Xa.T@Xa+1e-3*np.eye(Xa.shape[1]); b=Xa.T@(Ya-Ya.mean())
pred=Xa@np.linalg.solve(A,b)+Ya.mean()
log('   БАГАТОВИМІРНО (всі {} ознак, in-sample): corr {:.4f}  RMS {:.3e} -> {:.3e}  ({:.2f}x)'.format(
    len(keys),np.corrcoef(Ya,pred)[0,1],np.sqrt((Ya**2).mean()),
    np.sqrt(((Ya-pred)**2).mean()),np.sqrt((Ya**2).mean())/np.sqrt(((Ya-pred)**2).mean())))
log('done.')
