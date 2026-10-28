"""Два питання:
 1. Чи ранжує шари БЕЗКОШТОВНА норма спряженого градієнта ||G_s|| (без оракульного dK)?
 2. Який ефективний РАНГ у G_s — чи можна брати <G_s, a a^T> кількома проєкціями замість повної форми?
 + скільки дає точне знання топ-k скалярів.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
from torch.utils.checkpoint import checkpoint
from scipy.stats import spearmanr
LOG='/mnt/win_d/llm/White-Box/leverage.log'
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

SP={}; RK={}
for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    mt,KT=true_state(W)
    trL=float(torch.diagonal(KT[L-1]).sum())
    log(''); log('############ NET {} ############'.format(net))

    P=[torch.zeros(D,D,device=dev,requires_grad=True) for _ in range(L)]
    Pm=[torch.zeros(D,device=dev,requires_grad=True) for _ in range(L)]
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); Kg_free=[]
    for l in range(L):
        m,K=checkpoint(readout,m,K,W[l],use_reentrant=False)
        Kg_free.append(K.detach().clone())
        m=m+Pm[l]; K=K+P[l]
    torch.diagonal(K).sum().backward()
    gz=lambda p: torch.zeros_like(p) if p.grad is None else p.grad.detach().clone()
    G=[gz(p) for p in P]

    # істинні внески (через оракульне dK)
    contrib=[]; dKs=[]
    with torch.no_grad():
        for s in range(L-1):
            m0=torch.zeros(D,device=dev) if s==0 else mt[s-1].float()
            K0=torch.eye(D,device=dev) if s==0 else KT[s-1].float()
            mg,Kg=readout(m0,K0,W[s])
            dK=KT[s].float()-Kg; dKs.append(dK)
            contrib.append(-float((G[s]*dK).sum())/trL)
    contrib=np.array(contrib); ac=np.abs(contrib)

    # --- 1. безкоштовні проксі ---
    gn=np.array([float(G[s].norm()) for s in range(L-1)])
    kn=np.array([float(Kg_free[s].norm()) for s in range(L-1)])
    tk=np.array([float(torch.diagonal(Kg_free[s]).sum()) for s in range(L-1)])
    prox={'||G||':gn, '||G||*||K||':gn*kn, '||G||*Tr(K)':gn*tk}
    log('1. БЕЗКОШТОВНІ ПРОКСІ проти істинного |внеску|:')
    true_top=set(np.argsort(-ac)[:5].tolist())
    for k,v in prox.items():
        r=spearmanr(v,ac).correlation
        top=set(np.argsort(-v)[:5].tolist())
        log('   {:14s} spearman {:+.3f}   топ-5 перетин {}/5   {}'.format(k,r,len(top&true_top),sorted(top)))
        SP.setdefault(k,[]).append((r,len(top&true_top)))
    log('   істинний топ-5: {}'.format(sorted(true_top)))

    # --- 2. ранг G_s на топ-шарах ---
    log('2. ЕФЕКТИВНИЙ РАНГ G_s (скільки власних напрямків тримають <G,dK>):')
    for s in sorted(true_top)[:3]:
        Gs=G[s].double().cpu(); Gs=0.5*(Gs+Gs.t())
        ev,U=torch.linalg.eigh(Gs)
        o=torch.argsort(-ev.abs()); ev=ev[o]; U=U[:,o]
        dk=dKs[s].double().cpu()
        full=float((Gs*dk).sum())
        line=[]
        for r in [1,2,4,8,16,32,64]:
            Gr=U[:,:r]@torch.diag(ev[:r])@U[:,:r].t()
            line.append('r{}={:.2f}'.format(r,float((Gr*dk).sum())/full))
        pr=float((ev.abs().sum()**2)/((ev**2).sum())/D)
        log('   шар {:2d}: PR={:.3f}  частка <G,dK> при ранзі: {}'.format(s,pr,'  '.join(line)))
        RK.setdefault(s,[]).append(0)

    # --- 3. що дає точне знання топ-k скалярів ---
    log('3. ЗАЛИШКОВА похибка сліду після точної корекції топ-k шарів:')
    o=np.argsort(-ac)
    tot=contrib.sum()
    for kk in [0,1,3,5,10,32]:
        rest=contrib.copy(); rest[o[:kk]]=0
        log('   k={:2d}: {:+.3e}  ({:.2f}x краще)'.format(kk,rest.sum(),abs(tot)/max(abs(rest.sum()),1e-12)))

log('')
log('ПІДСУМОК проксі (сер. spearman / сер. перетин топ-5):')
for k,v in SP.items():
    log('   {:14s} {:+.3f}   {:.1f}/5'.format(k,np.mean([a for a,_ in v]),np.mean([b for _,b in v])))
log('done.')
