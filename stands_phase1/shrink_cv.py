"""Чи правильна була модель? Порогова ("треба 7.6x або нічого") проти гладкої (усадка beta).

MSE(l) = mean_j [ V0_j*(1 - l*(2-l)*R2_j) + l^2*bias_j^2 ]
l_opt  = G/(G+Bs)        MSE_opt = V0 - G^2/(G+Bs)

Пріор на delta беремо ЧЕСНО з банку 200 нетів (micro_trace_cache), а не з цього ж нета.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/shrink_cv.log'
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
TAPS=[3,7,11,15,19,23,27,30]; k=len(TAPS)
V2=3.09e-7
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))

def nngp(W):
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
            Kg+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        Kg[range(D),range(D)]=(mu*mu+sd*sd)*Ph+mu*sd*ph
        K=Kg
        if ell in TAPS: tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

def moments(W,seed0,nch,CH=10000):
    """антитетичні пари: Y=симетризована фінальна активація, T=симетризовані 8 норм"""
    A=torch.zeros(k,k,dtype=torch.float64,device=dev); BY=torch.zeros(k,D,dtype=torch.float64,device=dev)
    SX=torch.zeros(k,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev)
    N=0
    def fwd(x):
        out=[]
        for li in range(L):
            x=relu(x@W[li])
            if li in TAPS: out.append((x*x).sum(1))
        return x,out
    for c in range(nch):
        w=torch.randn(CH,D,device=dev,generator=torch.Generator(device=dev).manual_seed(seed0+c))
        ap,na=fwd(w); am,nb=fwd(-w)
        Y=(0.5*(ap+am)).double(); T=torch.stack([0.5*(a+b) for a,b in zip(na,nb)],1).double()
        A+=T.t()@T; BY+=T.t()@Y; SX+=T.sum(0); SY+=Y.sum(0); SY2+=(Y*Y).sum(0); N+=CH
    return [t.cpu() for t in (A,BY,SX,SY,SY2)]+[N]

# ---------- пріор на delta з банку 200 нетів (ЧЕСНО, не з цього нета) ----------
c=torch.load('/mnt/win_d/llm/White-Box/netbank/micro_trace_cache.pt',weights_only=False)
TRN200,TRT200=np.asarray(c['TRN']),np.asarray(c['TRT'])
REL=(TRN200-TRT200)/TRT200                      # (200,8) відносна похибка слідів
log('=== ПРІОР на похибку слідів (банк {} нетів) ==='.format(len(REL)))
log('шар   |  сер.rel.err |   std по нетах |  RMS')
for i,e in enumerate(TAPS):
    log('  {:2d}  | {:+.3e}  |   {:.3e}   | {:.3e}'.format(e,REL[:,i].mean(),REL[:,i].std(),np.sqrt((REL[:,i]**2).mean())))
COVREL=(REL.T@REL)/len(REL)                     # E[delta_rel delta_rel^T]
log('RMS усього: {:.3e}'.format(np.sqrt((REL**2).mean())))

# ---------- по нетах ----------
NETS=[0,1,2,3]
log('')
log('=== ПОРОГОВА МОДЕЛЬ ПРОТИ ГЛАДКОЇ (усадка) ===')
rows=[]
for net in NETS:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    trn=nngp(W)
    A,BY,SX,SY,SY2,N=moments(W,50000+1000*net,40)          # 400k пар
    Ah,BYh,SXh,SYh,SY2h,Nh=moments(W,80000+1000*net,20)    # 200k пар held-out
    SStot=SY2-SY*SY/N; alive=SStot>1e-9*float(SStot.max())
    Ac=A-torch.outer(SX,SX)/N; Bc=BY-torch.outer(SX,SY)/N
    beta=torch.linalg.solve(Ac+1e-6*torch.eye(k,dtype=torch.float64),Bc)   # (k,D)
    # held-out R^2 покомпонентно
    SStoth=SY2h-SYh*SYh/Nh
    Ach=Ah-torch.outer(SXh,SXh)/Nh; Bch=BYh-torch.outer(SXh,SYh)/Nh
    SSresh=SStoth-2*(beta*Bch).sum(0)+((beta.t()@Ach)*beta.t()).sum(1)
    R2=(1-SSresh/SStoth).clamp(0,1)
    # істинні сліди з тієї ж вибірки
    Etrue=(SX/N).numpy()
    delta=torch.tensor(trn-Etrue,dtype=torch.float64)       # абсолютна похибка
    bias=(beta*delta[:,None]).sum(0)                        # (D,)
    # V0_j: покомпонентна дисперсія v2, нормована так щоб середнє = 3.09e-7
    V0=SStot/N; V0=V0/float(V0[alive].mean())*V2
    G=float((V0*R2)[alive].mean())
    Bs=float((bias[alive]**2).mean())
    lam=G/(G+Bs)
    mse=lambda l: float((V0[alive]*(1-l*(2-l)*R2[alive])+l*l*bias[alive]**2).mean())
    rows.append((net,float(R2[alive].mean()),G,Bs,lam,mse(1.0),mse(lam)))
    log('net {}: R2_ho {:.4f}  G {:.3e}  Bs(l=1) {:.3e}  ->  l_opt {:.4f}'.format(
        net,float(R2[alive].mean()),G,Bs,lam))
    log('        MSE:  v2 {:.4e} | l=1 {:.4e} ({:+.1f}%) | l_opt {:.4e} ({:+.2f}%)'.format(
        V2,mse(1.0),100*(mse(1.0)/V2-1),mse(lam),100*(mse(lam)/V2-1)))
    if net==NETS[0]:
        log('        крива по точності слідів (уявно менша delta у f разів):')
        for f in [1,2,4,7.6,20,1e9]:
            b2=Bs/f**2; l=G/(G+b2)
            m=float((V0[alive]*(1-l*(2-l)*R2[alive])+l*l*(bias[alive]/f)**2).mean())
            log('          f={:>5}: Bs {:.2e}  l_opt {:.4f}  MSE {:.4e}  ({:+.2f}%)'.format(
                ('inf' if f>1e8 else f),b2,l,m,100*(m/V2-1)))

log('')
w1=np.mean([r[5] for r in rows]); wo=np.mean([r[6] for r in rows])
log('СЕРЕДНЄ по {} нетах:  l=1 -> {:.4e} ({:+.1f}%)   l_opt -> {:.4e} ({:+.2f}%)'.format(
    len(rows),w1,100*(w1/V2-1),wo,100*(wo/V2-1)))
log('done.')
