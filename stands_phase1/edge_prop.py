"""ЕТАП 2: ПОВНА ПРОПАГАЦІЯ 32 шари з виправленим замиканням (середнє + коваріація).

edge_step.py валідував формули на кроці 1→2: коваріація 1.15×, середнє 1.57× (амплітуда)
на рівні НЕЗАЛЕЖНОСТІ. Стенд із ПАРАМИ давав 8-10×, з оракулом 60×.
Уся різниця — у якості κ₃. Тут добудовуємо ПАРИ на всі шари.

Пари: κ₃(aᵢ,aᵢ,aⱼ) = f(ρ)·sᵢ²·sⱼ, де f — 1D таблиця по ρ (виведена при t=0).
ПРАВОМІРНІСТЬ t≈0 НА ВСІХ ШАРАХ: затвори всюди відкриті на 0.47-0.51 ⟹ Φ(t)≈0.5 ⟹ |t|≤0.08.

ВАРІАНТИ, що порівнюємо на фіналі E[a^32]:
   NNGP           — чисте гаусове замикання
   +m(d²)         — поправка лише середнього, кумулянти незалежні
   +m(d³)         — поправка середнього, кумулянти з парами
   +m+K (d³)      — повна: середнє І коваріація
   v2             — орієнтир
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/edge_prop.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
from numpy.polynomial.hermite import hermgauss
HQ=48
xq,wq=hermgauss(HQ)
zq=torch.tensor(xq*SQ2,dtype=DT,device=dev); wq1=torch.tensor(wq/math.sqrt(math.pi),dtype=DT,device=dev)

# ---- таблиця мішаного кумулянта κ₃(r₁,r₁,r₂)(ρ) для СТАНДАРТНИХ центрованих ----
NG=48
xg,wg=hermgauss(NG)
ug=torch.tensor(xg*SQ2,dtype=DT,device=dev); wg2=torch.tensor(wg/math.sqrt(math.pi),dtype=DT,device=dev)
RHO=torch.linspace(-0.995,0.995,401,dtype=DT,device=dev)
def _mixed(rho):
    a=ug[:,None,None]; b=ug[None,:,None]; rh=rho[None,None,:]
    g1=a; g2=rh*a+torch.sqrt((1-rh*rh).clamp(min=0))*b
    r1=relu(g1); r2=relu(g2); Wt=wg2[:,None,None]*wg2[None,:,None]
    E112=(Wt*r1*r1*r2).sum((0,1)); E11=(Wt*r1*r1).sum((0,1))
    E12=(Wt*r1*r2).sum((0,1)); E1=float((wg2*relu(ug)).sum())
    return E112-E11*E1-2*E12*E1+2*E1*E1*E1
TAB=_mixed(RHO)
def tab(rho):
    i=((rho+0.995)/1.99*(len(RHO)-1)).clamp(0,len(RHO)-1-1e-6)
    lo=i.floor().long(); fr=i-lo
    return TAB[lo]*(1-fr)+TAB[(lo+1).clamp(max=len(RHO)-1)]*fr

def cum_relu(mu,sd):
    y=relu(mu[None,:]+sd[None,:]*zq[:,None]); w=wq1[:,None]
    M1=(w*y).sum(0); M2=(w*y*y).sum(0); M3=(w*y**3).sum(0); M4=(w*y**4).sum(0)
    c3=M3-3*M1*M2+2*M1**3
    c4=M4-4*M1*M3-3*M2*M2+12*M1*M1*M2-6*M1**4
    return c3,c4

def gauss_step(m,K,W):
    mu=W.t()@m; C=W.t()@(K-torch.outer(m,m))@W
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t=mu/sd
    Ph=Phi(t); ph=phi(t); mn=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.995,0.995)
    sq=torch.sqrt((1-rho*rho).clamp(min=1e-14)); sdsq=sd[None,:]*sq
    Kg=torch.zeros(D,D,dtype=DT,device=dev)
    for a in range(HQ):
        Ri=relu(mu+sd*zq[a]); gjb=mu[None,:]+sd[None,:]*(rho*zq[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*zq[:,None,None])
        Kg=Kg+wq1[a]*Ri[:,None]*(wq1[:,None,None]*Rj).sum(0)
    dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
    Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg)
    return mn,0.5*(Kg+Kg.t()),mu,sd,t,rho

def edge_terms(mu,sd,t,rho):
    sq=torch.sqrt((1-rho*rho).clamp(min=1e-12))
    A=(phi(t)/sd)[:,None]*Phi((t[None,:]-rho*t[:,None])/sq)
    m0=mu[None,:]-rho*sd[None,:]*t[:,None]; s=sd[None,:]*sq
    Er=m0*Phi(m0/s)+s*phi(m0/s)
    B=-((t*phi(t)/(sd*sd))[:,None]*Er+(phi(t)/sd)[:,None]*Phi(m0/s)*(rho*sd[None,:]/sd[:,None]))
    return A,B

def run(W,mode):
    """mode: 'nngp' | 'm2' | 'm3' | 'm3K'"""
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev)
    c3a=torch.zeros(D,dtype=DT,device=dev); c4a=torch.zeros(D,dtype=DT,device=dev)
    sda=torch.ones(D,dtype=DT,device=dev); rhoa=torch.eye(D,dtype=DT,device=dev)
    for l in range(L):
        mn,Kg,mu,sd,t,rho=gauss_step(m,K,W[l])
        if mode!='nngp':
            W2=W[l]*W[l]
            k3h=(W2*W[l]).t()@c3a                        # d²: незалежність
            k4h=(W2*W2).t()@c4a
            if mode in ('m3','m3K') and l>0:
                # ПАРИ: B[i,i'] = κ₃(aᵢ,aᵢ,aᵢ') = f(ρᵢᵢ')·sᵢ²·sᵢ'
                Bp=tab(rhoa)*(sda[:,None]**2)*sda[None,:]
                Bp=Bp-torch.diag(torch.diagonal(Bp))
                k3h=k3h+3.0*torch.einsum('ij,ik,jk->k',Bp,W2,W[l])
            var=sd*sd
            mn=mn+(k3h/6.0)*(-t*phi(t)/var)+(k4h/24.0)*((t*t-1)*phi(t)/(sd*var))
            if mode=='m3K':
                M3=W2.t()@torch.diag(c3a)@W[l]           # κ_iij
                A,Bt=edge_terms(mu,sd,t,rho)
                kiii=torch.diagonal(M3)
                dK=(kiii[:,None]*Bt+3.0*M3*A+3.0*M3.t()*A.t()+kiii[None,:]*Bt.t())/6.0
                Kg=Kg+0.5*(dK+dK.t())
        c3a,c4a=cum_relu(mu,sd); sda=sd; rhoa=rho
        m,K=mn,Kg
    return m

def truth(W,N=6_000_000,CH=200_000):
    s=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(880+c),dtype=DT)
        x=torch.cat([z,-z],0)
        for li in range(L): x=relu(x@W[li])
        s+=x.double().sum(0); n+=x.shape[0]
    return s/n

def v2est(W,npair,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g,dtype=DT); x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    e_,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(e_.clamp(min=1e-12).rsqrt())@V.t()).to(dev)
    a=(x.double()@Wh)
    for li in range(L): a=relu(a@W[li])
    return a.mean(0)

for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev).double() for i in range(L)]
    TH=truth(W); p=float((TH*TH).mean())
    log(''); log('######## НЕТ {} ########'.format(net))
    base=None
    for mode,nm in (('nngp','NNGP'),('m2','+m(d²)'),('m3','+m(d³) пари'),('m3K','+m+K (d³)')):
        e=run(W,mode); r=float(((e-TH)**2).mean())/p
        if base is None: base=r
        log('  {:12s} відн.MSE = {:.4e}   ({:.2f}× проти NNGP)'.format(nm,r,base/max(r,1e-300)))
    vs=[float(((v2est(W,100_000,5500+net*7+k)-TH)**2).mean())/p for k in range(4)]
    log('  {:12s} відн.MSE = {:.4e}   <- орієнтир'.format('v2',float(np.mean(vs))))
log('done.')
