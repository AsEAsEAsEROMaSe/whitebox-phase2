"""АНАЛІТИЧНА K — версія 2, ВИПРАВЛЕНА КВАДРАТУРА.

БАГ v1: E[relu(h_i)relu(h_j)] брався 2D Гауссом-Ермітом (Q=24). Але ГЕ побудована для
ГЛАДКИХ підінтегральних, а в ReLU ЗЛАМ ⟹ збіжність жахлива. Контроль викрив: на шарі 1
похибка C вийшла 12.6%, хоча там вона мусить бути НУЛЬ (h⁰=x·W₀ гаусів ТОЧНО ⟹ моменти
relu від строго гаусового вектора рахуються точно, замикання ще не було).

ВИПРАВЛЕННЯ (робить інтеграл практично точним):
  h_i = m_i + s_i·u,  h_j = m_j + s_j(ρu + √(1−ρ²)v)
  ВНУТРІШНІЙ інтеграл по v — ЗАМКНЕНА ФОРМА:
      E_v[relu(a+bv)] = aΦ(a/b) + bφ(a/b),  a = m_j+s_jρu,  b = s_j√(1−ρ²)
  ЗОВНІШНІЙ по u — розрізаємо РІВНО НА ЗЛАМІ u₀=−m_i/s_i; при u>u₀ маємо relu=m_i+s_i·u,
  підінтегральна ГЛАДКА ⟹ Гаусс-Лежандр бере точно.

ПЕРЕВІРКА: при m=0 існує точна формула (ядро арккосинуса)
      E[X⁺Y⁺] = (σ₁σ₂/2π)(sinθ + (π−θ)cosθ),  θ=arccos(ρ)
звіряємось із нею перед прогоном.
"""
import torch, time, math, numpy as np
LOG='/mnt/win_d/llm/White-Box/analytic_K2.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32; NG=64
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def from_pre(W,h,l):
    a=torch.relu(h)
    for i in range(l+1,L): a=torch.relu(a@W[i])
    return a
def pre_at(W,x,l):
    a=x
    for i in range(l): a=torch.relu(a@W[i])
    return a@W[l]

_p,_w=np.polynomial.legendre.leggauss(NG)
GP=torch.tensor((_p+1)/2,dtype=torch.float64,device=dev)      # вузли на [0,1]
GW=torch.tensor(_w/2,dtype=torch.float64,device=dev)
SQ2PI=math.sqrt(2*math.pi); RANGE=14.0
def _Phi(x): return 0.5*(1+torch.erf(x/math.sqrt(2)))
def _phi(x): return torch.exp(-0.5*x*x)/SQ2PI

def relu_moments(m,C):
    s=C.diagonal().clamp(min=1e-20).sqrt(); z=m/s
    mu=m*_Phi(z)+s*_phi(z)
    rho=(C/torch.outer(s,s)).clamp(-1+1e-10,1-1e-10)
    E2=torch.empty(D,D,dtype=torch.float64,device=dev)
    CH=8
    for a0 in range(0,D,CH):
        mi=m[a0:a0+CH].view(-1,1,1); si=s[a0:a0+CH].view(-1,1,1)
        u0=(-mi/si)                                            # ЗЛАМ
        u=u0+RANGE*GP.view(1,1,NG)                             # інтегруємо [u₀, u₀+14]
        r=rho[a0:a0+CH].view(-1,D,1)
        mj=m.view(1,D,1); sj=s.view(1,D,1)
        a=mj+sj*r*u; b=(sj*torch.sqrt(1-r*r)).clamp(min=1e-12)
        inner=a*_Phi(a/b)+b*_phi(a/b)                          # ТОЧНО
        outer=(mi+si*u)*_phi(u)*inner
        E2[a0:a0+CH]=(outer*GW.view(1,1,NG)).sum(2)*RANGE
    return mu, E2-torch.outer(mu,mu)

# ---- самоперевірка проти ядра арккосинуса (m=0) ----
gg=torch.Generator(device=dev).manual_seed(5)
A=torch.randn(D,D,device=dev,generator=gg,dtype=torch.float64)/math.sqrt(D)
Cz=A@A.t()+torch.eye(D,dtype=torch.float64,device=dev)*0.1
mz=torch.zeros(D,dtype=torch.float64,device=dev)
mu_,S_=relu_moments(mz,Cz)
sz=Cz.diagonal().sqrt(); rz=(Cz/torch.outer(sz,sz)).clamp(-1,1)
th=torch.arccos(rz)
ref=torch.outer(sz,sz)/(2*math.pi)*(torch.sin(th)+(math.pi-th)*torch.cos(th))
got=S_+torch.outer(mu_,mu_)
log('САМОПЕРЕВІРКА квадратури проти ядра арккосинуса (m=0):')
log('  макс. відносна похибка = {:.3e}   (v1 на 2D Гауссі-Ерміті давав ~1e-1)'
    .format(float(((got-ref).abs()/ref.abs().clamp(min=1e-12)).max())))
log('  похибка E[relu] у 1D (замкнена форма): {:.3e}'
    .format(float((mu_-sz/SQ2PI).abs().max()/(sz/SQ2PI).max())))
log('')

for net in (0,1):
    W=load(net); log('='*86); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(30):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(60000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=from_pre(W,pre_at(W,xx,0),0).double().sum(0); n+=xx.shape[0]
    TH=S/n
    ta=time.time()
    W0=W[0].double(); mA={0:torch.zeros(D,dtype=torch.float64,device=dev)}; CA={0:W0.t()@W0}
    for l in range(0,L-1):
        mu,Sg=relu_moments(mA[l],CA[l]); Wl=W[l+1].double()
        mA[l+1]=mu@Wl; CA[l+1]=Wl.t()@Sg@Wl
    log('  аналітичний прохід {:.1f}с (разово)'.format(time.time()-ta))
    log('  {:>4s} {:>8s} {:>12s} {:>12s} {:>13s} {:>12s}'
        .format('шар','екон.','похибка C','похибка m','зсув MSE','score'))
    for l in (0,1,2,4,8,16,24,28):
        SM=torch.zeros(D,dtype=torch.float64,device=dev)
        SC=torch.zeros(D,D,dtype=torch.float64,device=dev); m=0
        for c in range(10):
            g=torch.Generator(device=dev).manual_seed(770000+c)
            z=torch.randn(20000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
            h=pre_at(W,xx,l).double(); SM+=h.sum(0); SC+=h.t()@h; m+=xx.shape[0]
        mT=SM/m; CT=SC/m-torch.outer(mT,mT)
        eC=float((CA[l]-CT).norm()/CT.norm()); eM=float((mA[l]-mT).norm()/(mT.norm()+1e-12))
        ev,V=torch.linalg.eigh(CA[l].cpu())
        Ch=(V@torch.diag(ev.clamp(min=1e-12).sqrt())).to(dev)
        SS=torch.zeros(D,dtype=torch.float64,device=dev); ns=0
        for c in range(10):
            g=torch.Generator(device=dev).manual_seed(660000+c)
            u=torch.randn(20000,D,device=dev,generator=g); uu=torch.cat([u,-u],0)
            h=(mA[l].unsqueeze(0)+uu.double()@Ch.t().double()).float()
            SS+=from_pre(W,h,l).double().sum(0); ns+=uu.shape[0]
        bias=float(((SS/ns-TH)**2).mean())
        Mspl=(6485.0-460.0)/max(1.0-l/32.0,1e-9)
        log('  {:4d} {:7.2f}× {:11.4f}% {:11.4f}% {:13.3e} {:12.3e}'
            .format(l,32.0/(32.0-l),100*eC,100*eM,bias,(2.5e-6*6485/max(Mspl,1.0)+bias)*0.10))
    log('  орієнтир: теперішній score 2.5e-7; ідеал «K безкоштовна й ТОЧНА» = 1.15e-7')
log('done.')
