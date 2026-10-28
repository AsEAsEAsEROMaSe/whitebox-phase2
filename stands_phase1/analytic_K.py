"""АНАЛІТИЧНА K БЕЗ ЖОДНОГО СЕМПЛА → ГІБРИД: аналітика до шару ℓ, семпли після.

Продовження [[finding-synthetic-deep-start]]: схема синтетичного старту працює, зсув
падає з глибиною, АЛЕ купівля K семплами коштує ×3 бюджету. Тут K береться АНАЛІТИЧНО.

⚠️ ВАЖЛИВО — це НЕ звичайний NNGP. У NNGP K береться по випадкових ВАГАХ при фіксованому
вході. Нам треба навпаки: розподіл h^ℓ по випадковому ВХОДУ при ФІКСОВАНИХ вагах.
  шар 0:  h⁰ = x·W₀ — ТОЧНО гаусів, m=0, C = W₀ᵀW₀.  Безкоштовно й ТОЧНО.
  крок:   (m,C) для h ⟹ (μ,Σ) для a=relu(h) ⟹ m' = μ·W,  C' = Wᵀ Σ W
          μ_i = m_iΦ(z_i)+s_iφ(z_i);  Σ_ij через 2D квадратуру Гаусса-Ерміта
  ЗАМИКАННЯ: на кожному кроці h вважається гаусовим (єдине наближення).

ЧОМУ ГІБРИД, А НЕ ЧИСТА АНАЛІТИКА: якщо дійти аналітикою до 31-го, відповідь виходить
у замкненій формі — і це рівно старий NNGP (записано: MSE 1.65e-4, погано). Сенс саме в
ГІБРИДІ: аналітика до ℓ, а шари ℓ+1..31 проганяються СЕМПЛАМИ з СПРАВЖНІМ ReLU.
  вартість семпла = (32−ℓ)/32 прогону + разова аналітика
  НАПРУГА: похибка аналітики РОСТЕ з ℓ, економія теж РОСТЕ з ℓ ⟹ шукаємо оптимум.

ВАРТІСТЬ АНАЛІТИКИ (разова): 32×(2d³ на WᵀΣW + d²Q² на квадратуру) ≈ 1.9e9 флоп ≈ 460
прогонів із 6485 = 7% бюджету. Прийнятно.
"""
import torch, time, math, numpy as np
LOG='/mnt/win_d/llm/White-Box/analytic_K.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32; Q=24
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

# вузли Гаусса-Ерміта для ймовірнісної нормалі
_t,_w=np.polynomial.hermite_e.hermegauss(Q)
GT=torch.tensor(_t,dtype=torch.float64,device=dev)
GW=torch.tensor(_w/np.sqrt(2*np.pi),dtype=torch.float64,device=dev)
SQ2PI=math.sqrt(2*math.pi)

def relu_moments(m,C):
    """(m,C) гаусового h ⟹ μ=E[relu(h)], Σ=Cov(relu(h))"""
    s=C.diagonal().clamp(min=1e-20).sqrt()
    z=m/s
    Phi=0.5*(1+torch.erf(z/math.sqrt(2))); phi=torch.exp(-0.5*z*z)/SQ2PI
    mu=m*Phi+s*phi
    rho=(C/torch.outer(s,s)).clamp(-1+1e-9,1-1e-9)
    E2=torch.empty(D,D,dtype=torch.float64,device=dev)
    u=GT.view(1,1,Q,1); v=GT.view(1,1,1,Q); wq=(GW.view(1,1,Q,1)*GW.view(1,1,1,Q))
    CH=16
    for a0 in range(0,D,CH):
        mi=m[a0:a0+CH].view(-1,1,1,1); si=s[a0:a0+CH].view(-1,1,1,1)
        r=rho[a0:a0+CH].view(-1,D,1,1)
        hi=torch.relu(mi+si*u)
        hj=torch.relu(m.view(1,D,1,1)+s.view(1,D,1,1)*(r*u+torch.sqrt(1-r*r)*v))
        E2[a0:a0+CH]=(wq*hi*hj).sum((2,3))
    return mu, E2-torch.outer(mu,mu)

for net in (0,1):
    W=load(net); log('='*86); log('МЕРЕЖА %d'%net)
    # істина
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(30):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(60000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=from_pre(W,pre_at(W,xx,0),0).double().sum(0); n+=xx.shape[0]
    TH=S/n

    # ---- аналітичний прохід ----
    ta=time.time()
    W0=W[0].double(); mA={0:torch.zeros(D,dtype=torch.float64,device=dev)}; CA={0:W0.t()@W0}
    for l in range(0,L-1):
        mu,Sg=relu_moments(mA[l],CA[l])
        Wl=W[l+1].double()
        mA[l+1]=mu@Wl; CA[l+1]=Wl.t()@Sg@Wl
    log('  аналітичний прохід: {:.1f}с (разово)'.format(time.time()-ta))

    log('  {:>4s} {:>8s} {:>12s} {:>12s} {:>13s} {:>12s}'
        .format('шар','екон.','похибка C','похибка m','зсув MSE','score'))
    for l in (0,1,2,4,8,16,24,28):
        # виміряна (m,C) для порівняння
        SM=torch.zeros(D,dtype=torch.float64,device=dev)
        SC=torch.zeros(D,D,dtype=torch.float64,device=dev); m=0
        for c in range(10):
            g=torch.Generator(device=dev).manual_seed(770000+c)
            z=torch.randn(20000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
            h=pre_at(W,xx,l).double(); SM+=h.sum(0); SC+=h.t()@h; m+=xx.shape[0]
        mT=SM/m; CT=SC/m-torch.outer(mT,mT)
        eC=float((CA[l]-CT).norm()/CT.norm()); eM=float((mA[l]-mT).norm()/(mT.norm()+1e-12))
        # гібрид: семпли з АНАЛІТИЧНОЇ (m,C), далі справжні шари
        ev,V=torch.linalg.eigh(CA[l].cpu())
        Ch=(V@torch.diag(ev.clamp(min=1e-12).sqrt())).to(dev)
        SS=torch.zeros(D,dtype=torch.float64,device=dev); ns=0
        for c in range(10):
            g=torch.Generator(device=dev).manual_seed(660000+c)
            u=torch.randn(20000,D,device=dev,generator=g); uu=torch.cat([u,-u],0)
            h=(mA[l].unsqueeze(0)+uu.double()@Ch.t().double()).float()
            SS+=from_pre(W,h,l).double().sum(0); ns+=uu.shape[0]
        bias=float(((SS/ns-TH)**2).mean())
        Mspl=(6485.0-460.0)/max(1.0-l/32.0,1e-9)          # 460 прогонів на аналітику
        mse=2.5e-6*6485/max(Mspl,1.0)+bias
        log('  {:4d} {:7.2f}× {:11.4f}% {:11.4f}% {:13.3e} {:12.3e}'
            .format(l,32.0/(32.0-l),100*eC,100*eM,bias,mse*0.10))
    log('  орієнтир: теперішній score 2.5e-7; ідеал «K безкоштовна й ТОЧНА» = 1.15e-7')
log('done.')
