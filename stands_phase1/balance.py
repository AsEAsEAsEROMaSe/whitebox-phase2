"""БАЛАНСУВАННЯ ВАГ (ентропійне) проти регресійної CV проти простого середнього.

Приблуда, що тримає рівновагу: ваги w_i зсуваються мінімально (за KL до рівномірних) так,
щоб зважені середні ознак із ТОЧНО ВІДОМИМ середнім стали рівно нулем.
   min Σ w log(wN)   s.t.  Σ w g_i = 0,  Σ w = 1
   двоїста: w_i ∝ exp(λ^T g_i);  λ з ньютона по опуклій L(λ)=log mean exp(λ^T g)

v2 = жорсткі умови (антитетика: непарні=0; вибілювання: коваріація=I).
Тут додаємо М'ЯКІ умови на He_4/He_6 у НАПРЯМКАХ МЕРЕЖІ (узгоджених, не випадкових).
Балансування != проєкція: ловить члени, яких лінійна CV не бачить.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/balance.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def he(t,p):
    """імовірнісний Ерміт He_p (рекурсія); E[He_p(N(0,1))]=0 для p>=1"""
    h0=torch.ones_like(t); h1=t
    if p==0: return h0
    if p==1: return h1
    for k in range(2,p+1):
        h0,h1=h1,t*h1-(k-1)*h0
    return h1

def truth(W,N=8_000_000,CH=200_000):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(4242+c))
        x=torch.cat([z,-z],0)
        for li in range(L): x=relu(x@W[li])
        s+=x.double().sum(0); n+=x.shape[0]
    return s/n

_GHX,_GHW=np.polynomial.hermite_e.hermegauss(120)
_GHW=_GHW/_GHW.sum()
def he_np(t,p):
    h0=np.ones_like(t); h1=t
    if p==0: return h0
    if p==1: return h1
    for k in range(2,p+1): h0,h1=h1,t*h1-(k-1)*h0
    return h1
def exact_mean(p,squash):
    """E[psi(s)] точною 1-D квадратурою (s~N(0,1) РІВНО)"""
    v=he_np(_GHX,p)/math.sqrt(math.factorial(p))
    if squash: v=np.tanh(v)
    return float((_GHW*v).sum())

def batch(W,npair,seed,U,degs=(4,6),squash=False):
    """v2-стиль: антитетика + ZCA-вибілювання. Повертає a^L і ознаки Ерміта."""
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g)
    x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    x=x@Wh                                   # емп.коваріація = I ТОЧНО
    s=x@U                                    # проєкції на напрямки мережі (одиничні)
    cols=[]
    for d in degs:
        v=he(s,d)/math.sqrt(math.factorial(d))
        if squash: v=torch.tanh(v)
        cols.append(v-exact_mean(d,squash))     # ЦЕНТРУВАННЯ точним середнім
    F=torch.cat(cols,1)
    a=x
    for li in range(L): a=relu(a@W[li])
    return a, F.double()          # a лишається float32 (пам'ять)

def balance(F,iters=60,ridge=1e-8):
    """ентропійне балансування: w ∝ exp(λ^T g), λ ньютоном"""
    N,k=F.shape
    sd=F.std(0)+1e-12; G=F/sd                     # лише обумовленість
    lam=torch.zeros(k,dtype=torch.float64,device=F.device)
    for it in range(iters):
        u=G@lam; u=u-u.max()
        w=torch.exp(u); w=w/w.sum()
        gbar=w@G
        if float(gbar.abs().max())<1e-12: break
        H=(G*w[:,None]).t()@G-torch.outer(gbar,gbar)
        Hc=(H+ridge*torch.eye(k,dtype=torch.float64,device=F.device)).cpu()
        step=torch.linalg.solve(Hc,gbar.cpu()).to(F.device)     # cusolver падає -> CPU
        # демпфування: гарантуємо спад цілі
        t=1.0
        base=float(torch.logsumexp(G@lam,0))
        for _ in range(30):
            cand=lam-t*step
            if float(torch.logsumexp(G@cand,0))<=base: break
            t*=0.5
        lam=lam-t*step
    u=G@lam; u=u-u.max(); w=torch.exp(u); w=w/w.sum()
    return w

NPAIR=100_000; REPS=20
CONFIGS=[((4,),64,False),((4,6),64,False),((4,6),128,False),((4,6,8),64,False)]
for net in [0,1,2,3,4,5]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    TH=truth(W)
    U0=W[0]/W[0].norm(dim=0,keepdim=True)         # напрямки МЕРЕЖІ (1-й шар)
    log('NET {}  (N={} ПРОДАКШН-РОЗМІР, {} повторів)'.format(net,2*NPAIR,REPS))
    for degs,K,SQ in CONFIGS:
     U=U0[:,:K]
     e_plain=[];e_reg=[];e_bal=[]
     for r in range(REPS):
        a,F=batch(W,NPAIR,7000+100*net+r,U,degs,SQ)
        N=a.shape[0]
        plain=a.double().mean(0) if a.shape[0]<=40000 else (a.sum(0,dtype=torch.float64)/N)
        # регресійна CV на тих самих ознаках (та сама інформація, лінійна форма)
        Fc=F-F.mean(0)
        A=Fc.t()@Fc/N+1e-6*torch.eye(F.shape[1],dtype=torch.float64,device=dev)
        b=((Fc.float().t()@a).double()-torch.outer(Fc.sum(0),plain))/N
        beta=torch.linalg.solve(A.cpu(),b.cpu()).to(a.device)  # cusolver падає -> CPU
        reg=plain-(F.mean(0)@beta)
        w=balance(F)
        bal=(w.float()@a).double()
        del a,F; torch.cuda.empty_cache()
        e_plain.append(plain-TH); e_reg.append(reg-TH); e_bal.append(bal-TH)
     f=lambda e: float(torch.stack(e).pow(2).mean())
     mp,mr,mb=f(e_plain),f(e_reg),f(e_bal)
     log('   He{} K={:<3d} {:9s} ({:3d} умов): регр {:.3f}x | БАЛАНС {:.3f}x'.format(
        ''.join(str(d) for d in degs),K,'TANH' if SQ else 'сирий',K*len(degs),mp/mr,mp/mb))
log('done.')
