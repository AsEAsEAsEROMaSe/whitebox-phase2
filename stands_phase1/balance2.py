"""balance.py, ВИПРАВЛЕНИЙ (аудит С48): оригінал захлинувся, лог = 1 рядок.
ЩО ЗМІНЕНО: демпфування векторизоване, ньютонів 25, і головне —
міряємо на ДВОХ N: 6000 (СПРАВЖНІЙ продакшн v2) і 200000 (те, що оригінал звав «продакшн»).
БАЗА = plain, а batch() уже робить антитетику+ZCA ⟹ plain ЦЕ І Є v2 ⟹ відношення = виграш НАД v2.
Умови м'які на He4/He6 у напрямках мережі; їхні точні середні = 0 (безкоштовно й ТОЧНО).
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/balance2.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def he(t,p):
    h0=torch.ones_like(t); h1=t
    if p==0: return h0
    if p==1: return h1
    for k in range(2,p+1): h0,h1=h1,t*h1-(k-1)*h0
    return h1
def truth(W,N=8_000_000,CH=200_000):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(4242+c))
        x=torch.cat([z,-z],0)
        for li in range(L): x=relu(x@W[li])
        s+=x.double().sum(0); n+=x.shape[0]
    return s/n
def batch(W,npair,seed,U,degs):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g)
    x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    x=x@Wh
    s=x@U
    cols=[he(s,d)/math.sqrt(math.factorial(d)) for d in degs]   # точне середнє = 0
    F=torch.cat(cols,1)
    a=x
    for li in range(L): a=relu(a@W[li])
    return a, F.double()
def balance(F,iters=25,ridge=1e-8):
    N,k=F.shape
    sd=F.std(0)+1e-12; G=F/sd
    lam=torch.zeros(k,dtype=torch.float64,device=F.device)
    Glam=G@lam
    for it in range(iters):
        u=Glam-Glam.max(); w=torch.exp(u); w=w/w.sum()
        gbar=w@G
        if float(gbar.abs().max())<1e-12: break
        H=(G*w[:,None]).t()@G-torch.outer(gbar,gbar)
        step=torch.linalg.solve((H+ridge*torch.eye(k,dtype=torch.float64,device=F.device)).cpu(),
                                gbar.cpu()).to(F.device)
        Gstep=G@step                                  # ОДИН матмул на всі демпфування
        base=float(torch.logsumexp(Glam,0)); t=1.0
        for _ in range(30):
            cand=Glam-t*Gstep
            if float(torch.logsumexp(cand,0))<=base: break
            t*=0.5
        lam=lam-t*step; Glam=Glam-t*Gstep
    u=Glam-Glam.max(); w=torch.exp(u); w=w/w.sum()
    return w
CONFIGS=[((4,),64),((4,6),64),((4,6),128)]
for net in (0,1):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    TH=truth(W); U0=W[0]/W[0].norm(dim=0,keepdim=True)
    for NPAIR,REPS in ((3_000,40),(100_000,12)):
        log('NET {}  N={} семплів ({}), {} повторів'.format(
            net,2*NPAIR,'ПРОДАКШН v2' if NPAIR==3000 else 'те, що оригінал звав продакшном',REPS))
        for degs,K in CONFIGS:
            U=U0[:,:K]; ep=[];er=[];eb=[]
            for r in range(REPS):
                a,F=batch(W,NPAIR,7000+100*net+r,U,degs)
                N=a.shape[0]
                plain=a.sum(0,dtype=torch.float64)/N
                Fc=F-F.mean(0)
                A=Fc.t()@Fc/N+1e-6*torch.eye(F.shape[1],dtype=torch.float64,device=dev)
                b=((Fc.float().t()@a).double()-torch.outer(Fc.sum(0),plain))/N
                beta=torch.linalg.solve(A.cpu(),b.cpu()).to(dev)
                reg=plain-(F.mean(0)@beta)
                bal=(balance(F).float()@a).double()
                ep.append(plain-TH); er.append(reg-TH); eb.append(bal-TH)
                del a,F; torch.cuda.empty_cache()
            f=lambda e: float(torch.stack(e).pow(2).mean())
            mp,mr,mb=f(ep),f(er),f(eb)
            log('   He{} K={:<4d}({:3d} умов): v2={:.3e} | регр {:.3f}x | БАЛАНС {:.3f}x'
                .format(''.join(map(str,degs)),K,K*len(degs),mp,mp/mr,mp/mb))
log('done.')
