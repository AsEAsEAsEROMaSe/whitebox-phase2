"""АРКСИНУС-CV У ПОВНІЙ КОНФІГУРАЦІЇ v2 (правило feedback-isolated-bench-lies).

Чистий стенд дав R^2=2.1-2.8% (антитетика на СИРИХ гаусових). Питання:
  (1) чи лишається щось ПОВЕРХ ZCA (вибілювання вже прибирає 22% і теж чіпає 2-й момент)?
  (2) чи не ЗМІЩУЄ це оцінювач: під ZCA вхід уже не строго N(0,Sig), а якір arcsin(rho)
      виведено саме для N(0,Sig). Зміщення фатальне (виграш дисперсії його не викупить).

СХЕМА: істина 8М семплів (антитетика). Далі R повторів v2 з різними сідами:
  base = mean(a^L)                        — це і є v2
  cv   = base - beta^T (mean(f) - 0)      — f центровано ТОЧНИМ якорем
  beta НАВЧАЄМО НА ОКРЕМОМУ сіді (не на тих самих семплах) => чесно, без переобучення.
МІРЯЄМО: MSE(base), MSE(cv), а також ЗМІЩЕННЯ mean(f) під ZCA проти нуля (в сигмах).
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/arcsine_full.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def zca(x):
    C=(x.t()@x)/x.shape[0]
    e,V=torch.linalg.eigh(C.double().cpu())
    return (V@torch.diag(e.clamp(min=1e-12).rsqrt())@V.t()).to(dev).float()

def prep(net,M):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    W0d=W[0].double(); Sig=W0d.t()@W0d; dg=Sig.diag().rsqrt()
    A=torch.arcsin(((Sig*dg[:,None])*dg[None,:]).clamp(-1,1))/(2*math.pi)
    ev,V=torch.linalg.eigh(A.cpu())
    lam=ev.flip(0)[:M].to(dev); Vf=V.flip(1)[:,:M].to(dev).float()
    iu=torch.triu_indices(M,M,device=dev)
    anc=torch.where(iu[0]==iu[1],lam[iu[0]],torch.zeros(iu.shape[1],dtype=lam.dtype,device=dev))
    return W,Vf,iu,anc

def fwd_feat(W,Vf,iu,anc,x):
    """повертає (парна частина a^L, ознаки f) для антитетичної пари x=[z;-z]"""
    n=x.shape[0]//2
    h0=x@W[0]; u=(h0>0).float()-0.5
    pz=(u[:n]@Vf).double()
    f=pz[:,iu[0]]*pz[:,iu[1]]-anc
    a=torch.relu(h0)
    for li in range(1,L): a=torch.relu(a@W[li])
    return ((a[:n]+a[n:])*0.5).double(), f

def sample(W,Vf,iu,anc,npair,seed,white):
    gg=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=gg)
    x=torch.cat([z,-z],0)
    if white: x=x@zca(x)
    return fwd_feat(W,Vf,iu,anc,x)

def truth(W,Vf,iu,anc,N=8_000_000,CH=200_000,seed=6100):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        y,_=sample(W,Vf,iu,anc,CH//2,seed+c,False)
        s+=y.sum(0); n+=y.shape[0]
    return s/n

NP=100_000; REPS=10
for M in (24,48):
    log('======== M={} мод ({} ознак) ========'.format(M,M*(M+1)//2))
    for net in (0,1):
        W,Vf,iu,anc=prep(net,M)
        TH=truth(W,Vf,iu,anc)
        # --- beta вчимо на ОКРЕМОМУ сіді, у ПОВНІЙ (вибіленій) конфігурації ---
        Y,F=sample(W,Vf,iu,anc,200_000,777_000+net,True)
        Yc=Y-Y.mean(0); sd=F.std(0).clamp(min=1e-12); Fs=(F-F.mean(0))/sd
        G=Fs.t()@Fs/Fs.shape[0]; I=torch.eye(G.shape[0],dtype=G.dtype,device=dev)
        beta=torch.linalg.solve(G+1e-2*I,Fs.t()@Yc/Fs.shape[0])
        # --- ЗМІЩЕННЯ якоря під ZCA: середнє f має бути 0 ---
        z_sig=float((F.mean(0)/(F.std(0)/F.shape[0]**0.5)).abs().max())
        log(' нет{}  зміщення якоря під ZCA: max|mean(f)|/sigma = {:.1f}  {}'
            .format(net,z_sig,'ОК' if z_sig<4 else '<<< ЗМІЩЕНО'))
        eb=[];ec=[]
        for r in range(REPS):
            Yr,Fr=sample(W,Vf,iu,anc,NP,3300+r,True)
            base=Yr.mean(0)
            corr=((Fr.mean(0)-F.mean(0)*0)/sd)@beta      # якір = ТОЧНИЙ нуль
            cv=base-corr
            eb.append(float(((base-TH)**2).mean())); ec.append(float(((cv-TH)**2).mean()))
        mb,mc=float(np.mean(eb)),float(np.mean(ec))
        log(' нет{}  v2={:.4e}   v2+arcsinCV={:.4e}   ЗМІНА={:+.1f}%'
            .format(net,mb,mc,100*(mc/mb-1)))
log('done.')
