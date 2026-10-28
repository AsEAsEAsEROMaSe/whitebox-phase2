"""СКІЛЬКИ СКЛАДНОСТІ ДАЄ КОЖЕН ШАР ReLU (ідея юзера: розібрати пошарово).

БАЗОВИЙ ФАКТ: без ReLU мережа лінійна, a^L = x·M, отже E[a^L] = E[x]·M = 0 ТОЧНО.
v2 антитетична -> на парі (x,−x) дає xM+(−x)M = 0 ПОБІТОВО. MSE = 0, не «мала».
⟹ УСЯ складність задачі — від ReLU. Питання: як вона наростає по шарах?

МІРЯЄМО відносну MSE методу v2 (ZCA+антитетика) для:
  A. ReLU на ПЕРШИХ k шарах, решта лінійні      (k = 0,1,2,4,8,16,24,32)
  B. ReLU на ОСТАННІХ k шарах, решта лінійні

Відносна MSE = MSE / mean(E[a^L]²) — щоб масштаб лінійного хвоста ділився.
(Лінійний хвіст множить і відповідь, і похибку — абсолютні числа непорівнянні.)
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/relu_ladder.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd(W,x,mask):
    """mask[li]=True -> ReLU на цьому шарі, False -> лінійний"""
    a=x
    for li in range(L):
        a=a@W[li]
        if mask[li]: a=relu(a)
    return a

def truth(W,mask,N=4_000_000,CH=200_000,seed=770):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=g); x=torch.cat([z,-z],0)
        s+=fwd(W,x,mask).double().sum(0); n+=x.shape[0]
    return s/n

def v2(W,mask,npair,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g); x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    return fwd(W,x@Wh,mask).double().mean(0)

NPAIR=100_000; REPS=8
KS=[0,1,2,3,4,6,8,12,16,24,32]
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    log(''); log('######## НЕТ {}  (N={}, {} повторів) ########'.format(net,2*NPAIR,REPS))
    for order in ('ПЕРШІ','ОСТАННІ'):
        log('')
        log('  ReLU на {} k шарах:'.format(order))
        log('    k    ||E[a^L]||      відн.MSE       абс.MSE      у скільки разів гірше за k=1')
        ref=None
        for k in KS:
            mask=[False]*L
            if order=='ПЕРШІ':
                for i in range(k): mask[i]=True
            else:
                for i in range(L-k,L): mask[i]=True
            TH=truth(W,mask,seed=770+net*97+k)
            errs=[]
            for r in range(REPS):
                e=v2(W,mask,NPAIR,1200+net*53+k*7+r)
                errs.append(float(((e-TH)**2).mean()))
            mse=float(np.mean(errs)); p=float((TH*TH).mean())
            rel=mse/p if p>0 else float('nan')
            if k==1: ref=rel
            fac='' if (ref is None or k==0) else '{:8.2f}×'.format(rel/ref)
            log('   {:2d}    {:.6e}   {:.4e}   {:.4e}   {}'.format(
                k, float(TH.norm()), rel, mse, fac))
            if k==0:
                log('        ^ ЛІНІЙНА: E[a^L] має бути 0 і MSE рівно 0 (антитетика точна)')
log('done.')
