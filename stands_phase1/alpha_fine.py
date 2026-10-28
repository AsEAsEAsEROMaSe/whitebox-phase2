"""ЧИ Є ГЛАДКИЙ РУХ ЛІНІЙНА -> ReLU?  (питання юзера, С47)

Перший скан (alpha_dial.py) мав ДВІ вади:
  1) стрибок 1.00 -> 0.99, між ними порожньо — саме там мав би жити «схил»;
  2) мережа З'ЇЖДЖАЛА З КРИТИЧНОСТІ: підсилення шару = (1+a^2), при a->1 це 2,
     за 32 шари 2^32 — тому колонка «сигнал» стрибала на 35 порядків.

ВИПРАВЛЕННЯ: тримаємо критичність для КОЖНОГО a.
   relu_a(z), z~N(0,s^2):  E[relu_a^2] = s^2 (1+a^2)/2
   ваги N(0, c/d):  s^2_next = c * E[relu_a^2]  =>  c = 2/(1+a^2)
   банк має c=2 (для a=0), отже множимо ваги на  sqrt(1/((1+a^2)/2)).

МІРЯЄМО як функцію eps=(1-a)/2 (сила нелінійності):
   sig  = mean(E[a^L]^2)     — очікування: sig ~ eps^2 (перший порядок delta-розкладу)
   mse  = помилка v2         — очікування: теж ~ eps^2 (антитетика вбиває лінійну частину ТОЧНО)
   rel  = mse/sig            — якщо ПЛАТО аж до eps=5e-5, малий параметр самознищується
                               остаточно; якщо ПАДАЄ — біля a=1 є справжній схил.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/alpha_fine.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W0=[torch.tensor(dat['w_0_%d'%i].astype(np.float32),device=dev).double() for i in range(L)]

def fwd(W,x,al):
    a=x
    for li in range(L):
        h=a@W[li]; a=torch.maximum(h,al*h)
    return a

def truth(W,al,N=2_000_000,CH=200_000,seed=770):
    s=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(N//CH):
        gg=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=gg,dtype=DT)
        xx=torch.cat([z,-z],0); s+=fwd(W,xx,al).sum(0); n+=xx.shape[0]
    return s/n

def v2(W,al,npair,seed):
    gg=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=gg,dtype=DT); xx=torch.cat([z,-z],0)
    C=(xx.t()@xx)/xx.shape[0]
    e_,V=torch.linalg.eigh(C.cpu()); Wh=(V@torch.diag(e_.clamp(min=1e-14).rsqrt())@V.t()).to(dev)
    return fwd(W,xx@Wh,al).mean(0)

log('КРИТИЧНІСТЬ ТРИМАЄМО: ваги *= sqrt(2/(1+a^2))/sqrt(2)')
log('   a          eps=(1-a)/2   сигнал mean(E^2)   абс.MSE       відн.MSE      sig/eps^2')
NP=50_000; REPS=6
for al in (1.0,0.9999,0.999,0.99,0.95,0.9,0.7,0.5,0.3,0.1,0.0):
    g=(2.0/(1.0+al*al))**0.5/(2.0**0.5)          # множник на ваги
    W=[w*g for w in W0]
    eps=(1-al)/2
    TH=truth(W,al,seed=770+int(al*1000))
    sig=float((TH*TH).mean())
    es=[float(((v2(W,al,NP,3300+r)-TH)**2).mean()) for r in range(REPS)]
    mse=float(np.mean(es))
    rel=mse/sig if sig>1e-300 else float('nan')
    law=sig/(eps*eps) if eps>0 else float('nan')
    log('  {:.4f}     {:.3e}     {:.4e}      {:.4e}    {:.4e}   {:.4e}'.format(al,eps,sig,mse,rel,law))
log('')
log('ЧИТАННЯ: sig/eps^2 стале => сигнал РІВНО першого порядку по нелінійності.')
log('         відн.MSE стале  => схилу нема, сходинка підтверджена ДО eps=5e-5.')
log('done.')
