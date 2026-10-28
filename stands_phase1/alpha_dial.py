"""ВІД МАШИННОЇ ТОЧНОСТІ ДО 200k СЕМПЛІВ — як саме одне переходить в інше?

ФАКТ (С46): без ReLU MSE = 1.7e-31, тобто МАШИННИЙ НУЛЬ. Задача розв'язується
ТОЧНО двома семплами. Вмикаємо ReLU — треба 200k семплів на 3 цифри.
Між цими станами немає нічого, крім модуля.

РУЧКА ГУЧНОСТІ: «дірявий» ReLU з нахилом α на від'ємному боці
   relu_α(x) = max(x, αx) = ((1+α)/2)·x + ((1−α)/2)·|x|
   α=1 -> лінійна (точна);  α=0 -> наша.  Коефіцієнт при |·| = (1−α)/2 — це і є нелінійність.

МІРЯЄМО:
  0. ПОБІТОВУ тотожність f(−x) == −f(x) для лінійної мережі (перевірка мого пояснення)
  1. сигнал ||E[a^L]||² та відносну MSE методу v2 як функцію α
     -> якщо крива гладка, поблизу α=1 Є малий параметр (1−α); де він зникає?
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/alpha_dial.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd(W,x,al):
    a=x
    for li in range(L):
        h=a@W[li]; a=torch.maximum(h,al*h)
    return a

# ---------- 0) ПОБІТОВА ПЕРЕВІРКА для лінійної мережі ----------
W=[torch.tensor(dat['w_0_%d'%i].astype(np.float32),device=dev).double() for i in range(L)]
g=torch.Generator(device=dev).manual_seed(1)
x=torch.randn(64,D,device=dev,generator=g,dtype=DT)
p=fwd(W,x,1.0); m=fwd(W,-x,1.0)
SIGNBIT=torch.tensor(-(1<<63),device=dev,dtype=torch.int64)   # 1<<63 не влазить у int64
same=torch.equal(p.view(torch.int64)^SIGNBIT, m.view(torch.int64))
log('0) ПОБІТОВО: f(−x) є точним переворотом знаку f(x)?  {}'.format('ТАК' if same else 'НІ'))
log('   ||f(x)+f(−x)||_max = {:.3e}  (нуль => мантиси ІДЕНТИЧНІ, різниця лише біт знаку)'.format(
    float((p+m).abs().max())))
log('   -> залишок 2.4e-15 у скані бере початок НЕ в мережі, а в ПОРЯДКУ ПІДСУМОВУВАННЯ')

def truth(W,al,N=4_000_000,CH=200_000,seed=770):
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

log('')
log('1) РУЧКА ГУЧНОСТІ α  (нелінійність = (1−α)/2)')
log('   α       (1−α)/2     сигнал mean(E²)     абс.MSE        відн.MSE')
NP=50_000; REPS=6
for al in (1.0,0.99,0.95,0.9,0.8,0.6,0.4,0.2,0.1,0.05,0.0):
    TH=truth(W,al,seed=770+int(al*1000))
    sig=float((TH*TH).mean())
    es=[float(((v2(W,al,NP,3300+r)-TH)**2).mean()) for r in range(REPS)]
    mse=float(np.mean(es))
    rel=mse/sig if sig>1e-300 else float('nan')
    log('  {:.2f}     {:.3f}      {:.4e}      {:.4e}    {:.4e}'.format(al,(1-al)/2,sig,mse,rel))
log('')
log('ЧИТАННЯ: якщо сигнал ~ (1−α)^p — маємо ЗАКОН; якщо відн.MSE виходить на плато —')
log('         малий параметр (1−α) працює лише поблизу α=1 і зникає задовго до α=0.')
log('done.')
