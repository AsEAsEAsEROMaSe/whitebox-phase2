"""Перевірка двох ланок: (1) J·x = a^L (Ейлер, однорідність степеня 1)
(2) тотожність Стайна для u=J^T c ПРОВАЛЮЄТЬСЯ рівно на величину відповіді
    -> уся інформація в граничних (поверхневих) доданках."""
import torch, math, numpy as np, time
LOG='/mnt/win_d/llm/White-Box/euler.log'; T0=time.time()
def log(m):
    l='[{:5.1f}s] {}'.format(time.time()-T0,m); print(l,flush=True); open(LOG,'a').write(l+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(dat['w_0_%d'%i].astype(np.float64),device=dev,dtype=torch.float64) for i in range(L)]
def fwd_jvp(x,v):
    a=x;t=v
    for li in range(L):
        h=a@W[li]; th=t@W[li]; g=(h>0).to(h.dtype); a=h*g; t=th*g
    return a,t
g=torch.Generator(device=dev).manual_seed(1)
x=torch.randn(4096,D,device=dev,generator=g,dtype=torch.float64)
a,Jx=fwd_jvp(x,x)                       # v = x  ->  t = J·x
log('1. ЕЙЛЕР  J·x = a^L ?')
log('   ||J·x - a||/||a|| = {:.3e}   (нуль => однорідність степеня 1 точна)'.format(
    float((Jx-a).norm()/a.norm())))
log('')
log('2. ЧИ мала б тотожність Стайна дати нуль (u = J^T c, Δφ=0 всередині комірок)?')
c=torch.randn(D,device=dev,generator=g,dtype=torch.float64); c=c/c.norm()
BL=20; CH=20000; vals=[]
for b in range(BL):
    gg=torch.Generator(device=dev).manual_seed(600+b)
    z=torch.randn(CH//2,D,device=dev,generator=gg,dtype=torch.float64)
    xx=torch.cat([z,-z],0)
    aa,Jxx=fwd_jvp(xx,xx)
    vals.append(float((aa@c).mean()))    # = E[x·u] бо x·(J^T c)=(Jx)·c=a·c
V=torch.tensor(vals); m=float(V.mean()); se=float(V.std()/math.sqrt(BL))
log('   E[x·u] = E[c·a^L] = {:+.6f} ± {:.6f}   |z| = {:.1f}'.format(m,se,abs(m)/se))
log('   Стайн вимагав би НУЛЯ. Розбіжність = {:.1f} сигм = ГРАНИЧНІ доданки.'.format(abs(m)/se))
log('   => уся відповідь сидить на поверхнях перемикання затворів.')
log('done.')
