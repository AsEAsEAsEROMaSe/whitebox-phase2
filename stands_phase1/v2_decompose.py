"""РОЗКЛАД v2 НА СКЛАДОВІ — скільки дає антитетика, скільки вибілювання.

signgroup.py: антитетика сама по собі дає ефективність 0.917, тобто виграш лише 8%.
А в пам'яті записано, що v2 = 0.47× від простого MC. Значить майже все — ВИБІЛЮВАННЯ.

ЧОМУ так (сьогоднішнє пояснення): антитетика ТОЧНО вбиває непарну частину (k=0 показав
побітово), але δ-розклад показав, що на глибині 32 та частина несе майже нуль дисперсії
(порядок 0 = точно нуль, порядок 1 = 2.6e-5 від відповіді).
⟹ антитетика ідеально прибирає те, чого майже немає.

ЧОТИРИ ВАРІАНТИ при ОДНАКОВІЙ кількості форвардів N:
  plain      — незалежні семпли
  anti       — лише антитетика (N/2 пар)
  zca        — лише ZCA-вибілювання
  v2         — обидва (прод)
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/v2_decompose.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd(W,x):
    a=x
    for li in range(L): a=relu(a@W[li])
    return a.double()

def whiten(x):
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    return x@Wh

def truth(W,N=6_000_000,CH=200_000):
    s=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(880+c))
        x=torch.cat([z,-z],0); s+=fwd(W,x).sum(0); n+=x.shape[0]
    return s/n

N=200_000; REPS=12
res={k:[] for k in ('plain','anti','zca','v2')}
for net in [0,1,2,3]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    TH=truth(W)
    for r in range(REPS):
        g=torch.Generator(device=dev).manual_seed(8800+net*41+r)
        xi=torch.randn(N,D,device=dev,generator=g)              # незалежні
        z=torch.randn(N//2,D,device=dev,generator=g)
        xa=torch.cat([z,-z],0)                                  # антитетичні
        est={'plain':fwd(W,xi).mean(0),
             'anti' :fwd(W,xa).mean(0),
             'zca'  :fwd(W,whiten(xi)).mean(0),
             'v2'   :fwd(W,whiten(xa)).mean(0)}
        for k,v in est.items(): res[k].append(float(((v-TH)**2).mean()))
    log('нет {} готовий'.format(net))
log('')
log('=== РОЗКЛАД v2 (4 нети × {} повторів, N={} форвардів у КОЖНОМУ) ==='.format(REPS,N))
mp=np.mean(res['plain'])
for k in ('plain','anti','zca','v2'):
    m=np.mean(res[k]); se=np.std(res[k])/math.sqrt(len(res[k]))
    log('  {:6s}  MSE {:.4e} ± {:.1e}   {:.4f}× від plain'.format(k,m,se,m/mp))
log('')
log('  внесок антитетики окремо : {:.4f}×'.format(np.mean(res['anti'])/mp))
log('  внесок вибілювання окремо: {:.4f}×'.format(np.mean(res['zca'])/mp))
log('  добуток окремих внесків  : {:.4f}×  (якби були незалежні)'.format(
    (np.mean(res['anti'])/mp)*(np.mean(res['zca'])/mp)))
log('  фактично разом (v2)      : {:.4f}×'.format(np.mean(res['v2'])/mp))
log('done.')
