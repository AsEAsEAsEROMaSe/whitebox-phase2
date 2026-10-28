"""КАВЕРЗА БОТА: чи пласка крива відн.MSE — артефакт нормування?

relu_ladder.py робив скан з ЛІНІЙНИМ ХВОСТОМ після k шарів ReLU. Хвіст погано
обумовлений і роздуває дисперсію тим сильніше, чим він довший — тобто найбільше
при МАЛИХ k. Це могло підняти ліву частину кривої.

ЧИСТИЙ ТЕСТ: мережа глибини рівно k, вихід a^k, БЕЗ хвоста. Жодного спотворення.
Друкуємо ЧИСЕЛЬНИК і ЗНАМЕННИК ОКРЕМО (як просив бот):
   ||E[a^k]||²/D          — сигнал
   Var оцінки (абс.)      — шум оцінки
   CV² = Var(a)/mean(E²)  — коеф. варіації, від N не залежить
   відн.MSE               — те, що малювали раніше

Теза бота: «при k=1 сигналу майже нема, тому відношення схоже».
Моя: E[relu(g)]=0.4σ — сигнал НЕ малий; падіння CV з глибиною = самоусереднення.
Розсудить CV², бо вона нормована на сам сигнал і не залежить від N.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/ladder_notail.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd(W,x,k):
    a=x
    for li in range(k): a=relu(a@W[li])
    return a

def truth(W,k,N=4_000_000,CH=200_000,seed=770):
    s=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=g); x=torch.cat([z,-z],0)
        s+=fwd(W,x,k).double().sum(0); n+=x.shape[0]
    return s/n

def v2(W,k,npair,seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(npair,D,device=dev,generator=g); x=torch.cat([z,-z],0)
    C=(x.t()@x)/x.shape[0]
    ev,V=torch.linalg.eigh(C.double().cpu())
    Wh=(V@torch.diag(ev.clamp(min=1e-12).rsqrt())@V.t()).float().to(dev)
    a=fwd(W,x@Wh,k).double()
    return a.mean(0), a.var(0)

NPAIR=100_000; REPS=8; N=2*NPAIR
KS=[1,2,3,4,6,8,12,16,20,24,28,32]
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(32)]
    log(''); log('######## НЕТ {}  (БЕЗ хвоста, N={}) ########'.format(net,N))
    log('   k   сигнал mean(E²)   Var оцінки(абс)   CV²=Var(a)/mean(E²)   відн.MSE')
    for k in KS:
        TH=truth(W,k,seed=770+net*97+k)
        sig=float((TH*TH).mean())
        es=[];vs=[]
        for r in range(REPS):
            e,va=v2(W,k,NPAIR,1200+net*53+k*7+r)
            es.append(float(((e-TH)**2).mean())); vs.append(float(va.mean()))
        mse=float(np.mean(es)); vpers=float(np.mean(vs))
        log('  {:2d}    {:.4e}       {:.4e}        {:.4e}          {:.4e}'.format(
            k, sig, mse, vpers/sig, mse/sig))
log('')
log('ЧИТАННЯ: якщо CV² ПАДАЄ з k -> самоусереднення РЕАЛЬНЕ, крива не артефакт.')
log('         якщо CV² стала -> бот має рацію, «пласка» нічого не означає.')
log('done.')
