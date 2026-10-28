"""ПАРНА перевірка обрізання + продакшн-облік із ЦІНОЮ ПІЛОТА.
Попередній стенд рахував істину й обрізану мережу РІЗНИМИ семплами ⟹ міряв власний шум.
Тут ТІ САМІ семпли крізь обидві мережі ⟹ шум скорочується побітово, лишається ЧИСТИЙ зсув.
Плюс: у продакшні мертві нейрони треба ЗНАЙТИ пілотом, і пілот коштує прогонів."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time()
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(Ws,x):
    a=x
    for W in Ws: a=torch.relu(a@W)
    return a
def prune(W,keep):
    Ws=[];prev=None
    for l in range(L):
        Wl=W[l]
        if prev is not None: Wl=Wl[prev]
        if l<L-1: Wl=Wl[:,keep[l]]
        Ws.append(Wl); prev=keep[l] if l<L-1 else None
    return Ws,sum(w.shape[0]*w.shape[1] for w in Ws)/(L*D*D)

for net in (0,1):
    W=load(net); log('='*80); log('МЕРЕЖА %d'%net)
    log('  --- ПІЛОТ малого розміру (як у продакшні) ---')
    log('  {:>7s} {:>10s} {:>9s} {:>13s} {:>13s} {:>9s}'
        .format('пілот','обрізано','флопів','ЧИСТИЙ зсув','score мн.','разом'))
    for NPIL in (128,256,512,1024,2048,200000):
        ON=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
        done=0
        while done<NPIL:
            b=min(20000,NPIL-done)
            g=torch.Generator(device=dev).manual_seed(770000+done)
            x=torch.randn(b,D,device=dev,generator=g); a=x
            for l in range(L):
                h=a@W[l]; ON[l]+=(h>0).double().sum(0); a=torch.relu(h)
            done+=b
        keep=[(ON[l]>0) for l in range(L)]
        Ws,fr=prune(W,keep)
        cut=1-sum(int(k.sum()) for k in keep[:L-1])/(D*(L-1))
        # ПАРНО: ті самі семпли крізь обидві мережі
        dS=torch.zeros(D,dtype=torch.float64,device=dev); m=0
        for c in range(10):
            g=torch.Generator(device=dev).manual_seed(444000+c)
            z=torch.randn(50000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
            dS+=(fwd(Ws,xx)-fwd(W,xx)).double().sum(0); m+=xx.shape[0]
        bias=float(((dS/m)**2).mean())
        # продакшн-облік: пілот повний коштом, решта — обрізаним
        B=6485.0
        N=(B-NPIL)/fr + NPIL          # пілотні прогони теж дають семпли
        mult=(B/N)+bias/2.5e-6*0      # варіаційна частина
        score=2.5e-6*(B/N)+bias
        log('  {:7d} {:9.1f}% {:8.1f}% {:13.3e} {:13.4f} {:9.4f}'
            .format(NPIL,100*cut,100*fr,bias,B/N,(score/2.5e-6)))
    log('  (score мн. = у скільки разів множиться MSE; <1 краще; зсув уже врахований у «разом»)')
