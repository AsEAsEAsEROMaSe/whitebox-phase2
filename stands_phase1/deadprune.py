"""ПОРТАЛ У НУЛЬ (метафора юзера): якщо нейрон прибитий у нуль — він НЕ ВПЛИВАЄ НІКОЛИ,
і робота на нього витрачена марно. Викинути його з мережі = ТОЧНА операція, не наближення.

ЧОМУ ЦЕ Б'Є В ЦІЛЬ: бюджет фіксований, N скорочується точно ⟹ важить ЦІНА ОДНОГО СЕМПЛА
([[reference-score-formula-cost]]). Прибрати частку q нейронів = флопи ×(1−q)² на шар
⟹ стільки ж більше семплів за ті самі гроші ⟹ score покращується прямо пропорційно.
І це лишається ВЕЛИКИМ ЩІЛЬНИМ МАТМУЛОМ — форма правильна ([[finding-flopscope-call-tax]]).

МІРЯЄМО:
 1) p_on_i по кожному шару — частка семплів, де нейрон відкритий
 2) скільки нейронів мертві (p_on нижче порога) — по шарах
 3) РЕАЛЬНИЙ прогін обрізаної мережі проти істини: зсув MSE
 4) економія флопів і що це дає в score
"""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time()
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(n): return [torch.tensor(dat['w_%d_%d'%(n,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(Ws,x):
    a=x
    for W in Ws: a=torch.relu(a@W)
    return a

for net in (0,1):
    W=load(net); log('='*76); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(20):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(50000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=fwd(W,xx).double().sum(0); n+=xx.shape[0]
    TH=S/n
    # p_on по шарах
    NP=200000; ON=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    for c in range(NP//20000):
        g=torch.Generator(device=dev).manual_seed(123000+c)
        x=torch.randn(20000,D,device=dev,generator=g); a=x
        for l in range(L):
            h=a@W[l]; ON[l]+=(h>0).double().sum(0); a=torch.relu(h)
    P=[o/NP for o in ON]
    log('  частка нейронів із p_on нижче порога (по шарах):')
    log('  {:>5s} {:>9s} {:>9s} {:>9s} {:>9s} {:>11s}'.format('шар','p_on=0','<1e-5','<1e-4','<1e-3','сер. p_on'))
    tot0=0
    for l in (0,1,2,4,8,12,16,20,24,28,30,31):
        p=P[l]
        log('  {:5d} {:8.1f}% {:8.1f}% {:8.1f}% {:8.1f}% {:10.3f}'
            .format(l,100*float((p==0).double().mean()),100*float((p<1e-5).double().mean()),
                    100*float((p<1e-4).double().mean()),100*float((p<1e-3).double().mean()),float(p.mean())))
    for thr in (0.0,1e-5,1e-4,1e-3):
        keep=[(P[l]>thr) for l in range(L)]
        # обрізаємо: стовпці W[l] і рядки W[l+1]
        Ws=[];prev=None
        for l in range(L):
            Wl=W[l]
            if prev is not None: Wl=Wl[prev]
            if l<L-1: Wl=Wl[:,keep[l]]
            Ws.append(Wl); prev=keep[l] if l<L-1 else None
        flops=sum(Ws[l].shape[0]*Ws[l].shape[1] for l in range(L))
        full=L*D*D
        Sp=torch.zeros(D,dtype=torch.float64,device=dev); m=0
        for c in range(10):
            g=torch.Generator(device=dev).manual_seed(555000+c)
            z=torch.randn(50000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
            Sp+=fwd(Ws,xx).double().sum(0); m+=xx.shape[0]
        bias=float(((Sp/m-TH)**2).mean())
        kept=sum(int(k.sum()) for k in keep[:L-1])/(D*(L-1))
        log('  поріг {:6.0e}: лишилось {:5.1f}% нейронів, флопів {:5.1f}%, зсув MSE {:9.3e}, score×{:.3f}'
            .format(thr,100*kept,100*flops/full,bias,flops/full))
