"""ЧИ ТРИВАЄ СПАД ЗСУВУ ДО САМОГО КІНЦЯ + ЧЕСНИЙ ОБЛІК ПІЛОТА.

synth_layer.py показав неочікуване: зсув гаусового старту ПАДАЄ з глибиною
(пік 2.2e-5 на шарі 4-8 → 2.6e-6 на шарі 28). Причина: на глибині розподіл найменш
гаусів, ЗАТЕ лишається мало шарів, щоб помилку розгойдати.
⟹ правильний хід — ЯКНАЙГЛИБШЕ. Дивимось 26..31.

ДРУГА ПАСТКА, яку тут же рахуємо: (m,K) на шарі ℓ треба ЗВІДКИСЬ УЗЯТИ. Пілотний семпл
доводиться гнати через ℓ шарів ⟹ коштує ℓ/32 прогону. Матриця 256×256 потребує тисяч
семплів. МІРЯЄМО зсув при РЕАЛІСТИЧНОМУ пілоті (2000/5000/20000), не при 800000.

ПОВНИЙ score-облік (підлога бюджету = 6485 прогонів):
  вартість = P·(ℓ/32) [пілот] + M·(1−ℓ/32) [синтетичні]  прогонів
  MSE = дисперсія/M + зсув(P)      ⟸ зсув НЕ падає з M
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/synth_deep.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def pre_at(W,x,l):
    a=x
    for i in range(l): a=torch.relu(a@w) if False else torch.relu(a@W[i])
    return a@W[l]
def from_pre(W,h,l):
    a=torch.relu(h)
    for i in range(l+1,L): a=torch.relu(a@W[i])
    return a

for net in (0,1):
    W=load(net); log('='*84); log('МЕРЕЖА %d'%net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(40):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(60000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=from_pre(W,pre_at(W,xx,0),0).double().sum(0); n+=xx.shape[0]
    TH=S/n
    log('  {:>4s} {:>7s} {:>9s} {:>12s} {:>12s} {:>12s}'
        .format('шар','екон.','пілот P','зсув MSE','вартість P','score оцінка'))
    for l in (26,28,29,30,31):
        for P in (2000,5000,20000,200000):
            SM=torch.zeros(D,dtype=torch.float64,device=dev)
            SC=torch.zeros(D,D,dtype=torch.float64,device=dev); m=0
            CH=min(P//2,20000)
            for c in range(max(1,P//(2*CH))):
                g=torch.Generator(device=dev).manual_seed(770000+c)
                z=torch.randn(CH,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
                h=pre_at(W,xx,l).double(); SM+=h.sum(0); SC+=h.t()@h; m+=xx.shape[0]
            mu=SM/m; K=SC/m-torch.outer(mu,mu)
            ev,V=torch.linalg.eigh(K.cpu())
            Ch=(V@torch.diag(ev.clamp(min=1e-12).sqrt())).to(dev)
            SS=torch.zeros(D,dtype=torch.float64,device=dev); ns=0
            for c in range(10):
                g=torch.Generator(device=dev).manual_seed(660000+c)
                u=torch.randn(20000,D,device=dev,generator=g); uu=torch.cat([u,-u],0)
                h=(mu.unsqueeze(0)+uu.double()@Ch.t().double()).float()
                SS+=from_pre(W,h,l).double().sum(0); ns+=uu.shape[0]
            bias=float(((SS/ns-TH)**2).mean())
            cost_p=P*(l/32.0)                      # прогонів на пілот
            left=6485.0-cost_p
            if left>0:
                M=left/(1.0-l/32.0)
                mse=2.5e-6*6485/max(M,1.0)+bias    # дисперсія масштабується як 1/M
                sc=mse*0.10
            else:
                sc=float('nan')
            log('  {:4d} {:6.1f}× {:9d} {:12.3e} {:11.0f}  {:12.3e}'
                .format(l,32.0/(32.0-l),P,bias,cost_p,sc))
        log('')
    log('  орієнтир: теперішній score = 2.5e-7;  бюджет-підлога = 6485 прогонів')
log('done.')
