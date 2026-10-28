"""СИНТЕТИЧНИЙ СТАРТ ІЗ ГЛИБИНИ (ідея юзера С47): народжувати семпл одразу на шарі ℓ.

НАВІЩО: score = σ²·f/B, вартість входить ЛІНІЙНО. Старт із шару ℓ економить ℓ/32 прогону:
  ℓ=8  ⟹ 1.33×      ℓ=16 ⟹ 2.0×      ℓ=24 ⟹ 4.0×
Це той самий порядок, якого бракує до 1e-8.

ЦІНА: істинного розподілу на шарі ℓ ми не знаємо — це й є задача. Наближаємо ГАУСОМ по
ПЕРЕДактивації h^ℓ (природно: h = сума 256 доданків), далі a=relu(h) і шари ℓ..31.
З'являється ЗСУВ, який НЕ лікується кількістю семплів.

НАПРУГА, через яку відповідь неочевидна:
  замикати РАНО  — розподіл майже гаусів (добре), але попереду 30 шарів розгойдування
  замикати ПІЗНО — шарів мало, зате розподіл уже сильно негаусів
⟹ може бути мінімум десь усередині. МІРЯЄМО зсув як функцію ℓ.

ОРІЄНТИР: наш робочий MSE ≈ 2.5e-6 (v2 на підлозі бюджету). Зсув СИСТЕМАТИЧНИЙ ⟹ щоб
ідея була жива, треба bias² ≪ 2.5e-6. Сусідній вимір: замикання на КОЖНОМУ шарі давало
1.12-1.16% відносної похибки (26× заграбо) — але то 32 замикання, тут ОДНЕ.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/synth_layer.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]

def pre_at(W,x,l):
    """передактивація на шарі l: h^l = a^{l-1} @ W[l]"""
    a=x
    for i in range(l): a=torch.relu(a@W[i])
    return a@W[l]
def from_pre(W,h,l):
    """далі від h^l: a=relu(h), потім шари l+1..31"""
    a=torch.relu(h)
    for i in range(l+1,L): a=torch.relu(a@W[i])
    return a

for net in (0,1):
    W=load(net); log('='*80); log('МЕРЕЖА %d'%net)
    # ---- істина ----
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(40):
        g=torch.Generator(device=dev).manual_seed(990000+c)
        z=torch.randn(60000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        S+=from_pre(W,pre_at(W,xx,0),0).double().sum(0); n+=xx.shape[0]
    TH=S/n
    log('  істина на {:d} семплах; ‖E[a]‖ = {:.4f}'.format(n,float(TH.norm())))
    log('')
    log('  {:>4s} {:>10s} {:>13s} {:>13s} {:>11s} {:>9s}'
        .format('шар','економія','зсув MSE','відн.похибка','проти 2.5e-6','вердикт'))
    for l in (1,2,4,8,12,16,20,24,28):
        # ---- пілот: m,K передактивації h^l ----
        SM=torch.zeros(D,dtype=torch.float64,device=dev)
        SC=torch.zeros(D,D,dtype=torch.float64,device=dev); m=0
        for c in range(20):
            g=torch.Generator(device=dev).manual_seed(770000+c)
            z=torch.randn(20000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
            h=pre_at(W,xx,l).double()
            SM+=h.sum(0); SC+=h.t()@h; m+=xx.shape[0]
        mu=SM/m; K=SC/m-torch.outer(mu,mu)
        ev,V=torch.linalg.eigh(K.cpu())
        Ch=(V@torch.diag(ev.clamp(min=1e-12).sqrt())).to(dev)      # корінь з K
        # ---- синтетичні семпли з N(mu,K), далі шари l+1..31 ----
        SS=torch.zeros(D,dtype=torch.float64,device=dev); ns=0
        for c in range(20):
            g=torch.Generator(device=dev).manual_seed(660000+c)
            u=torch.randn(20000,D,device=dev,generator=g)
            uu=torch.cat([u,-u],0)                                  # антитетика в просторі h
            h=(mu.unsqueeze(0)+uu.double()@Ch.t().double()).float()
            SS+=from_pre(W,h,l).double().sum(0); ns+=uu.shape[0]
        est=SS/ns
        bias=float(((est-TH)**2).mean())
        rel=float((est-TH).norm()/TH.norm())
        save=32.0/(32.0-l)
        verd='✔ живе' if bias<2.5e-6 else ('~ межа' if bias<2.5e-5 else '✗ зсув')
        log('  {:4d} {:9.2f}× {:13.3e} {:12.4f}% {:11.1f}× {:>9s}'
            .format(l,save,bias,100*rel,bias/2.5e-6,verd))
    log('  ЧИТАННЯ: зсув систематичний — семплами не лікується. Ідея жива лише якщо')
    log('           зсув MSE ≪ 2.5e-6 при економії, вартій заходу.')
log('done.')
