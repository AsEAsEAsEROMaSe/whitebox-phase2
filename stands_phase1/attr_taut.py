"""АУДИТ attr_zero.py: чи cos(v*, топ-вісь)=0.99 — тавтологія?
Гіпотеза: обидва = напрямок середнього µ, бо a^L>=0 і однорідна степеня 1
⟹ домінантна флуктуація = загальний масштаб.
Додаємо КОСИНУСИ ДО µ, яких в оригіналі не було, + позитивний контроль.
"""
import torch, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time()
def log(m):
    s='[{:5.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True)
    open('/mnt/win_d/llm/White-Box/attr_taut.log','a').write(s+'\n')
open('/mnt/win_d/llm/White-Box/attr_taut.log','w').close()
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
for net in (0,1,2):
    W=load(net); log('='*72); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g); X=X/X.norm(dim=1,keepdim=True)
    for it in range(6):
        Y=fwd(W,X); Xn=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-30)
        if it>=2:
            sp=float((Xn*X).sum(1).min())   # чи зійшлась ітерація
        X=Xn
    v=X[0]/X[0].norm()
    # чи всі 8 стартів дали ОДИН атрактор
    spread=float((X@v).min())
    C=torch.zeros(D,D,dtype=torch.float64,device=dev); S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(10):
        gg=torch.Generator(device=dev).manual_seed(1234+c)
        x=torch.randn(20000,D,device=dev,generator=gg)
        with torch.no_grad(): r=(0.5*(fwd(W,x)+fwd(W,-x))).double()
        S+=r.sum(0); C+=r.t()@r; n+=x.shape[0]
    mu=S/n; C=C/n-torch.outer(mu,mu)
    tr=float(C.trace()); vd=v.double()
    fv=float(vd@C@vd)/tr
    ev,V=torch.linalg.eigh(C); ev=ev.flip(0); V=V.flip(1)
    top=V[:,0]; mun=mu/mu.norm()
    log('  ітерація: min cos(старт_i, v*) = {:.4f}  (1.0 = всі 8 в один атрактор)'.format(spread))
    log('  дисперсія вздовж v*   : {:6.2f}%   (оригінал: 65.4/49.9)'.format(100*fv))
    log('  дисперсія вздовж топ-осі: {:6.2f}%'.format(100*float(ev[0])/tr))
    log('  cos(v*, топ-вісь)       = {:.4f}   <-- число з пам\'яті'.format(abs(float(vd@top))))
    log('  ---- КОСИНУСИ ДО СЕРЕДНЬОГО (яких в оригіналі НЕ БУЛО) ----')
    log('  cos(v*,  µ)             = {:.4f}'.format(abs(float(vd@mun))))
    log('  cos(топ-вісь, µ)        = {:.4f}'.format(abs(float(top@mun))))
    log('  дисперсія вздовж µ      = {:6.2f}%'.format(100*float(mun@C@mun)/tr))
    # позитивний контроль: випадковий напрямок
    gr=torch.Generator(device=dev).manual_seed(77+net)
    rnd=torch.randn(D,device=dev,generator=gr,dtype=torch.float64); rnd/=rnd.norm()
    log('  [контроль] cos(випадк., µ)={:.4f}  дисперсія вздовж нього={:.3f}%  (рівномірно 0.39%)'
        .format(abs(float(rnd@mun)),100*float(rnd@C@rnd)/tr))
log('done.')
