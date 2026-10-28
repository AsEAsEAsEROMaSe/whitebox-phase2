"""«Чи може атрактор бути нулем?» — розбір.
СТАЛИЙ зсув не міняє дисперсії (Var(a−c)=Var(a)) ⟹ як опорна точка виграш РІВНО 0.
Змістовне питання: чи ШУМІТЬ оцінювач уздовж v*? Якщо шум сидить в одній ВІДОМІЙ осі —
це принципово інша ситуація, ніж рівномірний розмаз по 256.
Міряємо на ПАРНІЙ частині r=½(y(x)+y(−x)) — саме її бачить v2 після антитетики."""
import torch, math, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time()
def log(m): print('[{:5.1f}s] {}'.format(time.time()-T0,m),flush=True)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for i in range(L): a=torch.relu(a@W[i])
    return a
for net in (0,1):
    W=load(net); log('='*70); log('МЕРЕЖА %d'%net)
    g=torch.Generator(device=dev).manual_seed(4242)
    X=torch.randn(8,D,device=dev,generator=g); X=X/X.norm(dim=1,keepdim=True)
    for _ in range(3):
        Y=fwd(W,X); X=Y/Y.norm(dim=1,keepdim=True).clamp(min=1e-30)
    v=X[0]/X[0].norm()
    # коваріація ПАРНОЇ частини
    C=torch.zeros(D,D,dtype=torch.float64,device=dev); S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(10):
        gg=torch.Generator(device=dev).manual_seed(1234+c)
        x=torch.randn(20000,D,device=dev,generator=gg)
        with torch.no_grad(): r=(0.5*(fwd(W,x)+fwd(W,-x))).double()
        S+=r.sum(0); C+=r.t()@r; n+=x.shape[0]
    mu=S/n; C=C/n-torch.outer(mu,mu)
    tr=float(C.trace())
    fv=float(v.double()@C@v.double())/tr
    ev,V=torch.linalg.eigh(C); ev=ev.flip(0); V=V.flip(1)
    log('  дисперсія парної частини вздовж v* (атрактор):     {:6.2f}%   (рівномірно 0.39%)'.format(100*fv))
    log('  дисперсія вздовж НАЙГІРШОЇ осі (топ-власний вектор): {:6.2f}%'.format(100*float(ev[0])/tr))
    log('  cos(v*, найгірша вісь шуму) = {:.4f}'.format(abs(float(v.double()@V[:,0]))))
    log('  топ-8 осей шуму тримають {:.1f}% дисперсії; топ-32 — {:.1f}%'
        .format(100*float(ev[:8].sum()/ev.sum()),100*float(ev[:32].sum()/ev.sum())))
    log('  ⟹ якби ми ІДЕАЛЬНО знали відповідь уздовж v*, MC-виграш був би {:.3f}×'
        .format(1.0/(1.0-fv)))
