"""δ-РОЗКЛАД НАВКОЛО ПОЛОВИНИ (ідея юзера: «шар дає лише половину можливостей»).

ВИМІРЯНО (gatespec.py): частка відкритих затворів = 0.50 на КОЖНОМУ шарі.

  a^L = x · Π_l (W_l G_l),   G_l = diag(g_l)
  G_l = ½I + δ_l,            δ_l = diag(g_l − ½),  E[δ_l]=0
  нульовий член = 2^{-32}·x·ΠW_l  ->  E = 0 ТОЧНО (бо E[x]=0, ΠW = константа)

⟹ УСЯ відповідь = кореляція ЗАТВОРІВ зі ЗНАЧЕННЯМИ.
   Якби затвори були незалежними монетками — E[a^L] був би РІВНО нуль.

ТЕСТ: підмінити затвори шару l незалежними монетками (та сама p відкриття,
незалежні від x) і подивитись, наскільки поїде E[a^L].
  - підміна ВСІХ шарів -> має дати ~0  (перевірка самого розкладу)
  - шар, підміна якого майже не рухає відповідь -> його кореляцію можна не рахувати
    (це був би важіль із БЕЗКОШТОВНИМ якорем)
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/halfgate.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def run(W,n,seed,swap=None,pop=None,gseed=0):
    """swap=None -> звичайний форвард.
       swap=список шарів -> на цих шарах затвор замінюється НЕЗАЛЕЖНОЮ монеткою
       з імовірністю pop[l] (та сама частка відкритих, але 0 кореляції з x)."""
    g=torch.Generator(device=dev).manual_seed(seed)
    z=torch.randn(n//2,D,device=dev,generator=g)
    x=torch.cat([z,-z],0)
    gc=torch.Generator(device=dev).manual_seed(gseed)
    a=x; frac=[]
    for li in range(L):
        h=a@W[li]
        gt=(h>0).float()
        frac.append(float(gt.mean()))
        if swap is not None and li in swap:
            coin=(torch.rand(h.shape,device=dev,generator=gc)<pop[li]).float()
            a=h*coin
        else:
            a=h*gt
    return a.double().mean(0), frac

N=400_000
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    log(''); log('######## НЕТ {}  (N={}) ########'.format(net,N))
    base,frac=run(W,N,7700+net)
    nb=float(base.norm())
    log('  частка відкритих затворів: мін {:.3f}  сер {:.3f}  макс {:.3f}'.format(
        min(frac),sum(frac)/len(frac),max(frac)))
    log('  ||E[a^L]|| = {:.6f}   (еталон)'.format(nb))

    # 1) ПІДМІНА ВСІХ ШАРІВ -> перевірка самого розкладу (має дати ~0)
    allsw,_=run(W,N,7700+net,swap=set(range(L)),pop=frac,gseed=1)
    log('')
    log('  1) ВСІ затвори -> незалежні монетки:')
    log('     ||E|| = {:.3e}   це {:.4f} від еталона   (розклад передбачає ~0)'.format(
        float(allsw.norm()), float(allsw.norm())/nb))

    # 2) підміна ОДНОГО шару за раз
    log('')
    log('  2) підміна ОДНОГО шару (наскільки поїде відповідь):')
    log('     шар   ||E_підм|| / ||E||    відн.зсув ||ΔE||/||E||')
    for li in [0,1,2,4,8,12,16,20,24,28,30,31]:
        e,_=run(W,N,7700+net,swap={li},pop=frac,gseed=2)
        log('     {:2d}      {:7.4f}              {:7.4f}'.format(
            li, float(e.norm())/nb, float((e-base).norm())/nb))

    # 3) підміна ПЕРШИХ k шарів (чи можна ранні взагалі не рахувати)
    log('')
    log('  3) підміна ПЕРШИХ k шарів (чи ранні кореляції взагалі потрібні):')
    for k in (1,2,4,8,16,24):
        e,_=run(W,N,7700+net,swap=set(range(k)),pop=frac,gseed=3)
        log('     k={:2d}   ||E_підм||/||E|| = {:7.4f}   ||ΔE||/||E|| = {:7.4f}'.format(
            k, float(e.norm())/nb, float((e-base).norm())/nb))

    # 4) підміна ОСТАННІХ k шарів
    log('')
    log('  4) підміна ОСТАННІХ k шарів:')
    for k in (1,2,4,8,16,24):
        e,_=run(W,N,7700+net,swap=set(range(L-k,L)),pop=frac,gseed=4)
        log('     k={:2d}   ||E_підм||/||E|| = {:7.4f}   ||ΔE||/||E|| = {:7.4f}'.format(
            k, float(e.norm())/nb, float((e-base).norm())/nb))
log('done.')
