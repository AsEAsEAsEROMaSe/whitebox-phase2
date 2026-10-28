"""КОНТРОЛЬ ФОРМУЛИ δ-РОЗКЛАДУ на МІЛКІЙ мережі (L=4), повний перебір 2^4 підмножин.

Твердження: Π_l (W_l G_l) = Π_l (A_l + W_l δ_l),  A_l=½W_l,  δ_l=diag(g_l−½)
   ⟹ a^L = Σ_{S⊆{0..L-1}}  x · Π_l (W_l δ_l якщо l∈S інакше A_l)

Якщо сума ПО ВСІХ підмножинах відтворює a^L побітово — формула вірна,
і вимір «перший порядок = 2.6e-5» можна брати на віру.
Якщо ні — вимір сміття, і добре що перевірив.

Плюс: розподіл ||внеску|| ПО ПОРЯДКАХ k=|S| — це прямо показує, де сидить відповідь.
"""
import torch, math, time, numpy as np, itertools
LOG='/mnt/win_d/llm/White-Box/delta_check.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; DT=torch.float64
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

for L in (4,8):
    W=[torch.tensor(dat['w_0_%d'%i].astype(np.float32),device=dev,dtype=DT) for i in range(L)]
    log(''); log('======== ГЛИБИНА L={} ========'.format(L))
    g0=torch.Generator(device=dev).manual_seed(4242)
    n=4096
    z=torch.randn(n//2,D,device=dev,generator=g0,dtype=DT)
    x=torch.cat([z,-z],0)

    # істина + збережені затвори
    a=x; G=[]
    for li in range(L):
        h=a@W[li]; gt=(h>0).to(DT); G.append(gt); a=h*gt
    truth=a

    # повний перебір підмножин, ПОСЕМПЛОВО (не в середньому!)
    tot=torch.zeros_like(truth)
    by_order=[torch.zeros_like(truth) for _ in range(L+1)]
    for mask in range(1<<L):
        v=x.clone()
        for li in range(L):
            v=v@W[li]
            if mask>>li & 1: v=v*(G[li]-0.5)      # W_l δ_l
            else:            v=v*0.5              # A_l = ½W_l
        tot=tot+v
        by_order[bin(mask).count('1')] += v
    err=float((tot-truth).norm()/truth.norm())
    log('  ПОСЕМПЛОВА тотожність: ||Σ_S − a^L||/||a^L|| = {:.3e}   {}'.format(
        err, 'OK' if err<1e-10 else 'ФОРМУЛА НЕВІРНА'))

    # тепер у СЕРЕДНЬОМУ — де сидить E[a^L] по порядках k=|S|
    Et=truth.mean(0); nb=float(Et.norm())
    log('  ||E[a^L]|| = {:.6f}'.format(nb))
    log('  внесок ПОРЯДКУ k у СЕРЕДНЄ:')
    log('    k    ||E[order k]||        частка від ||E[a^L]||')
    cum=torch.zeros(D,dtype=DT,device=dev)
    for k in range(L+1):
        e=by_order[k].mean(0); cum=cum+e
        log('   {:2d}    {:.6e}        {:8.5f}      (накопич. {:8.5f})'.format(
            k, float(e.norm()), float(e.norm())/nb, float(cum.norm())/nb))
log('done.')
