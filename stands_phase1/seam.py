"""НАСКІЛЬКИ ГРУБИЙ ШОВ? (ідея юзера про «зсунути блоки поряд»)

Блоки вже зсунуті: ReLU-мережа НЕПЕРЕРВНА, шматки склеєні без розривів.
Ламається не значення, а ПОХІДНА. І вся відповідь сидить саме в швах:
лапласіан усередині комірок = 0, а Стайн промахує на 2438σ ([[finding-poisson-reduces-to-surface]]).

ВЕРСІЯ ІДЕЇ, ЩО МАЄ ЗУБИ: наскільки ГРУБИЙ шов?
Коли x перетинає поверхню нейрона (l,j), якобіан J=∂a^L/∂x міняється на РАНГ-1:
   J = A_l · diag(g_l) · B_l,      A_l = ∂h^l/∂x  (префікс),  B_l = ∂a^L/∂a^l  (суфікс)
   переворот затвора j  ->  ΔJ = ± A_l[:,j] ⊗ B_l[j,:]
   відносний стрибок  =  ||A_l[:,j]||·||B_l[j,:]|| / ||J||

ЧИТАННЯ:
  стрибок ~1/√d ≈ 6%  -> функція «майже гладка», теорія збурень за швом дала б ~0.4%
                          (замало для нас, але це вже не «неможливо»)
  стрибок ~1          -> грубість суттєва, напрямок мертвий
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/seam.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev).double() for i in range(L)]
    log(''); log('######## НЕТ {} ########'.format(net))
    log('   шар   відн.стрибок ||ΔJ||/||J||: медіана   сер     макс     1/√d={:.4f}'.format(1/math.sqrt(D)))
    g=torch.Generator(device=dev).manual_seed(3131+net)
    NS=8
    acc={l:[] for l in range(L)}
    for s in range(NS):
        x=torch.randn(D,dtype=DT,device=dev,generator=g)
        # прямий хід: зберігаємо затвори
        a=x; G=[]
        for li in range(L):
            h=a@W[li]; gt=(h>0).to(DT); G.append(gt); a=h*gt
        # префікси A_l = ∂h^l/∂x  (D×D), суфікси B_l = ∂a^L/∂a^l
        A=[]; M=torch.eye(D,dtype=DT,device=dev)
        for li in range(L):
            M=M@W[li]; A.append(M.clone()); M=M*G[li][None,:]
        J=M                                     # повний якобіан ∂a^L/∂x
        nJ=float(J.norm())
        B=[None]*L; Bc=torch.eye(D,dtype=DT,device=dev)
        for li in range(L-1,-1,-1):
            B[li]=(G[li][:,None]*Bc)            # ∂a^L/∂h^l = diag(g) · Bc
            Bc=W[li]@B[li] if li>0 else Bc
        for li in range(L):
            na=A[li].norm(dim=0)                # ||A_l[:,j]||
            nb=B[li].norm(dim=1)                # ||B_l[j,:]||
            acc[li].append((na*nb/max(nJ,1e-300)).cpu())
    for li in (0,1,2,4,8,12,16,20,24,28,30,31):
        v=torch.cat(acc[li])
        log('   {:2d}      {:.4e}   {:.4e}   {:.4e}'.format(
            li, float(v.median()), float(v.mean()), float(v.max())))
    allv=torch.cat([torch.cat(acc[l]) for l in range(L)])
    log('   ---- по всіх 8192 нейронах: медіана {:.4e}  сер {:.4e}  макс {:.4e}'.format(
        float(allv.median()),float(allv.mean()),float(allv.max())))
    log('   сума квадратів усіх стрибків / ||J||² = {:.4f}'.format(float((allv**2).sum())))
log('done.')
