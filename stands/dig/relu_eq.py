"""«Вивести з активацій і ReLU, розв'язати рівняння» (юзер 15.10). Середнє виходу нейрона j = E[max(p_j,0)] залежить ЛИШЕ від
1-D закону p_j. Рівняння, які ми можемо мати точно: моменти p_j до порядку K (через лінійність p = hW вони точні, якщо точні моменти h).
Питання: чи визначають ці рівняння E[ReLU] з точністю, кращою за нашу похибку? Розв'язуємо задачу моментів як LP:
min/max Σ w(x)·max(x,0) при Σ w x^k = M_k (k ≤ K), w ≥ 0, сітка x. Моменти ІСТИННІ (банк), шар 15, дамп 0, 256 нейронів.
Ширина інтервалу = скільки відповіді рівняння НЕ визначають. Порівняння: похибка v34 |m3-gt| того ж нейрона.
Передбачення до запуску: ширина при K=4 у ×100+ більша за похибку v34 ⟹ рівняння моментів НЕ замикаються, відповідь дає
припущення про форму (гладкість, майже-гаус). Шанс, що ширина при K=4 < похибки v34: 2%."""
import numpy as np
from scipy.optimize import linprog
i=0; L=15
kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
a=kp['pre_mean'][L].astype(float); E2=kp['pre_m2'][L].astype(float); E3=kp['pre_m3'][L].astype(float); E4=kp['pre_m4'][L].astype(float)
gt=kp['gt_mean'][L].astype(float); m3=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'][L].astype(float)
v=E2-a*a; s=np.sqrt(v); t=a/s
c3=(E3-3*a*E2+2*a**3)/s**3; c4=(E4-4*a*E3+6*a*a*E2-3*a**4)/s**4    # стандартизовані центральні 3-й і 4-й
x=np.linspace(-14,14,2801); rng=np.random.default_rng(0); J=rng.choice(1024,256,replace=False)
rows=[]
for j in J:
    f=np.maximum(x+t[j],0)*s[j]
    Ms={2:[1,0,1],3:[1,0,1,c3[j]],4:[1,0,1,c3[j],c4[j]]}
    r=[]
    for K,M in Ms.items():
        A=np.stack([x**k for k in range(K+1)]); lo=linprog(f,A_eq=A,b_eq=M,bounds=(0,None),method='highs'); hi=linprog(-f,A_eq=A,b_eq=M,bounds=(0,None),method='highs')
        r.append((hi.fun*-1-lo.fun) if lo.status==0 and hi.status==0 else np.nan)
    rows.append([t[j],abs(m3[j]-gt[j]),abs(gt[j])]+r)
R=np.array(rows); ok=np.isfinite(R).all(1); R=R[ok]
print('нейронів %d; |t| медіана %.2f'%(len(R),np.median(np.abs(R[:,0]))))
for k,K in enumerate([2,3,4]):
    w=R[:,3+k]; print('K=%d: ширина/|gt| медіана %.2e  · ширина / похибка v34: медіана %.0f  мін %.1f  частка < 1: %.3f'%(K,np.median(w/R[:,2]),np.median(w/R[:,1]),np.min(w/R[:,1]),np.mean(w<R[:,1])))
for lo_,hi_ in [(0,1),(1,2),(2,3),(3,99)]:
    sel=(np.abs(R[:,0])>=lo_)&(np.abs(R[:,0])<hi_)
    if sel.sum(): print('  |t| у [%g,%g): n=%d  ширина K=4 / похибка v34 медіана %.0f'%(lo_,hi_,sel.sum(),np.median(R[sel,5]/R[sel,1])))
