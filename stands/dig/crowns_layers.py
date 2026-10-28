"""Крони на кожному шарі: R(k) похибки середнього ланцюга v34 по шарах l (свої координати нейронів шару), групи за t_l і випадкові.
Частка MSE у кронах = (R-1)/k. Питання юзера: чи структура росте з глибиною."""
import numpy as np
rng=np.random.default_rng(0); ks=[4,16,64,256]
rows={}
for i in range(8):
    z=np.load('/home/trim/wb2data/pred_%d.npz'%i); kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
    for l in [1,4,8,12,14,15]:
        e=(z['pred'][l]-z['gt'][l]).astype(float); mu=kp['pre_mean'][l]; t=mu/np.sqrt(kp['pre_m2'][l]-mu**2)
        def R(order,k):
            g=e[order][:1024//k*k].reshape(-1,k).sum(1); return (g**2).mean()/(k*(e**2).mean())
        o=np.argsort(t)
        rt=[(R(o,k)-1)/k for k in ks]; rr=[(np.mean([R(rng.permutation(1024),k) for _ in range(30)])-1)/k for k in ks]
        tot=(e.sum()**2)/(1024*(e**2).sum())  # частка глобального зсуву ~ (R1024-1)/1024
        rows.setdefault(l,[]).append(rt+rr+[tot])
print('частка MSE у кронах (R-1)/k, сер 8 мереж; групи за t | випадкові | глобальний зсув')
print('шар   '+' '.join('t:k=%-4d'%k for k in ks)+' | '+' '.join('в:k=%-4d'%k for k in ks)+' | зсув')
for l,v in rows.items():
    v=np.mean(v,0); print('%2d    '%l+' '.join('%8.3f'%x for x in v[:4])+' | '+' '.join('%8.3f'%x for x in v[4:8])+' | %.4f'%v[8])
