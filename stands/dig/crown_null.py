import numpy as np
B=2.199e12; cs=16*2*1024*1024
for i in range(8):
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i); m=kp['pre_mean'][15].astype(float)
    Cov=kp['pre_M11'][15].astype(float)-np.outer(m,m); lam,U=np.linalg.eigh(Cov)
    z=np.load('/home/trim/wb2data/pred_%d.npz'%i); e=(z['pred'][15]-z['gt'][15]).astype(float); ek2=(U.T@e)**2; tot=ek2.sum()
    s=[]
    for th in [1e-6,1e-5,1e-4,1e-3]:
        msk=lam<th; s.append('λ<%.0e: %d напр., частка e² %.3f'%(th,msk.sum(),ek2[msk].sum()/tot))
    # малий МК: N семплів, оракульні ваги Вінера
    r=[]
    for N in [16,64,256]:
        noise=np.clip(lam,0,None)/N; w=ek2*noise/(ek2+noise); r.append('N=%d (%.2f%%B) x%.3f'%(N,100*N*cs/B,tot/w.sum()))
    print('дамп %d: λ найменші %s | %s | %s'%(i,' '.join('%.1e'%x for x in lam[:3]),' · '.join(s),' · '.join(r)))
