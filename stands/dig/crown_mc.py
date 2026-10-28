"""Ідея юзера: виміряти головні крони (грубі суми) і з них підігнати корені. Крони істини можна дешево виміряти
МК лише там, де дисперсія виходу мала. Стеля (оракул напрямків): у власному базисі Cov(out) по кожному напрямку k
наша похибка e_k^2 проти шуму МК lam_k/N; Вінер-суміш дає e_k^2*(lam_k/N)/(e_k^2+lam_k/N). Ціна N семплів = N*3.36e7 флопів."""
import numpy as np
B=2.199e12; cs=16*2*1024*1024
for i in range(4):
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i); m=kp['pre_mean'][15].astype(float)
    Cov=kp['pre_M11'][15].astype(float)-np.outer(m,m)
    lam,U=np.linalg.eigh(Cov); lam=np.clip(lam,0,None)
    z=np.load('/home/trim/wb2data/pred_%d.npz'%i); e=(z['pred'][15]-z['gt'][15]).astype(float)
    ek2=(U.T@e)**2; base=ek2.sum()/1024
    out=[]
    for frac in [0.05,0.2,0.77]:
        N=frac*B/cs; noise=lam/N; w=ek2*noise/(ek2+noise)
        out.append('%.0f%%B N=%d: MSE x%.3f'%(frac*100,N,base/(w.sum()/1024)))
    pr=lam.sum()**2/(lam**2).sum()
    print('дамп %d: слід Cov %.1f PR %.0f, λ мін %.1e мед %.1e | частка e² у 512 найменших λ %.2f | '%(i,lam.sum(),pr,lam[0],np.median(lam),ek2[:512].sum()/ek2.sum())+' · '.join(out))
