"""Ідея юзера: знизу (гілки) шум, вище (крони) видно зміну суми. Похибка v34 e на 1024 виходах, групи розміру k:
R(k) = Var(сума e по групі) / (k * сер e^2). Незалежний шум: R=1 на всіх k. Структура в кронах: R>1 і росте з k.
Групи: випадкові; впорядковані за t15, за істинним значенням, за ієрархічною кластеризацією кореляцій входів (рядки W15)."""
import numpy as np
rng=np.random.default_rng(0)
def ratio(e,order,k):
    g=e[order][:len(e)//k*k].reshape(-1,k).sum(1); return (g**2).mean()/(k*(e**2).mean())
ks=[1,4,16,64,256,1024]
acc={}
for i in range(8):
    z=np.load('/home/trim/wb2data/pred_%d.npz'%i); e=(z['pred'][15]-z['gt'][15]).astype(float); gt=z['gt'][15]
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i); mu=kp['pre_mean'][15]; s=np.sqrt(kp['pre_m2'][15]-mu**2); t=mu/s
    W=np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'][15]  # x@w: колонки = вихідні нейрони
    Wc=W/np.linalg.norm(W,axis=0); Cw=Wc.T@Wc
    # ієрархія: спектральне впорядкування за провідним власним вектором кореляцій колонок
    ev=np.linalg.eigh(Cw)[1][:,-1]
    orders={'випадкові':None,'за t':np.argsort(t),'за значенням':np.argsort(gt),'за кореляцією ваг':np.argsort(ev)}
    for nm,o in orders.items():
        vals=[]
        for k in ks:
            if o is None: v=np.mean([ratio(e,rng.permutation(1024),k) for _ in range(50)])
            else: v=ratio(e,o,k)
            vals.append(v)
        acc.setdefault(nm,[]).append(vals)
print('R(k) = Var(суми похибки в групі)/(k·сер e²), 8 мереж, сер [мін..макс]; незалежний шум = 1')
print('групування       '+''.join('   k=%-5d'%k for k in ks))
for nm,v in acc.items():
    v=np.array(v); print('%-16s'%nm+''.join('  %.2f[%.2f..%.2f]'%(v[:,j].mean(),v[:,j].min(),v[:,j].max()) if j<5 else '  %.2f'%v[:,j].mean() for j in range(len(ks))))
