"""Прискорення ряду за порядками: m1 (середнє поле), m2 (K=2), m3 (v34, K=3) на виході -> екстраполяція до K=inf.
Передбачення до запуску: e3 майже ортогональна d3=m3-m2 (cos<0.2), Aitken по нейронах шкодить (знаменники шумні),
спільний скаляр a·d3 з LOO по мережах дає <=x1.05. Шанс x1.2+: 10-15%.
Контроль: d3 і d2 чужої мережі (перестановка дампів) мусять дати ~x1.00."""
import numpy as np, torch, math, sys
sys.argv=[sys.argv[0],'0','6']; exec(open('/home/trim/wb2lean/dig/k2ladder.py').read().split("i=int(sys.argv[1])")[0])
D=[]
for i in range(8):
    W=torch.tensor(np.load('/home/trim/wb2data/dumps_p2/mlp_%04d.npz'%i)['weights'],device=dev).double()
    gt=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)['gt_mean'][15].astype(float)
    m1=run(W,'diag')[15].cpu().numpy(); m2=run(W,'full')[15].cpu().numpy()
    m3=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred'][15].astype(float)
    D.append((m1,m2,m3,gt))
mse=lambda x,g: float(((x-g)**2).mean())
cos=lambda a,b: float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()))
base=np.mean([mse(m3,g) for m1,m2,m3,g in D])
print('база v34 %.4e | K2 %.4e | K1 %.4e'%(base,np.mean([mse(m2,g) for m1,m2,m3,g in D]),np.mean([mse(m1,g) for m1,m2,m3,g in D])))
for i,(m1,m2,m3,g) in enumerate(D):
    e3=m3-g; d3=m3-m2; d2=m2-m1
    print('дамп %d cos(e3,d3) %+.3f cos(e3,d2) %+.3f |d3|/|e3| %.1f |d2|/|d3| %.1f'%(i,cos(e3,d3),cos(e3,d2),np.linalg.norm(d3)/np.linalg.norm(e3),np.linalg.norm(d2)/np.linalg.norm(d3)))
def fitpred(tr,te,shift=0):
    # спільний МНК e3 ~ a d3 + b d2 на мережах tr, застосувати до te (shift: брати d чужої мережі = контроль)
    X=[];Y=[]
    for j in tr:
        m1,m2,m3,g=D[j]; k=(j+shift)%8; n1,n2,n3,_=D[k]
        X.append(np.stack([n3-n2,n2-n1],1)); Y.append(m3-g)
    X=np.concatenate(X);Y=np.concatenate(Y); c=np.linalg.lstsq(X,Y,rcond=None)[0]
    out=[]
    for j in te:
        m1,m2,m3,g=D[j]; k=(j+shift)%8; n1,n2,n3,_=D[k]
        out.append(mse(m3-np.stack([n3-n2,n2-n1],1)@c,g))
    return np.mean(out),c
for nm,sh in [('свої',0),('контроль чужі',3)]:
    r=[fitpred([j for j in range(8) if j!=t],[t],sh)[0] for t in range(8)]
    print('%s: LOO a·d3+b·d2  MSE %.4e  x%.4f'%(nm,np.mean(r),base/np.mean(r)))
    r=[]
    for t in range(8):
        tr=[j for j in range(8) if j!=t]; X=np.concatenate([D[(j+sh)%8][2]-D[(j+sh)%8][1] for j in tr]); Y=np.concatenate([D[j][2]-D[j][3] for j in tr]); a=(X*Y).sum()/(X*X).sum()
        m1,m2,m3,g=D[t]; r.append(mse(m3-a*(D[(t+sh)%8][2]-D[(t+sh)%8][1]),g))
    print('%s: LOO a·d3 сам  MSE %.4e  x%.4f'%(nm,np.mean(r),base/np.mean(r)))
# Aitken / геометрія по нейронах
r=[];rg=[]
for m1,m2,m3,g in D:
    d2=m2-m1; d3=m3-m2; den=d3-d2
    ait=np.where(np.abs(den)>1e-12, m3-d3*d3/den, m3)
    r.append(mse(ait,g))
    rho=np.clip(np.where(np.abs(d2)>1e-12,d3/d2,0),-0.5,0.5); rg.append(mse(m3+d3*rho/(1-rho),g))
print('Aitken по нейронах MSE %.4e x%.4f | геом. з rho_i (кліп 0.5) %.4e x%.4f'%(np.mean(r),base/np.mean(r),np.mean(rg),base/np.mean(rg)))
