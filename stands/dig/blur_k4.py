"""«Розмита мапа» для κ4 (юзер 15.10): чи є ГЛАДКА загальна картинка, яка дає ексцес кожного нейрона γ4 з величин, які ланцюг уже знає
(t, γ3 = власна асиметрія, яку несе K=3)? Для суми небагатьох доданків γ4 і γ3² жорстко пов'язані (ранг стану ~3, фаза 1).
Фіт γ4 ~ ознаки на 7 дампах, перевірка на 8-му (LOO), істинні моменти банку, шари 8/12/15.
Передбачення до запуску: γ4 ~ рівень + b·γ3² пояснює R² 0.3..0.6 понад рівень; шанс > 0.8: 15%."""
import numpy as np
def load(i,L):
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
    a=kp['pre_mean'][L].astype(float); E2=kp['pre_m2'][L].astype(float); E3=kp['pre_m3'][L].astype(float); E4=kp['pre_m4'][L].astype(float)
    v=E2-a*a; s=np.sqrt(v); t=a/s; g3=(E3-3*a*E2+2*a**3)/s**3; g4=(E4-4*a*E3+6*a*a*E2-3*a**4)/s**4-3
    return t,g3,g4
feats={'рівень':lambda t,g3:[np.ones_like(t)],
       '+t':lambda t,g3:[np.ones_like(t),t,t*t],
       '+γ3²':lambda t,g3:[np.ones_like(t),g3*g3],
       '+γ3,γ3²':lambda t,g3:[np.ones_like(t),g3,g3*g3],
       'усе (t,γ3 до 2-го)':lambda t,g3:[np.ones_like(t),t,t*t,g3,g3*g3,g3*t]}
for L in [8,12,15]:
    D=[load(i,L) for i in range(8)]
    print('\nшар %d: γ4 сер %.3f sd по нейронах %.3f; corr(γ4,γ3²) %.3f corr(γ4,γ3) %.3f corr(γ4,t) %.3f'%(L,np.mean([d[2].mean() for d in D]),np.mean([d[2].std() for d in D]),
          np.mean([np.corrcoef(d[2],d[1]**2)[0,1] for d in D]),np.mean([np.corrcoef(d[2],d[1])[0,1] for d in D]),np.mean([np.corrcoef(d[2],d[0])[0,1] for d in D])))
    for nm,f in feats.items():
        r=[]
        for k in range(8):
            X=np.concatenate([np.stack(f(D[j][0],D[j][1]),1) for j in range(8) if j!=k]); Y=np.concatenate([D[j][2] for j in range(8) if j!=k])
            c=np.linalg.lstsq(X,Y,rcond=None)[0]; Xt=np.stack(f(D[k][0],D[k][1]),1); yt=D[k][2]
            r.append(1-np.sum((yt-Xt@c)**2)/np.sum((yt-yt.mean())**2))
        print('  %-20s LOO R² (понад середнє дампа) %.3f'%(nm,np.mean(r)))
