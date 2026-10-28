"""Яка ФОРМА закону p_j дає найточніше E[ReLU] при ТОЧНИХ моментах (банк), шари 12 і 15, 8 дампів, усі 1024 нейрони.
Форми: G = гаус (m,σ); E = Еджворт до κ4 (1-D: ψ(t) − γ3/6·tφ + γ4/24·(t²−1)φ, ψ=φ+tΦ); ME = максимум ентропії з 4 моментами;
порівняння з похибкою v34 і з підлогою шуму істини.
Передбачення до запуску: G гірше за v34 (нема κ3); E і ME близькі, обидва кращі за v34 у ×2..×10 (точні моменти = оракул кроку);
шанс, що ME б'є E більш ніж у ×2: 20%."""
import numpy as np
from scipy.stats import norm
x=np.linspace(-12,12,4801); dx=x[1]-x[0]
def maxent(c3,c4):
    # опуклий дуал: min_λ log Z(λ) − λ·M, демпфований Ньютон з бектрекінгом
    M=np.array([0.,1.,c3,c4+3.]); P=np.stack([x,x**2,x**3,x**4])
    def obj(l):
        e=l@P; mx=e.max(); return mx+np.log(np.exp(e-mx).sum())-l@M
    lam=np.array([0.,-0.5,0.,0.]); f=obj(lam)
    for _ in range(200):
        e=lam@P; e-=e.max(); w=np.exp(e); w/=w.sum(); mom=P@w; g=mom-M
        if np.abs(g).max()<1e-11: break
        H=(P*w)@P.T-np.outer(mom,mom); d=np.linalg.solve(H+1e-12*np.eye(4),g); st=1.0
        while st>1e-8:
            ln=lam-st*d; fn=obj(ln)
            if np.isfinite(fn) and fn<=f-1e-4*st*(g@d): break
            st*=0.5
        lam,f=ln,fn
    e=lam@P; e-=e.max(); w=np.exp(e); return w/w.sum()
res={L:{k:[] for k in ['G','E','ME','v34']} for L in [12,15]}
for i in range(8):
    kp=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i); pr=np.load('/home/trim/wb2data/pred_%d.npz'%i)['pred']
    for L in res:
        a=kp['pre_mean'][L].astype(float); E2=kp['pre_m2'][L].astype(float); E3=kp['pre_m3'][L].astype(float); E4=kp['pre_m4'][L].astype(float); gt=kp['gt_mean'][L].astype(float)
        v=E2-a*a; s=np.sqrt(v); t=a/s; c3=(E3-3*a*E2+2*a**3)/s**3; c4=(E4-4*a*E3+6*a*a*E2-3*a**4)/s**4-3
        ph=norm.pdf(t); Ph=norm.cdf(t); psi=ph+t*Ph
        G=s*psi; E=s*(psi-c3/6*t*ph+c4/24*(t*t-1)*ph)
        ME=np.array([s[j]*(maxent(c3[j],c4[j])@np.maximum(x+t[j],0)) for j in range(1024)])
        for k,p in [('G',G),('E',E),('ME',ME),('v34',pr[L].astype(float))]: res[L][k].append(np.mean((p-gt)**2))
    print('дамп',i,flush=True)
for L in res:
    b=np.mean(res[L]['v34']); print('шар %d: MSE v34 %.3e'%(L,b)+''.join(' | %s %.3e (x%.2f до v34)'%(k,np.mean(res[L][k]),b/np.mean(res[L][k])) for k in ['G','E','ME']))
