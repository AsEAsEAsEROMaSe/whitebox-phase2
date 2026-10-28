"""Похибка ОДНОГО кроку при ТОЧНИХ понейронних кумулянтах передактивації (з банку 1e8):
скільки точності дає кожен порядок у самому кроці E[relu(p)], без переносу стану.
Передбачення: Гаус ~1e-7..4e-6 (як K=2 на шарі 1: 1.5e-7), +γ3 ×20..×200, +γ4 ще ×3..×10, далі підлога шуму 1e8 (~7e-10)."""
import numpy as np, math
from math import erf
def He(k,t):
    a,b=np.ones_like(t),t
    if k==0: return a
    for n in range(1,k): a,b=b,t*b-n*a
    return b
Phi=np.vectorize(lambda x:0.5*(1+erf(x/math.sqrt(2))))
rows=[]
for i in range(8):
    z=np.load('/home/trim/wb2data/kp_p2/mlp_%05d.npz'%i)
    mu=z['pre_mean'];m2=z['pre_m2'];m3=z['pre_m3'];m4=z['pre_m4'];gt=z['gt_mean']
    v=m2-mu**2; s=np.sqrt(v); k3=m3-3*mu*m2+2*mu**3; k4=m4-4*mu*m3-3*m2**2+12*mu**2*m2-6*mu**4
    g3=k3/s**3; g4=k4/s**4; t=mu/s; ph=np.exp(-t*t/2)/math.sqrt(2*math.pi); P=Phi(t)
    c=lambda k: s*((-1)**k)*He(k-2,t)*ph
    E0=s*(ph+t*P)
    E3=E0+g3/6*c(3)
    E4=E3+g4/24*c(4)
    E6=E4+g3**2/72*c(6)
    r=[((e-gt)**2).mean(1) for e in (E0,E3,E4,E6)]
    rows.append(np.array(r))
R=np.array(rows)  # 8 x 4 x 16
print('шар | Гаус | +γ3 | +γ4 | +γ3² | сер|γ3| сер γ4 (дамп0)')
z=np.load('/home/trim/wb2data/kp_p2/mlp_00000.npz')
for l in [0,1,2,4,8,12,15]:
    a=R[:,:,l].mean(0); print('%2d  %.2e  %.2e  %.2e  %.2e'%(l,*a))
np.save('/home/trim/wb2data/stepcum.npy',R)
