"""НУЛЬ-РОЗПОДІЛ таблиці децилів keys.py.
Претензія: «похибка/сигнал максимальна на СЕРЕДНІХ частотах (4.0x), майже ідентично
на 2 мережах => ЗАКОН, не реалізація».
Питання: а який НУЛЬ? Візьмемо ОДИН нейрон БЕЗ жодної мережі: a = max(h,0), h~N(m,s^2).
p = P(h>0) керується m/s. Рахуємо коеф. варіації sd(a)/E(a) — це і є «похибка/сигнал»
(MC-похибка ~ sd/sqrt(N), сигнал ~ E). Якщо форма збігається з виміряною — таблиця
не несе інформації про мережу.
"""
import numpy as np
from scipy.stats import norm
SQ2PI=np.sqrt(2*np.pi)
def moments(z):
    """h=s*(z+u), u~N(0,1); z=m/s. E[a]/s, sd[a]/s, p"""
    Phi=norm.cdf(z); phi=norm.pdf(z)
    E1=z*Phi+phi
    E2=(1+z*z)*Phi+z*phi
    var=np.maximum(E2-E1*E1,0)
    return E1,np.sqrt(var),Phi
print('  p(частота)   CV=sd/E   (нормовано на дециль-10)')
zs=np.linspace(-3.5,3.5,400)
E1,sd,p=moments(zs)
cv=sd/np.maximum(E1,1e-300)
# беремо децилі за p так само, як keys.py: рівні кошики по p
targets=[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
rows=[]
for t in targets:
    i=np.argmin(np.abs(p-t)); rows.append((t,cv[i]))
base=rows[-1][1]
for t,c in rows: print('   {:.2f}        {:6.3f}     {:6.2f}x'.format(t,c,c/base))
print()
print('ВИМІРЯНО в keys.py (децилі 5..10, нормовано так само):')
print('   нет0: 4.0  3.6  1.7  0.92  0.86  0.77')
print('   нет1: 4.0  3.6  1.8  1.21  0.71  0.69')
print('НУЛЬ  (децилі 5..10 з таблиці вище, БЕЗ МЕРЕЖІ):')
sel=[rows[i][1]/base for i in range(4,10)]
print('        '+'  '.join('{:.2f}'.format(v) for v in sel))
print()
print('монотонність CV по p: {}'.format('СПАДАЄ ЗАВЖДИ' if np.all(np.diff([r[1] for r in rows])<0) else 'НЕ монотонна'))
