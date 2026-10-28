"""ЧИ СПРАВЖНЯ ОБЕРНЕНІСТЬ СІРОЇ КРИВОЇ? (спостереження юзера на панелі MC, С47)

ЮЗЕР ПОМІТИВ: на графіку збіжності сіра крива (звичайний MC) відхиляється від теоретичної
прямої 1/√N В ІНШИЙ БІК, ніж антитетика й v2, і саме в тому самому проміжку.

МЕХАНІЗМ, ЩО МІГ БИ ЦЕ ДАТИ: f = f_парн + f_непарн.
  антитетика/v2 вбивають НЕПАРНУ частину ТОЧНО ⟹ їхня помилка = лише парна.
  звичайний MC несе ОБИДВІ ⟹ має ДОДАТКОВУ незалежну складову.
Тобто криві зроблені частково з РІЗНОГО матеріалу — відхилення не зобов'язані збігатися.

АЛЕ: криві НАКОПИЧУВАЛЬНІ (сусідні точки залежні, одна невдача тягнеться проміжком),
а усереднення було лише по 12 повторах (~20% власного шуму). Тобто горб міг бути шумом.

ЦЕЙ СКРИПТ РОЗРІЗНЯЄ. R повторів; для кожного рахуємо помилку MC та v2 на сітці N;
беремо ЛОГ-ВІДХИЛЕННЯ від середнього тренду і корелюємо їх МІЖ МЕТОДАМИ на кожному N.
  систематично ВІД'ЄМНА кореляція => оберненість СПРАВЖНЯ
  ~0 або додатна                   => горб був шумом
Теоретичний прогноз: MC і v2 ділять ті самі z ⟹ спільна ПАРНА частина ⟹ кореляція має бути
ДОДАТНОЮ. Якщо вийде від'ємна — це сюрприз, який треба пояснювати.
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/odd_even.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; R=60; NP=20_000
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(dat['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
def fwd(x):
    a=x
    for li in range(L): a=torch.relu(a@W[li])
    return a
def zca(x):
    C=(x.t()@x)/x.shape[0]
    e,V=torch.linalg.eigh(C.double().cpu())
    return (V@torch.diag(e.clamp(min=1e-14).rsqrt())@V.t()).to(dev).float()

S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
for c in range(40):
    g=torch.Generator(device=dev).manual_seed(6100+c)
    z=torch.randn(100_000,D,device=dev,generator=g)
    xx=torch.cat([z,-z],0); S+=fwd(xx).double().sum(0); n+=xx.shape[0]
TH=S/n
log('істина готова. {} повторів, до {} пар…'.format(R,NP))
kp=np.unique(np.logspace(1,math.log10(NP),24).astype(int))
E={nm:np.zeros((R,len(kp))) for nm in ('MC','anti','v2')}
for r in range(R):
    g=torch.Generator(device=dev).manual_seed(4400+7*r)
    z=torch.randn(NP,D,device=dev,generator=g); z2=torch.randn(NP,D,device=dev,generator=g)
    yz=fwd(z).double()
    plain=torch.cat([yz,fwd(z2).double()],0)
    anti=(yz+fwd(-z).double())*0.5
    Xw=torch.cat([z,-z],0); Xw=Xw@zca(Xw)
    yv=fwd(Xw).double(); v2=(yv[:NP]+yv[NP:])*0.5
    for nm,Y,kk in (('MC',plain,2*kp),('anti',anti,kp),('v2',v2,kp)):
        cs=torch.cumsum(Y,0)
        E[nm][r]=[float(((cs[k-1]/k)-TH).norm()) for k in kk]
    if (r+1)%15==0: log('  повтор {}/{}'.format(r+1,R))
log('')
log('  N пар | корел(відх. MC, відх. v2) | корел(MC, anti) | RMS: MC/v2')
res={nm:np.log(E[nm])-np.log(E[nm]).mean(0,keepdims=True) for nm in E}
cs1=[];cs2=[]
for j,k in enumerate(kp):
    c1=float(np.corrcoef(res['MC'][:,j],res['v2'][:,j])[0,1])
    c2=float(np.corrcoef(res['MC'][:,j],res['anti'][:,j])[0,1])
    cs1.append(c1); cs2.append(c2)
    if j%4==0 or j==len(kp)-1:
        log('  {:6d} |          {:+.3f}            |     {:+.3f}       |   {:.2f}'
            .format(k,c1,c2,float(np.sqrt((E['MC'][:,j]**2).mean()/(E['v2'][:,j]**2).mean()))))
se=1/(R-3)**0.5
log('')
log('  СЕРЕДНЄ по всіх N:  корел(MC,v2) = {:+.3f}   корел(MC,anti) = {:+.3f}   (похибка ±{:.3f})'
    .format(float(np.mean(cs1)),float(np.mean(cs2)),se/len(kp)**0.5))
log('  від\'ємних точок: {} з {} для (MC,v2)'.format(int(np.sum(np.array(cs1)<0)),len(cs1)))
log('ЧИТАННЯ: додатна кореляція => спільна ПАРНА частина, горб на панелі був ШУМОМ.')
log('done.')
