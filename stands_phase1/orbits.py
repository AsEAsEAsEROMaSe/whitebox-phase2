"""СУСІДИ В ОРБІТІ (ідея юзера С47: «чи можна проводити схожі семпли поряд?»).

ПЕРЕВЕРТАЄМО ПИТАННЯ. Схожі семпли ⟹ схожі похибки ⟹ кореляція ДОДАТНА ⟹ гірше за
незалежні. Але АНТИТЕТИКА це і є «пов'язаний семпл», просто з правильним знаком: T(x)=−x
дає кореляцію −1 на всій непарній частині, тому вона зникає побітово.
⟹ Питання: чи є ІНШІ дешеві ортогональні T з ВІД'ЄМНОЮ кореляцією?

МЕТРИКА. Оцінка з пари ½(y(x)+y(Tx)) має Var = (1+ρ)/2·Var(y).
Дві НЕЗАЛЕЖНІ вибірки дали б Var/2. ⟹ виграш = 1/(1+ρ). Перемога лише при ρ<0.
Рахуємо ρ, зважену дисперсією по 256 координатах.

ВАЖЛИВО: y тут = вже ПАРНА частина (антитетика застосована). Тобто питаємо, чи можна
виграти ПОВЕРХ антитетики, а не замість неї.

ЧОМУ ЦЕ ОБХОДИТЬ СТІНУ [[finding-fourth-order-has-no-cheap-handle]]: кореляція сусідів у
орбіті не має стосунку до ермітових порядків — це інше представлення. 4-й порядок у d³
структурах їй не заважає.

КОНТРОЛІ: T=I (ρ має бути рівно 1), T=малий поворот (ρ→1 — буквальна відповідь юзера),
T=незалежний семпл (ρ→0).
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/orbits.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def even(W,x):
    a=x
    for w in W: a=torch.relu(a@w)
    b=-x
    for w in W: b=torch.relu(b@w)
    return (a+b)*0.5

def rot_planes(seed,ang):
    """блоково-діагональний поворот на кут ang у 128 випадкових ортогональних площинах"""
    g=torch.Generator(device=dev).manual_seed(seed)
    Q,_=torch.linalg.qr(torch.randn(D,D,device=dev,generator=g))
    c,s=float(np.cos(ang)),float(np.sin(ang))
    B=torch.zeros(D,D,device=dev)
    for k in range(0,D,2):
        B[k,k]=c; B[k,k+1]=-s; B[k+1,k]=s; B[k+1,k+1]=c
    return Q@B@Q.t()

def reflect(seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    v=torch.randn(D,device=dev,generator=g); v/=v.norm()
    return torch.eye(D,device=dev)-2.0*torch.outer(v,v)

def perm_T(seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    p=torch.randperm(D,generator=torch.Generator().manual_seed(seed)).to(dev)
    return torch.eye(D,device=dev)[p]

def halfflip(seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    s=torch.where(torch.rand(D,device=dev,generator=g)<0.5,-1.0,1.0)
    return torch.diag(s)

def randrot(seed):
    g=torch.Generator(device=dev).manual_seed(seed)
    Q,_=torch.linalg.qr(torch.randn(D,D,device=dev,generator=g))
    return Q

TESTS=[
 ('T = I  (контроль, має бути +1.000)',      lambda: torch.eye(D,device=dev)),
 ('поворот 1°  (буквально «схожий семпл»)',  lambda: rot_planes(1,np.pi/180)),
 ('поворот 10°',                             lambda: rot_planes(1,np.pi/18)),
 ('поворот 45°',                             lambda: rot_planes(1,np.pi/4)),
 ('поворот 90°  (орбіта порядку 4)',         lambda: rot_planes(1,np.pi/2)),
 ('поворот 135°',                            lambda: rot_planes(1,3*np.pi/4)),
 ('дзеркало у випадковій гіперплощині',      lambda: reflect(7)),
 ('перестановка координат',                  lambda: perm_T(9)),
 ('зміна знаку випадкової половини',         lambda: halfflip(11)),
 ('випадковий поворот',                      lambda: randrot(13)),
]

NCH=25; CH=2000
for net in (0,1):
    W=load(net); log('='*82); log('МЕРЕЖА %d   ({:d} семплів, ×2 прогони на антитетику)'.format(NCH*CH)%net)
    # еталонна статистика без перетворення
    log('  {:44s} {:>9s} {:>10s} {:>9s}'.format('перетворення T','ρ','виграш','вердикт'))
    for name,mk in TESTS:
        Tm=mk()
        sx=torch.zeros(D,dtype=torch.float64,device=dev); sy=torch.zeros(D,dtype=torch.float64,device=dev)
        sxx=torch.zeros(D,dtype=torch.float64,device=dev); syy=torch.zeros(D,dtype=torch.float64,device=dev)
        sxy=torch.zeros(D,dtype=torch.float64,device=dev); n=0
        for c in range(NCH):
            g=torch.Generator(device=dev).manual_seed(500000+c)
            z=torch.randn(CH,D,device=dev,generator=g)
            a=even(W,z).double(); b=even(W,z@Tm).double()
            sx+=a.sum(0); sy+=b.sum(0); sxx+=(a*a).sum(0); syy+=(b*b).sum(0); sxy+=(a*b).sum(0); n+=CH
        vx=sxx/n-(sx/n)**2; vy=syy/n-(sy/n)**2
        cov=sxy/n-(sx/n)*(sy/n)
        rho=float((cov.sum())/((vx.sum()*vy.sum())**0.5))     # зважена дисперсією
        gain=1.0/(1.0+rho) if rho>-0.999 else float('inf')
        verd='✔ ВИГРАШ' if rho<-0.02 else ('—' if rho<0.02 else '✗ гірше')
        log('  {:44s} {:+9.4f} {:9.3f}× {:>9s}'.format(name,rho,gain,verd))
log('done.')
