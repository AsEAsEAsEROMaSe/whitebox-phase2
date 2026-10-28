"""ТОЧНА частка 2-го порядку — без підвибірки ознак (виправлення orders.py).

ЧОМУ orders.py не дав відповіді: 1200 випадкових квадратичних напрямків із простору
розмірності d(d+1)/2 = 32896 — це підвибірка 3.6%. Виміряні 2.7% однаково читаються
як «квадратики 3%» і як «квадратики 75%, ізотропної». Стенд був НЕДОСТАТНЬО ПОТУЖНИЙ.

ВИПРАВЛЕННЯ: повний квадратичний проєкт не треба перебирати ознаками.
Для парної частини y кожної координати:
    C = E[ y · (x xᵀ − I) ]                     (d×d, симетрична)
Найкраща квадратична поправка ∝ C, і пояснена дисперсія має ЗАМКНЕНУ ФОРМУ:
    Var_expl = ‖C‖²_F / 2
бо для q(x)=xᵀAx−tr(A):  Var(q)=2‖A‖²,  Cov(y,q)=⟨A,C⟩  ⟹  max = ‖C‖²/2.

ЗМІЩЕННЯ: ‖C‖² на скінченній вибірці роздувається шумом (65536 елементів!).
Тому рахуємо C на ДВОХ НЕЗАЛЕЖНИХ половинах і беремо ⟨C_A, C_B⟩ — незміщено.
Контроль: та сама формула на ПЕРЕМІШАНОМУ y ⟹ має дати ~0.

ЯКОРЕМ СЛУЖИТЬ ЗАПИС: ZCA робить вибіркову коваріацію рівно I ⟹ прибирає квадратику
ТОЧНО ⟹ виміряні 22% виграшу ZCA мають збігтися з часткою, яку порахуємо тут.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/orders2.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def even(W,x):
    """парна частина відповіді = те, що лишається після антитетики"""
    a=x
    for w in W: a=torch.relu(a@w)
    b=-x
    for w in W: b=torch.relu(b@w)
    return (a+b)*0.5

def collect(W,seed0,nch,CH=2000):
    """повертає T[i,a,b] = E[y_i x_a x_b] та середнє y — накопичення блоками"""
    T=torch.zeros(D,D*D,dtype=torch.float64,device=dev)
    SY=torch.zeros(D,dtype=torch.float64,device=dev)
    SY2=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        y=even(W,z).double()
        P=(z.unsqueeze(2)*z.unsqueeze(1)).reshape(CH,D*D).double()   # x_a x_b
        T+=y.t()@P; SY+=y.sum(0); SY2+=(y*y).sum(0); n+=CH
        del P
    T/=n; SY/=n; SY2/=n
    eye=torch.eye(D,dtype=torch.float64,device=dev).reshape(-1)
    C=T-SY.unsqueeze(1)*eye.unsqueeze(0)          # C = E[y(xxᵀ−I)]
    return C, SY, SY2-SY*SY, n

for net in (0,1):
    W=load(net); log('='*74); log('МЕРЕЖА %d'%net)
    CA,mA,vA,nA=collect(W,20000,15)
    CB,mB,vB,nB=collect(W,50000,15)
    log('  семплів на половину: {:d} (×2 прогони на антитетику)'.format(nA))
    cross=(CA*CB).sum(1)                                   # ⟨C_A,C_B⟩ по координатах — НЕЗМІЩЕНО
    expl=(cross*0.5).clamp(min=0)                          # ‖C‖²/2
    var=(vA+vB)*0.5
    r2=(expl/var.clamp(min=1e-30)).clamp(0,1)
    # контроль: перемішані мітки координат ⟹ C_A і C_B неспоріднені
    perm=torch.randperm(D,generator=torch.Generator().manual_seed(7),device='cpu').to(dev)
    ctl=((CA*CB[perm]).sum(1)*0.5/var.clamp(min=1e-30)).clamp(min=-1)
    log('  R²_quad ТОЧНИЙ: медіана {:.4f}   сер {:.4f}   контроль(перемішано) {:.5f}'
        .format(float(r2.median()),float(r2.mean()),float(ctl.median())))
    tot=float(var.sum()); res=float((var*(1-r2)).sum())
    log('  зважено дисперсією: парна Var {:.4e} → після зняття 2-го порядку {:.4e}  ⟹ {:.2f}×'
        .format(tot,res,tot/max(res,1e-30)))
    log('  ⟹ ЧАСТКА 2-го ПОРЯДКУ = {:.1f}%   (звірити з виміряними 22% виграшу ZCA)'
        .format(100*(1-res/tot)))
    log('  ⟹ ЧАСТКА 4-го і вище = {:.1f}%'.format(100*res/tot))
log('done.')
