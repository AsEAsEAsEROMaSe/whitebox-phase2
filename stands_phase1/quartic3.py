"""ВИРІШАЛЬНИЙ ВИМІР 4-го ПОРЯДКУ: точно, без підвибірки, по ДВОІНДЕКСНІЙ структурі.

ЧОМУ попередні не годяться:
  spectrum.py — тотожність вірна, але дисперсія оцінки ~d^k (контроль: непарні 0.0005→917)
  quartic2.py — 3876 ознак із 1.8e8 = підвибірка 2e-5; фізично не бачить розмазаний сигнал

ТУТ: точна замкнена форма для ВСІЄЇ двоіндексної частини 4-го порядку (d² ≈ 33000 ознак).
Ермітів базис ортогональний ⟹ пояснена дисперсія = Σ_m E[y·φ_m]² / ‖φ_m‖², без підгонки.
  He₄(xᵢ)              d штук,        ‖·‖² = 24
  He₃(xᵢ)He₁(xⱼ) i≠j   d(d−1),        ‖·‖² = 6
  He₂(xᵢ)He₂(xⱼ) i<j   d(d−1)/2,      ‖·‖² = 4
ЗМІЩЕННЯ: E[yφ]² на скінченній вибірці роздувається шумом на 33000 елементах ⟹
рахуємо на ДВОХ незалежних половинах і беремо добуток (як в orders2.py).
ЦЕНТРУЄМО y (урок quartic.py: шум ∝ E[y²], а не Var(y); для активацій це ×20).
КОНТРОЛЬ: перемішані мітки координат ⟹ має дати ~0.

ЩО ЛИШАЄТЬСЯ ПОЗА ВИМІРОМ: тричі- й чотириіндексні квартики (d³, d⁴) — їх не порахувати.
Тому результат = НИЖНЯ оцінка 4-го порядку, але саме та частина, до якої міг би дотягнутись
дешевий метод. Якщо й тут ~0 — дешевого зняття 4-го порядку не існує.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/quartic3.log'; T0=time.time()
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

def moments(W,seed0,nch,mu,CH=2000):
    """коефіцієнти y при двоіндексних квартиках; y ЦЕНТРОВАНИЙ"""
    A4=torch.zeros(D,D,dtype=torch.float64,device=dev)
    A31=torch.zeros(D,D*D,dtype=torch.float64,device=dev)
    A22=torch.zeros(D,D*D,dtype=torch.float64,device=dev)
    P2=torch.zeros(D,D,dtype=torch.float64,device=dev)
    P11=torch.zeros(D,D*D,dtype=torch.float64,device=dev)
    V2=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g).double()
        y=even(W,z.float()).double()-mu
        h1=z; h2=z*z-1.0; h3=z*(z*z-3.0); h4=z*z*(z*z-6.0)+3.0
        A4+=y.t()@h4
        A31+=y.t()@(h3.unsqueeze(2)*h1.unsqueeze(1)).reshape(CH,-1)
        A22+=y.t()@(h2.unsqueeze(2)*h2.unsqueeze(1)).reshape(CH,-1)
        # ПОЗИТИВНИЙ КОНТРОЛЬ: 2-й порядок тим самим кодом — має вийти 51-57%
        P2+=y.t()@h2
        P11+=y.t()@(h1.unsqueeze(2)*h1.unsqueeze(1)).reshape(CH,-1)
        V2+=(y*y).sum(0); n+=CH
    return A4/n, (A31/n).reshape(D,D,D), (A22/n).reshape(D,D,D), V2/n, P2/n, (P11/n).reshape(D,D,D)

def mean_only(W,seed0,nch,CH=2000):
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        S+=even(W,z).double().sum(0); n+=CH
    return S/n

NCH=25   # 50000 семплів на половину
for net in (0,1):
    W=load(net); log('='*78); log('МЕРЕЖА %d'%net)
    mu=mean_only(W,300000,NCH)
    a4A,a31A,a22A,vA,p2A,p11A=moments(W,300000,NCH,mu)
    a4B,a31B,a22B,vB,p2B,p11B=moments(W,400000,NCH,mu)
    var=float(((vA+vB)*0.5).sum())
    off=~torch.eye(D,dtype=torch.bool,device=dev)                 # i≠j
    iu=torch.triu(torch.ones(D,D,dtype=torch.bool,device=dev),1)  # i<j
    e4 =float((a4A*a4B).sum())/24.0
    e31=float((a31A*a31B)[:,off].sum())/6.0
    e22=float((a22A*a22B)[:,iu].sum())/4.0
    perm=torch.randperm(D,generator=torch.Generator().manual_seed(11)).to(dev)
    c4 =float((a4A*a4B[perm]).sum())/24.0
    c31=float((a31A*a31B[perm])[:,off].sum())/6.0
    c22=float((a22A*a22B[perm])[:,iu].sum())/4.0
    q2=float((p2A*p2B).sum())/2.0 + float((p11A*p11B)[:,iu].sum())/1.0
    log('  парна Var = {:.4e}   (50000 семплів на половину)'.format(var))
    log('  ✔ ПОЗИТИВНИЙ КОНТРОЛЬ (той самий код, 2-й порядок): {:.4f}  — очікуємо 0.51-0.57'
        .format(q2/var))
    log('  {:>22s} {:>10s} {:>14s} {:>16s}'.format('сім\'я','ознак','частка Var','контроль(перемішано)'))
    log('  {:>22s} {:10d} {:14.5f} {:16.5f}'.format('He4(xi)',D,e4/var,c4/var))
    log('  {:>22s} {:10d} {:14.5f} {:16.5f}'.format('He3(xi)He1(xj)',D*(D-1),e31/var,c31/var))
    log('  {:>22s} {:10d} {:14.5f} {:16.5f}'.format('He2(xi)He2(xj)',D*(D-1)//2,e22/var,c22/var))
    tot=(e4+e31+e22)/var
    log('  ⟹ УСЯ ДВОІНДЕКСНА ЧАСТИНА 4-го ПОРЯДКУ = {:.4f}  ({:.2f}% дисперсії)'.format(tot,100*tot))
    log('     для порівняння: 2-й порядок = 51-57%;  порядок ≥4 разом = 43-49%')
    if tot>1e-4:
        log('     ⟹ виграш від зняття ЦІЄЇ частини поверх v2: {:.2f}×'
            .format((1-0.55)/max(1-0.55-tot,1e-9)))
log('done.')
