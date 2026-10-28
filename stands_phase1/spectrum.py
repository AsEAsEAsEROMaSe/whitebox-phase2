"""СПЕКТР ПОРЯДКІВ — скільки дисперсії парного залишку сидить у кожному порядку Ерміта.

ТОТОЖНІСТЬ, що робить це можливим (репродукуюче ядро Ерміта):
    Σ_{|α|=k}  He_α(x)·He_α(x') / α!   =   (x·x')^k / k!
Ліворуч — сума по 1.8e8 мультиіндексах для k=4. Праворуч — скаляр.
⟹ частка дисперсії порядку k:
    V_k = E_A E_B [ y_A · y_B · (x_A·x_B)^k / k! ]
де A,B — ДВІ НЕЗАЛЕЖНІ половини (⟹ незміщено, без роздування шумом).

y = ПАРНА частина (те, що лишається після антитетики) ⟹ НЕПАРНІ k мають дати ~0.
Це ВБУДОВАНИЙ КОНТРОЛЬ: (x·x')^k при |x·x'|~16 і k=6 це ~1e7, тож іде сильне взаємознищення;
якщо непарні k вийдуть малими — точності вистачило, якщо ні — fp64 не витягнув.

НАВІЩО: [[finding-v2-is-exact-order-two]] показав 2-й порядок = 51-57%, вичерпаний v2.
Дорога до 1e-8 потребує 5.4× із 4-го порядку. Треба знати, чи він там Є.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/spectrum.log'; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32; KMAX=7
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def even(W,x):
    a=x
    for w in W: a=torch.relu(a@w)
    b=-x
    for w in W: b=torch.relu(b@w)
    return (a+b)*0.5

def half(W,seed0,nch,CH):
    XS=[];YS=[]
    for c in range(nch):
        g=torch.Generator(device=dev).manual_seed(seed0+c)
        z=torch.randn(CH,D,device=dev,generator=g)
        XS.append(z.double()); YS.append(even(W,z).double())
    return torch.cat(XS,0), torch.cat(YS,0)

import math as _m
fact=[float(_m.factorial(k)) for k in range(KMAX+1)]
CH=2000; NCH=5                                  # 10000 семплів на половину
for net in (0,1):
    W=load(net); log('='*76); log('МЕРЕЖА %d'%net)
    XA,YA=half(W,60000,NCH,CH); XB,YB=half(W,80000,NCH,CH)
    nA,nB=XA.shape[0],XB.shape[0]
    mA=YA.mean(0); mB=YB.mean(0)
    YAc=YA-mA; YBc=YB-mB                        # центруємо ⟹ порядок 0 прибрано
    varY=float(((YAc*YAc).sum(0)/nA).sum())
    log('  семплів на половину {:d}; сумарна парна Var = {:.4e}'.format(nA,varY))

    V=torch.zeros(KMAX+1,D,dtype=torch.float64,device=dev)
    for a0 in range(0,nA,CH):
        xa=XA[a0:a0+CH]; ya=YAc[a0:a0+CH]
        for b0 in range(0,nB,CH):
            G=xa@XB[b0:b0+CH].t()               # (x_A · x_B)
            yb=YBc[b0:b0+CH]
            Gk=torch.ones_like(G)
            for k in range(1,KMAX+1):
                Gk=Gk*G
                V[k]+=(ya*(Gk@yb)).sum(0)/fact[k]
    V/=(nA*nB)
    tot=V.sum(1)
    log('   k |  частка Var  |  накопичено парних |  примітка')
    acc=0.0; pos=float(sum(max(float(tot[k]),0.0) for k in range(2,KMAX+1,2)))
    for k in range(1,KMAX+1):
        f=float(tot[k])/varY
        if k%2==0:
            acc+=f
            log('  {:2d} | {:+12.5f} | {:18.5f} | ПАРНИЙ'.format(k,f,acc))
        else:
            log('  {:2d} | {:+12.5f} | {:>18s} | непарний — КОНТРОЛЬ (має бути ~0)'.format(k,f,'—'))
    rest2=1.0-float(tot[2])/varY
    rest4=1.0-(float(tot[2])+float(tot[4]))/varY
    rest6=1.0-(float(tot[2])+float(tot[4])+float(tot[6]))/varY
    log('  ЗАЛИШОК після зняття порядків:')
    log('    ≤2 (це і є v2): {:.4f}  ⟹ {:.2f}×'.format(rest2,1.0/max(rest2,1e-9)))
    log('    ≤4            : {:.4f}  ⟹ {:.2f}×  (додатково до v2: {:.2f}×)'
        .format(rest4,1.0/max(rest4,1e-9),rest2/max(rest4,1e-9)))
    log('    ≤6            : {:.4f}  ⟹ {:.2f}×  (додатково до v2: {:.2f}×)'
        .format(rest6,1.0/max(rest6,1e-9),rest2/max(rest6,1e-9)))
    log('  ПОТРІБНО додатково до v2 для 1e-8 (після flopscope×1.77 і поворотів×3.25): 5.4×')
log('done.')
