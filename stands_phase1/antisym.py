"""АНТИСИМЕТРИЧНА ЗВ'ЯЗАНІСТЬ (ідея юзера).

Квадратична щільність = гаусіана. Її група інваріантності — повороти, а їх алгебра Лі —
рівно АНТИСИМЕТРИЧНІ матриці A (A^T=-A), бо exp(eA) не міняє N(0,I) взагалі.
Диференціюючи E[f(x)] = E[f(e^{eA}x)] по e в нулі:

        E[ grad f(x)^T A x ] = 0        ТОЧНО, для БУДЬ-ЯКОЇ f, без знання про мережу

(перевірка Стайном: E[d_i f x_k] = E[d_i d_k f] -> згортка антисиметричної A з СИМЕТРИЧНИМ
гесіаном = тотожний нуль)

Форма: g_A = J(x)·(A x) — спершу v=Ax (матвек), потім J·v = похідна за напрямком
= ОДИН прохід у forward-режимі. Ті самі великі матмули, що й у v2.

Перевіряємо: (1) чи справді середнє нуль, (2) чи є кореляція з a^L (інакше CV марна).
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/antisym.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd_jvp(W,x,v):
    """одночасно a^L і J·v (forward-mode): для ReLU похідна = маска затвора"""
    a=x; t=v
    for li in range(L):
        h=a@W[li]; th=t@W[li]
        g=(h>0).to(h.dtype)
        a=h*g; t=th*g
    return a,t

def make_A(seed,scale='rand'):
    g=torch.Generator(device=dev).manual_seed(seed)
    M=torch.randn(D,D,device=dev,generator=g)
    A=(M-M.t())/math.sqrt(2.0*D)
    return A

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    log(''); log('######## NET {} ########'.format(net))
    K=8
    As=[make_A(500+net*50+j) for j in range(K)]
    # антисиметричні у НАПРЯМКАХ МЕРЕЖІ: A = u w^T - w u^T з колонок W0
    U=W[0]/W[0].norm(dim=0,keepdim=True)
    for j in range(K):
        u=U[:,2*j]; w=U[:,2*j+1]
        As.append(torch.outer(u,w)-torch.outer(w,u))
    NA=len(As)
    NP=50_000; CH=25_000
    SG=torch.zeros(NA,D,dtype=torch.float64,device=dev)      # sum g
    SGG=torch.zeros(NA,dtype=torch.float64,device=dev)       # sum ||g||^2 (масштаб)
    SA=torch.zeros(D,dtype=torch.float64,device=dev)
    SAA=torch.zeros(D,dtype=torch.float64,device=dev)
    CROSS=torch.zeros(NA,D,dtype=torch.float64,device=dev)   # sum a*g покомпонентно (ДІАГОНАЛЬ)
    FULL=torch.zeros(NA,D,D,dtype=torch.float64,device=dev)  # sum a_i g_j (ПОВНА матриця)
    SGSUM=torch.zeros(NA,D,dtype=torch.float64,device=dev)
    SGG2=torch.zeros(NA,D,dtype=torch.float64,device=dev)
    n=0
    for c in range(NP//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(31000+net*10+c))
        x=torch.cat([z,-z],0)
        a=None
        for j,A in enumerate(As):
            v=x@A.t()                        # v = A x  (рядки)
            aa,t=fwd_jvp(W,x,v)
            if a is None:
                a=aa; SA+=aa.double().sum(0); SAA+=(aa.double()**2).sum(0)
            SG[j]+=t.double().sum(0); SGG[j]+=float((t.double()**2).sum())
            ad=aa.double(); td=t.double()
            CROSS[j]+=(ad*td).sum(0)
            FULL[j]+=ad.t()@td
            SGSUM[j]+=td.sum(0); SGG2[j]+=(td*td).sum(0)
        n+=x.shape[0]
    Ea=SA/n; Va=(SAA/n-Ea*Ea).clamp(min=1e-30)
    log('перевірка НУЛЬОВОГО середнього (має бути ~шум 1/sqrt(N)):')
    for j in range(0,NA,4):
        g_mean=SG[j]/n; g_rms=math.sqrt(float(SGG[j])/n/D)
        log('  A[{:2d}] {:9s}: |сер|/rms = {:.4f}   (очік. шум {:.4f})'.format(
            j,'випадк.' if j<K else 'мережа',
            float(g_mean.abs().mean())/max(g_rms,1e-30), 1.0/math.sqrt(n)))
    log('')
    log('КОРЕЛЯЦІЯ g_A з a^L (покомпонентно, усереднено по 256):')
    alive=Va>1e-12*float(Va.max())
    best=0.0
    for j in range(NA):
        Eg=SG[j]/n
        cov=CROSS[j]/n-Ea*Eg
        Vg=(SGG[j]/n/D)                      # груба дисперсія g
        r=(cov/ (Va.sqrt()*math.sqrt(max(Vg,1e-30))))
        rm=float(r[alive].abs().mean()); best=max(best,rm)
        if j%4==0:
            log('  A[{:2d}] {:9s}: сер|corr| = {:.4f}'.format(j,'випадк.' if j<K else 'мережа',rm))
    log('  НАЙКРАЩА |corr| серед {} матриць: {:.4f}   (R^2 = {:.4f})'.format(NA,best,best**2))
    log('')
    log('ПОВНА матриця Cov(a_i, g_j) — те, чого теорема НЕ занулює:')
    sa=Va.sqrt()
    for j in range(0,NA,4):
        Eg=SGSUM[j]/n; Vg=(SGG2[j]/n-Eg*Eg).clamp(min=1e-30)
        Cm=FULL[j]/n-torch.outer(Ea,Eg)
        Rm=Cm/torch.outer(sa,Vg.sqrt())
        off=Rm-torch.diag(torch.diagonal(Rm))
        log('  A[{:2d}] {:9s}: діаг сер|r| {:.4f} | ПОЗАдіаг сер|r| {:.4f} | макс {:.4f}'.format(
            j,'випадк.' if j<K else 'мережа',
            float(torch.diagonal(Rm)[alive].abs().mean()),
            float(off[alive][:,alive].abs().mean()), float(off[alive][:,alive].abs().max())))
    # багатовимірний R^2: регресія a_i на ВСІ g (усі NA*D ознак завелико -> беремо по одній A)
    log('  R^2 регресії a на g (по одній матриці A, повний вектор g 256-вим):')
    for j in range(0,NA,4):
        Eg=SGSUM[j]/n; Vg=(SGG2[j]/n-Eg*Eg).clamp(min=1e-30)
        Cm=(FULL[j]/n-torch.outer(Ea,Eg))
        Gcov=torch.diag(Vg)            # наближення: g-компоненти слабо корельовані
        beta=Cm@torch.linalg.inv(Gcov.cpu()+1e-12*torch.eye(D,dtype=torch.float64)).to(dev)
        expl=(beta*Cm).sum(1)
        r2=float((expl[alive]/Va[alive]).mean())
        log('    A[{:2d}] {:9s}: R^2 = {:.5f}'.format(j,'випадк.' if j<K else 'мережа',r2))
log('done.')
