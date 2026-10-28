"""СІМ'Я СТАЙНА з проміжних активацій — рівняння Пуассона, стеля = НУЛЬ дисперсії.

Для БУДЬ-ЯКОГО векторного поля u:   E[ div u(x) - x·u(x) ] = 0   точно, без знання мережі.
Беремо u(x) = M·a^k(x) — поле з самої мережі (не поліном від входу!).
   x·u          = x^T M a^k                     два матвеки
   div u = tr(M J_k)  ~  xi^T M (J_k xi)        зонд Хатчінсона; J_k xi = ОДИН forward-mode прохід
=> один додатковий прохід дає СКІЛЬКИ ЗАВГОДНО контрольних змінних (спільний xi).

Ідеальний розв'язок L phi = a^L - E[a^L] дав би НУЛЬ дисперсії. Питання: скільки дає
скінченна параметрична сім'я, і чи вона поза вичерпаною ермітовою оболонкою.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/stein.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def fwd_jvp(W,x,v,upto=L):
    """a^k і J_k·v одним проходом (для ReLU похідна = маска затвора)"""
    a=x; t=v
    for li in range(upto):
        h=a@W[li]; th=t@W[li]; g=(h>0).to(h.dtype)
        a=h*g; t=th*g
    return a,t

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float64),device=dev,dtype=torch.float64)
       for i in range(L)]
    log(''); log('######## NET {} (fp64) ########'.format(net))
    gs=torch.Generator(device=dev).manual_seed(800+net)
    # набір полів: випадкові M, одинична, та побудовані з ваг останнього шару
    Ms=[torch.randn(D,D,device=dev,generator=gs,dtype=torch.float64)/math.sqrt(D) for _ in range(6)]
    Ms.append(torch.eye(D,device=dev,dtype=torch.float64))
    Ms.append(W[L-1].clone()); Ms.append(W[L-1].t().clone())
    Ms.append(W[0].clone());   Ms.append(W[0].t().clone())
    KM=len(Ms)
    MS=torch.stack(Ms)                                  # (KM,D,D)

    BL=24; CH=20_000
    SG=[];SA=[];SAG=[];SGG=[]
    for b in range(BL):
        gg=torch.Generator(device=dev).manual_seed(21000+net*100+b)
        z=torch.randn(CH//2,D,device=dev,generator=gg,dtype=torch.float64)
        x=torch.cat([z,-z],0)
        xi=torch.randn(x.shape[0],D,device=dev,generator=gg,dtype=torch.float64)
        a,t=fwd_jvp(W,x,xi)                             # ОДИН прохід: a^L і J·xi
        # g_m = xi^T M t  -  x^T M a          (на семпл)
        Mt=torch.einsum('kij,nj->kni',MS,t)             # (KM,n,D)
        Ma=torch.einsum('kij,nj->kni',MS,a)
        gm=(xi[None]*Mt).sum(-1)-(x[None]*Ma).sum(-1)   # (KM,n)
        SG.append(gm.mean(1).cpu()); SGG.append((gm@gm.t()/gm.shape[1]).cpu())
        SA.append(a.mean(0).cpu())
        SAG.append((gm@a/a.shape[0]).cpu())             # (KM,D) крос-момент
        if b==0: sa=a.std(0).cpu(); sg=gm.std(1).cpu(); Va0=a.var(0).cpu()
    G=torch.stack(SG); A=torch.stack(SA); AG=torch.stack(SAG); GG=torch.stack(SGG)
    gm_=G.mean(0); gse=G.std(0)/math.sqrt(BL)
    log('1. перевірка E[g]=0 (має бути |z|~1):')
    for m in range(KM):
        z_=abs(float(gm_[m]))/max(float(gse[m]),1e-300)
        if m<3 or m>=KM-4:
            log('   M[{:2d}] {:10s}: E[g]={:+.3e}  se={:.3e}  |z|={:.2f}'.format(
                m,'випадк.' if m<6 else ('одинич.' if m==6 else 'з ваг'),
                float(gm_[m]),float(gse[m]),z_))
    log('   |z| сер по всіх {}: {:.2f}'.format(KM,float((gm_/gse.clamp(min=1e-300)).abs().mean())))

    log('2. КОРЕЛЯЦІЯ g з a^L та R²:')
    Ea=A.mean(0); Cov=AG.mean(0)-torch.outer(gm_,Ea)     # (KM,D)
    alive=Va0>1e-12*float(Va0.max())
    R=Cov/torch.outer(sg.clamp(min=1e-300),sa.clamp(min=1e-300))
    for m in range(KM):
        if m<3 or m>=KM-4:
            log('   M[{:2d}] {:10s}: сер|corr| = {:.4f}  макс {:.4f}'.format(
                m,'випадк.' if m<6 else ('одинич.' if m==6 else 'з ваг'),
                float(R[m][alive].abs().mean()),float(R[m][alive].abs().max())))
    # багатовимірний R^2 по всіх KM змінних
    S=GG.mean(0)-torch.outer(gm_,gm_) + 1e-12*torch.eye(KM,dtype=torch.float64)   # ПОСЕМПЛОВА
    beta=torch.linalg.solve(S,Cov)                       # (KM,D)
    expl=(beta*Cov).sum(0)
    r2=float((expl[alive]/Va0[alive]).clamp(0,1).mean())
    log('   БАГАТОВИМІРНИЙ R² по {} полях: {:.5f}'.format(KM,r2))
log('done.')
