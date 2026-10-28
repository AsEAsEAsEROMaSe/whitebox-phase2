"""НУЛЬ-РОЗПОДІЛ U-подібності critical_surface.py.
Гіпотеза: sigma береться ГЛОБАЛЬНО по батчу (h.std()), мережа однорідна ⟹ n_crit це
переодягнена ЗВОРОТНА ГУЧНІСТЬ. Тоді U-форма ||p-E||^2 по децилях n_crit виводиться
без геометрії: зліва p велике, справа p~0 при E!=0, мінімум де ||p||~||E||.
ТЕСТ: децильна таблиця, побудована по ГУЧНОСТІ ||p||, має відтворити ту саму U.
"""
import torch, numpy as np, time
dev='cuda:4'; D=256; L=32; T0=time.time()
def log(m):
    s='[{:6.1f}s] {}'.format(time.time()-T0,m); print(s,flush=True)
    open('/mnt/win_d/llm/White-Box/crit_null.log','a').write(s+'\n')
open('/mnt/win_d/llm/White-Box/crit_null.log','w').close()
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def deciles(key,err):
    idx=torch.argsort(key); e=err[idx]; n=len(e)//10
    sh=[float(e[i*n:(i+1)*n].sum()/e.sum()) for i in range(10)]
    return sh
for net in (1,):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    NP=100_000; CH=10_000
    NC=[];P=[];XN=[]
    for c in range(NP//CH):
        gg=torch.Generator(device=dev).manual_seed(7700+c)
        z=torch.randn(CH,D,device=dev,generator=gg)
        x=torch.cat([z,-z],0); a=x
        nc=torch.zeros(CH,device=dev)
        for li in range(L):
            h=a@W[li]; s=h.std()
            nc+=((h[:CH].abs()/s)<0.10).float().sum(1)
            a=torch.relu(h)
        NC.append(nc); P.append(((a[:CH]+a[CH:])*0.5).double()); XN.append(z.norm(dim=1).double())
    NC=torch.cat(NC).double(); P=torch.cat(P); XN=torch.cat(XN)
    E=P.mean(0); err=((P-E)**2).sum(1); loud=P.norm(dim=1)
    cc=lambda a,b: float(torch.corrcoef(torch.stack([a,b]))[0,1])
    log('==== НЕТ {} ===='.format(net))
    log('  ||E||={:.3f}   сер||p||={:.3f}'.format(float(E.norm()),float(loud.mean())))
    log('  корел(n_crit, ГУЧНІСТЬ ||p||) = {:+.4f}   <-- якщо сильно від\'ємна, n_crit=зворотна гучність'.format(cc(NC,loud)))
    log('  корел(n_crit, ||x||)          = {:+.4f}'.format(cc(NC,XN)))
    log('  корел(n_crit, помилка)        = {:+.4f}   (в оригіналі -0.079/-0.102)'.format(cc(NC,err)))
    log('  ДЕЦИЛІ за n_crit (оригінал):')
    log('     '+'  '.join('{:.3f}'.format(v) for v in deciles(NC,err)))
    log('  ДЕЦИЛІ за ГУЧНІСТЮ ||p|| (нуль-гіпотеза):')
    dl=deciles(loud,err)
    log('     '+'  '.join('{:.3f}'.format(v) for v in dl))
    log('     макс/мін = {:.2f}x   (оригінал 2.47x/2.31x)'.format(max(dl)/min(dl)))
    # де мінімум? має бути там, де ||p||~||E||
    idx=torch.argsort(loud); n=len(loud)//10
    mid=[float(loud[idx[i*n:(i+1)*n]].mean()) for i in range(10)]
    jmin=int(np.argmin(dl))
    log('     дециль з МІНІМУМОМ = #{}, сер||p|| там = {:.3f}, а ||E|| = {:.3f}  {}'
        .format(jmin+1,mid[jmin],float(E.norm()),
                '<-- ЗБІГ' if abs(mid[jmin]-float(E.norm()))/float(E.norm())<0.15 else ''))
log('done.')
