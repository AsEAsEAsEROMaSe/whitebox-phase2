"""КРИТИЧНІ МОМЕНТИ НА ПОВЕРХНЯХ ЗАТВОРІВ (ідея юзера С47).

ГІПОТЕЗА: уся дія на поверхнях |h_i^l| = 0 (їх 8192). Семпли, що проходять БЛИЗЬКО до них,
мали б нести більше складності => витрачати семпли ПРИЦІЛЬНО біля поверхонь (стратифікація /
importance sampling), а не рівномірно.

КОНТРАРГУМЕНТ (перевіряємо): затворів 8192; у кожного семпла ЗАВЖДИ купа близьких затворів.
Концентрація міри може зробити всі семпли однаково критичними => прицілюватись нікуди.

МІРЯЄМО, для кожного семпла:
  n_crit(tau) = #{затворів з |h|/sigma < tau}, tau = 0.05 / 0.10 / 0.25   (по ВСІХ 32 шарах)
  dmin        = min по всіх 8192 затворах від |h|/sigma
ПОТІМ:
  (1) відносна флуктуація n_crit (std/mean) — чи взагалі семпли РІЗНЯТЬСЯ за критичністю
  (2) кореляція критичності з ПОМИЛКОЮ семпла ||p - E||^2
  (3) стратифікація: 10 децилів за критичністю, частка сумарної дисперсії в кожному
ЧИТАННЯ: якщо дисперсія рівномірна по децилях => прицілюватись НІКУДИ (концентрація міри).
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/critical_surface.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAUS=(0.05,0.10,0.25)

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    NP=200_000; CH=20_000
    NC=[]; DM=[]; P=[]
    for c in range(NP//CH):
        gg=torch.Generator(device=dev).manual_seed(7700+c)
        z=torch.randn(CH,D,device=dev,generator=gg)
        x=torch.cat([z,-z],0)
        a=x
        nc=torch.zeros(CH,len(TAUS),device=dev)
        dmin=torch.full((CH,),1e9,device=dev)
        for li in range(L):
            h=a@W[li]
            s=h.std()                                     # sigma шару
            r=(h[:CH].abs()/s)                            # беремо гілку z (пара дзеркальна)
            for ti,tau in enumerate(TAUS):
                nc[:,ti]+=(r<tau).float().sum(1)
            dmin=torch.minimum(dmin,r.min(1).values)
            a=torch.relu(h)
        NC.append(nc); DM.append(dmin)
        P.append(((a[:CH]+a[CH:])*0.5).double())
    NC=torch.cat(NC); DM=torch.cat(DM); P=torch.cat(P); N=P.shape[0]
    E=P.mean(0); err=((P-E)**2).sum(1)                    # ||p-E||^2 на семпл
    log('  всього затворів = {} ; ||E||={:.3f} ; сер. помилка семпла = {:.4e}'
        .format(32*D,float(E.norm()),float(err.mean())))
    for ti,tau in enumerate(TAUS):
        v=NC[:,ti]
        log('  tau={:.2f}: n_crit сер={:8.1f}  std={:6.1f}  ВІДН.ФЛУКТ={:.4f}  корел(n_crit,помилка)={:+.4f}'
            .format(tau,float(v.mean()),float(v.std()),float(v.std()/v.mean()),
                    float(torch.corrcoef(torch.stack([v.double(),err]))[0,1])))
    log('  dmin: сер={:.4e} std={:.4e}  корел(dmin,помилка)={:+.4f}'
        .format(float(DM.mean()),float(DM.std()),
                float(torch.corrcoef(torch.stack([DM.double(),err]))[0,1])))
    # стратифікація по n_crit(tau=0.10)
    key=NC[:,1]
    idx=torch.argsort(key)
    log('  ДЕЦИЛІ за критичністю (tau=0.10): частка сумарної дисперсії в кожному (рівномірно=0.100)')
    parts=[]
    tot=float(err.sum())
    for b in range(10):
        sel=idx[b*N//10:(b+1)*N//10]
        parts.append(float(err[sel].sum())/tot)
    log('     '+'  '.join('{:.3f}'.format(p) for p in parts))
    log('     макс/мін по децилях = {:.2f}×'.format(max(parts)/min(parts)))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
