"""РОУТЕРИ vs ЗАМКИ (ідея юзера С47): чи прибиті-закриті нейрони останнього шару
вимагають «особливого патерну», і чи саме на них сидить наша похибка.

ВАЖЛИВО: мережа НЕ навчена — «знань» нема. Але геометрія перевіряється.

Три питання:
  (1) НАПРЯМОК. Для мовчуна беремо ті рідкісні x, на яких він таки спрацював.
      Якщо ключ існує — середнє цих x має ДОВЖИНУ, помітно більшу за шум sqrt(d/n).
      Контроль: те саме на ВИПАДКОВІЙ підмножині тієї ж потужності.
  (2) ПЕРЕДБАЧУВАНІСТЬ. Ключ, знайдений на першій половині семплів, перевіряється
      на ДРУГІЙ: чи справді проєкція на нього піднімає частоту спрацювання.
  (3) ЦІНА. Шар 31 = сама відповідь. Чи корелює похибка v2 по координаті з рідкістю p_i?
      Якщо так — ми знаємо, ДЕ саме в 256 числах живе наш MSE.
"""
import torch, time, numpy as np
LOG = '/mnt/win_d/llm/White-Box/keys.log'; T0 = time.time()
def log(m):
    s = '[{:6.1f}s] {}'.format(time.time()-T0, m); print(s, flush=True); open(LOG,'a').write(s+'\n')
open(LOG,'w').close()

dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]

def fwd_h(W,x):
    """повертає ПЕРЕДактивацію останнього шару (h^31) і саму відповідь relu(h^31)"""
    a=x
    for li in range(L-1): a=torch.relu(a@W[li])
    h=a@W[L-1]
    return h, torch.relu(h)

for net in (0,1):
    W=load(net)
    log('='*78); log('МЕРЕЖА %d'%net)

    # ---- прохід 1: частоти спрацювання p_i та істина ----
    P=torch.zeros(D,dtype=torch.float64,device=dev); S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(60):
        g=torch.Generator(device=dev).manual_seed(9000+c)
        z=torch.randn(50_000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        h,a=fwd_h(W,xx)
        P+=(h>0).double().sum(0); S+=a.double().sum(0); n+=xx.shape[0]
    p=(P/n); TH=S/n
    off=(p<0.02).nonzero().flatten(); on=(p>0.98).nonzero().flatten()
    live=((p>=0.02)&(p<=0.98)).nonzero().flatten()
    log('  n={:d}  замкнених(p<.02)={:d}  відкритих(p>.98)={:d}  живих={:d}'
        .format(n,len(off),len(on),len(live)))
    if len(off)==0: log('  замкнених нема — пропуск'); continue
    po=p[off]
    log('  частота спрацювання мовчунів: медіана {:.2e}  min {:.2e}  max {:.2e}'
        .format(float(po.median()),float(po.min()),float(po.max())))
    log('  їхній внесок у відповідь: сума E[a_i] по мовчунах {:.3e} проти всіх {:.3e}  ({:.2f}%)'
        .format(float(TH[off].sum()),float(TH.sum()),100*float(TH[off].sum()/TH.sum())))

    # ---- (1)+(2) ключ: збираємо x умовно на спрацюванні, для 12 найтиповіших мовчунів ----
    idx=off[po.argsort(descending=True)[:12]]          # найчастіші серед мовчунів (є статистика)
    KS=torch.zeros(len(idx),D,dtype=torch.float64,device=dev); KN=torch.zeros(len(idx),device=dev)
    for c in range(60):
        g=torch.Generator(device=dev).manual_seed(9000+c)          # ТІ САМІ семпли
        z=torch.randn(50_000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        h,_=fwd_h(W,xx); hs=h[:,idx]>0
        KS+=(hs.double().t()@xx.double()); KN+=hs.double().sum(0)
    key=KS/KN.clamp(min=1).unsqueeze(1)                            # середнє x | спрацював
    kn=key.norm(dim=1)
    noise=(D/KN.clamp(min=1)).sqrt()                               # очікувана довжина для ВИПАДКОВОЇ підмножини
    log('  ключі (12 найактивніших мовчунів):')
    log('    {:>8s} {:>9s} {:>9s} {:>9s} {:>7s}'.format('p_i','спрацб','|ключ|','шум','×'))
    for j in range(len(idx)):
        log('    {:8.2e} {:9.0f} {:9.4f} {:9.4f} {:7.2f}'
            .format(float(p[idx[j]]),float(KN[j]),float(kn[j]),float(noise[j]),float(kn[j]/noise[j])))

    # ---- (2) перевірка ключа на СВІЖИХ семплах ----
    u=key/kn.clamp(min=1e-12).unsqueeze(1)
    hit_hi=torch.zeros(len(idx),device=dev); hit_lo=torch.zeros(len(idx),device=dev); nb=0
    for c in range(20):
        g=torch.Generator(device=dev).manual_seed(77000+c)          # НОВІ семпли
        z=torch.randn(50_000,D,device=dev,generator=g); xx=torch.cat([z,-z],0)
        h,_=fwd_h(W,xx); fire=(h[:,idx]>0).double()
        pr=xx.double()@u.t()                                        # проєкція на ключ
        top=(pr> pr.std(0)*2.0); bot=(pr< -pr.std(0)*2.0)
        hit_hi+=(fire*top.double()).sum(0)/top.double().sum(0).clamp(min=1)
        hit_lo+=(fire*bot.double()).sum(0)/bot.double().sum(0).clamp(min=1)
        nb+=1
    log('  перевірка ключа на СВІЖИХ семплах (частота спрацювання):')
    log('    {:>10s} {:>12s} {:>12s} {:>9s}'.format('базова','вздовж ключа','проти ключа','підйом'))
    for j in range(len(idx)):
        b=float(p[idx[j]]); hi=float(hit_hi[j]/nb)
        log('    {:10.2e} {:12.2e} {:12.2e} {:9.1f}×'.format(b,hi,float(hit_lo[j]/nb),hi/max(b,1e-12)))

    # ---- (3) де сидить похибка v2 по координатах ----
    def zca(x):
        C=(x.t()@x)/x.shape[0]
        e,V=torch.linalg.eigh(C.double().cpu())
        return (V@torch.diag(e.clamp(min=1e-14).rsqrt())@V.t()).to(dev).float()
    ERR=torch.zeros(D,dtype=torch.float64,device=dev); R=24
    for r in range(R):
        g=torch.Generator(device=dev).manual_seed(31000+97*r)
        z=torch.randn(3200,D,device=dev,generator=g); X=torch.cat([z,-z],0)
        Xw=X@zca(X); _,y=fwd_h(W,Xw)
        est=((y[:3200]+y[3200:])*0.5).double().mean(0)
        ERR+=(est-TH)**2
    ERR/=R
    q=p.argsort()                                    # від найтихіших до найгучніших
    log('  похибка v2 по децилях частоти спрацювання p_i (тихі → гучні), частка сумарної MSE:')
    dec=[float(ERR[q[k*D//10:(k+1)*D//10]].sum()/ERR.sum()) for k in range(10)]
    log('    '+' '.join('{:.3f}'.format(v) for v in dec))
    log('  для довідки, частка СИГНАЛУ E[a] по тих самих децилях:')
    dsg=[float(TH[q[k*D//10:(k+1)*D//10]].sum()/TH.sum()) for k in range(10)]
    log('    '+' '.join('{:.3f}'.format(v) for v in dsg))
    log('  ⟹ відносна похибка мовчунів vs гучних: {:.2f}× / {:.2f}×'
        .format(dec[0]/max(dsg[0],1e-12), dec[9]/max(dsg[9],1e-12)))
log('done.')
