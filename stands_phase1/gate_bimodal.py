"""ЗАТВОРИ — МОНЕТКИ ЧИ ПРИБИТІ? (випливло з діагностики viz, С47)

Записано: «затвори відкриті рівно ½». АЛЕ це може означати ДВІ РІЗНІ речі:
  (A) кожен нейрон перемикається як МОНЕТКА: p_i ≈ 0.5 для кожного i
  (B) нейрони майже ФІКСОВАНІ: половина p_i ≈ 1, половина p_i ≈ 0, а ½ — середнє ПО НЕЙРОНАХ
Діагностика останнього шару натякнула на (B): |середнє|/σ ≈ 3.3 по нейронах.

МІРЯЄМО для кожного шару: розподіл p_i = P(h_i > 0) по 256 нейронах.
  (A) => гістограма ГОСТРА біля 0.5
  (B) => гістограма БІМОДАЛЬНА (маса біля 0 і 1)
Плюс: скільки нейронів «прибиті» (p<0.02 або p>0.98) і ЯКА ЧАСТКА ВИХОДУ від них.
Плюс: |mean|/sd по шарах — глибина ефекту.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/gate_bimodal.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    N=200_000; CH=25_000
    ssum=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]   # сума затворів
    hs=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]     # сума h
    hs2=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]    # сума h^2
    n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(1234+c)
        x=torch.randn(CH,D,device=dev,generator=g)
        a=x
        for li in range(L):
            h=a@W[li]
            ssum[li]+=(h>0).double().sum(0); hs[li]+=h.double().sum(0); hs2[li]+=(h.double()**2).sum(0)
            a=torch.relu(h)
        n+=CH
    log('  шар | сер.p | част. p<0.02 або p>0.98 | медіана |mean|/sd | частка «монеток» 0.4<p<0.6')
    for li in (0,1,2,4,8,16,24,30,31):
        p=ssum[li]/n
        mu=hs[li]/n; sd=(hs2[li]/n-mu*mu).clamp(min=1e-30).sqrt()
        pinned=float(((p<0.02)|(p>0.98)).double().mean())
        coins=float(((p>0.4)&(p<0.6)).double().mean())
        log('   {:3d} | {:.4f} |        {:.3f}          |      {:6.2f}      |      {:.3f}'
            .format(li,float(p.mean()),pinned,float((mu.abs()/sd).median()),coins))
    # гістограма p для останнього шару
    p=(ssum[L-1]/n).cpu().numpy()
    hist,edges=np.histogram(p,bins=10,range=(0,1))
    log('  гістограма p_i (шар 31), 10 кошиків 0..1:')
    log('     '+'  '.join('{:3d}'.format(int(v)) for v in hist))
    p0=(ssum[0]/n).cpu().numpy()
    hist0,_=np.histogram(p0,bins=10,range=(0,1))
    log('  гістограма p_i (шар 0) для контролю (теорія: усі рівно 0.5):')
    log('     '+'  '.join('{:3d}'.format(int(v)) for v in hist0))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
