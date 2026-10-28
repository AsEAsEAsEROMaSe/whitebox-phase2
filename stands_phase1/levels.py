"""РІВНІ ПЕРЕХОДУ (ідея юзера, С47): затвори як числа заповнення, не як 8192 імен.

МОТИВ: поіменні затвори вимагають 8192 ймовірностей, наш якір для них 100× грубий
(тотожність вартості якоря з'їдає виграш). РІВЕНЬ = скільки затворів ВІДКРИТО на шарі —
одне число на шар, і його середнє відоме ТОЧНО: 128 (затвори відкриті рівно ½).
Якір БЕЗКОШТОВНИЙ і ТОЧНИЙ -> тотожність якоря НЕ спрацьовує.

ПЕРЕВІРЯЄМО:
 A) чи справді E[рівень]=128 ТОЧНО на кожному шарі (на 1-му це теорема h(-x)=-h(x);
    глибше — питання). Друкуємо відхилення в сигмах.
 B) R^2 передбачення a^L (ПАРНА частина = середнє антитетичної пари) по 32 рівнях,
    на ВІДКЛАДЕНІЙ половині. Рівень 1-го шару має бути МАРНИЙ (непарний ⟂ парний залишок) —
    це вбудований контроль правильності виміру.
 C) де сидить сигнал по глибині (покомпонентна кореляція).
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/levels.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    NP=400_000; CH=25_000                      # NP пар
    F=[]; Y=[]
    for c in range(NP//CH):
        gg=torch.Generator(device=dev).manual_seed(9100+c)
        z=torch.randn(CH,D,device=dev,generator=gg)
        x=torch.cat([z,-z],0)
        a=x; cnt=[]
        for li in range(L):
            h=a@W[li]; a=torch.relu(h)
            cnt.append((h>0).float().sum(1))    # рівень: скільки відкрито
        C=torch.stack(cnt,1)                    # (2CH, L)
        # ПАРНА частина: середнє по антитетичній парі (саме її й оцінює v2)
        F.append(((C[:CH]+C[CH:])*0.5).double())
        Y.append(((a[:CH]+a[CH:])*0.5).double())
    F=torch.cat(F); Y=torch.cat(Y); N=F.shape[0]
    # ---- A) чи E[рівень]=128 точно? ----
    # беремо СИРІ рівні (не парну частину): відновлюємо середнє з парної частини — воно те саме
    mu=F.mean(0); se=F.std(0)/N**0.5
    log('A) E[рівень]/256 по шарах (очікуємо рівно 0.5). Відхилення в сигмах:')
    s=' '.join('{}:{:+.1f}'.format(l,float((mu[l]-128.0)/se[l])) for l in range(0,L,1))
    log('   '+s)
    log('   максимальне |зсув|/256 = {:.3e}  (шар {})'.format(
        float(((mu-128.0)/256).abs().max()), int(((mu-128.0)).abs().argmax())))
    # ---- B) R^2 по рівнях ----
    h=N//2
    Fc=F-F[:h].mean(0); Yc=Y-Y[:h].mean(0)
    Fs=Fc/Fc[:h].std(0)
    A=Fs[:h]; B=Yc[:h]
    G=A.t()@A/h; rhs=A.t()@B/h
    for lam in (1e-6,1e-4,1e-2):
        beta=torch.linalg.solve(G+lam*torch.eye(L,dtype=G.dtype,device=dev),rhs)
        pred=Fs[h:]@beta; res=Yc[h:]-pred
        r2=1-float((res*res).sum()/(Yc[h:]**2).sum())
        log('B) lam={:.0e}  R^2(відкладена, всі 32 рівні) = {:+.4f}'.format(lam,r2))
    # без 1-го шару (контроль: не має нічого змінити)
    idx=list(range(1,L))
    A2=Fs[:h][:,idx]; G2=A2.t()@A2/h; r2b=A2.t()@B/h
    beta2=torch.linalg.solve(G2+1e-4*torch.eye(L-1,dtype=G2.dtype,device=dev),r2b)
    res2=Yc[h:]-Fs[h:][:,idx]@beta2
    log('   без шару 0:                          R^2 = {:+.4f}'.format(
        1-float((res2*res2).sum()/(Yc[h:]**2).sum())))
    # ---- C) поодинокі рівні ----
    log('C) R^2 ОДНОГО рівня (шар: R^2):')
    out=[]
    for l in range(L):
        a1=Fs[:h][:,l:l+1]; g1=a1.t()@a1/h; b1=torch.linalg.solve(g1+1e-4*torch.eye(1,dtype=g1.dtype,device=dev),a1.t()@B/h)
        r=Yc[h:]-Fs[h:][:,l:l+1]@b1
        out.append('{}:{:+.4f}'.format(l,1-float((r*r).sum()/(Yc[h:]**2).sum())))
    for i in range(0,L,8): log('   '+' '.join(out[i:i+8]))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
