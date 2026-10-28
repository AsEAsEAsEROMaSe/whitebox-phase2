"""ЧИ ДОПОМОЖЕ БІЛЬШЕ СЕМПЛІВ? (питання юзера С47)

МЕТРИКА: score = MSE × max(0.10, C/B),  C = F + 1e11·R,  B = 2.72e11.
Один прогін мережі: 32 шари × 2·256² ≈ 4.19e6 флоп. ZCA додає ~2d² на семпл + d³ разово.
ПІДЛОГА множника: 0.10·B = 2.72e10 флоп ⟹ ≈ 6.3к прогонів. Ось звідки наші ~6к.

ТОЧНИЙ АРГУМЕНТ: вище підлоги MSE ∝ 1/N, а множник ∝ N ⟹ у добутку N СКОРОЧУЄТЬСЯ:
    score = (σ²/N)·(N·f/B) = σ²·f/B  — від N НЕ ЗАЛЕЖИТЬ.
Тобто точність не купується семплами; важать лише дисперсія ОДНОГО семпла і ціна ОДНОГО.

ЩІЛИНА, ЯКУ ПЕРЕВІРЯЄМО: скорочення точне лише якщо MSE падає РІВНО як 1/N. У v2
вибілювання оцінюється з ТИХ САМИХ семплів, отже при більшому N воно ТОЧНІШЕ —
дисперсія на семпл може сама залежати від N. (Слід у записах: той самий метод дав
−29% при N=20k і +4.1% при N=200k.)

МІРЯЄМО: MSE(N) для v2 і для простої антитетики, множимо на реальний множник вартості,
дивимось, чи справді score плаский вище підлоги.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/score_vs_n.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; B=2.72e11
FWD=2.0*D*D*L                      # флоп на один прогін мережі
ZCA_PS=2.0*D*D                     # флоп на семпл на вибілювання (ковар + перетворення)
ZCA_FIX=D**3                       # разово (розклад)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def load(net): return [torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
def fwd(W,x):
    a=x
    for li in range(L): a=torch.relu(a@W[li])
    return a
def zca(x):
    C=(x.t()@x)/x.shape[0]
    e,V=torch.linalg.eigh(C.double().cpu())
    return (V@torch.diag(e.clamp(min=1e-14).rsqrt())@V.t()).to(dev).float()

log('вартість одного прогону = {:.3e} флоп; підлога множника = {:.3e} флоп ⟹ {:.0f} прогонів'
    .format(FWD,0.10*B,0.10*B/FWD))
log('')
NS=[1000,2000,4000,6300,10000,20000,50000,100000,200000]
REP=24
for net in (0,):
    W=load(net)
    S=torch.zeros(D,dtype=torch.float64,device=dev); n=0
    for c in range(40):
        g=torch.Generator(device=dev).manual_seed(6100+c)
        z=torch.randn(100_000,D,device=dev,generator=g)
        xx=torch.cat([z,-z],0); S+=fwd(W,xx).double().sum(0); n+=xx.shape[0]
    TH=S/n
    log('  N прогонів |   MSE(v2)   |  множник C/B | SCORE(v2)  | SCORE(анти) | v2 краще')
    for NF in NS:
        npair=NF//2
        ev=[];ea=[]
        for r in range(REP):
            g=torch.Generator(device=dev).manual_seed(51000+97*r)
            z=torch.randn(npair,D,device=dev,generator=g)
            X=torch.cat([z,-z],0)
            yA=fwd(W,X).double(); a_est=(yA[:npair]+yA[npair:]).mean(0)*0.5*2/2  # антитетика
            a_est=((yA[:npair]+yA[npair:])*0.5).mean(0)
            Xw=X@zca(X)
            yV=fwd(W,Xw).double(); v_est=((yV[:npair]+yV[npair:])*0.5).mean(0)
            ea.append(float(((a_est-TH)**2).mean())); ev.append(float(((v_est-TH)**2).mean()))
        mv,ma=float(np.mean(ev)),float(np.mean(ea))
        Fv=NF*FWD+NF*ZCA_PS+ZCA_FIX; Fa=NF*FWD
        mulv=max(0.10,Fv/B); mula=max(0.10,Fa/B)
        log('  {:9d} | {:.4e} |    {:.4f}    | {:.4e} | {:.4e} |  {:.2f}×'
            .format(NF,mv,mulv,mv*mulv,ma*mula,(ma*mula)/(mv*mulv)))
log('')
log('ЧИТАННЯ: якщо стовпчик SCORE плаский вище підлоги — N НЕ КУПУЄ нічого (скорочення точне).')
log('         якщо падає — дисперсія на семпл сама поліпшується з N (щілина реальна).')
log('done.')
