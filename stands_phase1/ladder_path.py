"""ШЛЯХ ПО ЩАБЛЯХ (ідея юзера С47): вимкнути всі ReLU, вмикати по одному, дивитись зміни.

КЛЮЧ: якщо ReLU увімкнено лише на шарах 1..k, а далі лінійно, то модель = a^k @ Q_k,
де a^k — активація СПРАВЖНЬОЇ мережі, Q_k = W_{k+1}...W_L (33 матриці, ціна d^3 = безкоштовно).
=> ОДИН форвард дає ОДРАЗУ ВСІ 33 щаблі.

ТЕЛЕСКОП: M_k - M_{k-1} = E[relu(h^k) - h^k] @ Q_k = -E[relu(-h^k)] @ Q_k  («зрізане»).
   M_0 = x@Q_0 (середнє 0 точно),  M_L = a^L (відповідь).  Sum Delta_k = вся відповідь.

МІРЯЄМО (на антитетичних парах — те, що реально оцінює v2):
  A) ШЛЯХ: ||E[M_k]|| — як відповідь наростає по щаблях
  B) ШУМ прямої оцінки на щаблі k: Var(M_k) на пару
  C) ШУМ приросту: Var(Delta_k) на пару
  D) КРИТЕРІЙ ТЕЛЕСКОПА: (sum_k sqrt(Var Delta_k))^2  проти  Var(M_L).
     менше => телескоп дешевший за пряму оцінку при однаковій точності.
  Все нормовано на ||E[a^L]||^2, щоб числа читались.
"""
import torch, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/ladder_path.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; DT=torch.float64
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev).double() for i in range(L)]
    # Q[k] = W_{k+1}...W_{L}  (хвіст ПІСЛЯ шару k). Q[L]=I
    Q=[None]*(L+1); Q[L]=torch.eye(D,dtype=DT,device=dev)
    for k in range(L-1,-1,-1): Q[k]=W[k]@Q[k+1]
    qn=[float(Q[k].norm()) for k in range(L+1)]
    log('  ||Q_k||: k=0 {:.3e}  k=8 {:.3e}  k=16 {:.3e}  k=24 {:.3e}  k=32 {:.3e}'
        .format(qn[0],qn[8],qn[16],qn[24],qn[32]))
    NP=200_000; CH=20_000
    S=torch.zeros(L+1,D,dtype=DT,device=dev)          # сума E[M_k]
    S2=torch.zeros(L+1,dtype=DT,device=dev)           # сума ||M_k||^2 (для Var)
    DS=torch.zeros(L+1,D,dtype=DT,device=dev)         # сума Delta_k
    DS2=torch.zeros(L+1,dtype=DT,device=dev)
    n=0
    for c in range(NP//CH):
        gg=torch.Generator(device=dev).manual_seed(5500+c)
        z=torch.randn(CH,D,device=dev,generator=gg,dtype=DT)
        x=torch.cat([z,-z],0)
        a=x; prev=None
        for k in range(L+1):
            Mk=a@Q[k]                                  # модель зі щаблем k
            Mp=(Mk[:CH]+Mk[CH:])*0.5                   # парна частина (антитетична пара)
            S[k]+=Mp.sum(0); S2[k]+=(Mp*Mp).sum()
            if k>0:
                Dl=Mp-prev
                DS[k]+=Dl.sum(0); DS2[k]+=(Dl*Dl).sum()
            prev=Mp; del Mk
            if k<L: a=torch.relu(a@W[k])
        n+=CH
    E=S/n; ED=DS/n
    varM=S2/n-(E*E).sum(1)                             # дисперсія на пару, сумарна по D
    varD=DS2/n-(ED*ED).sum(1)
    ref=float((E[L]*E[L]).sum())                       # ||E[a^L]||^2
    log('  A) ШЛЯХ ||E[M_k]||^2 / ||E[a^L]||^2:')
    s=['{}:{:.4f}'.format(k,float((E[k]*E[k]).sum())/ref) for k in range(L+1)]
    for i in range(0,L+1,11): log('     '+'  '.join(s[i:i+11]))
    log('  B) ШУМ прямої оцінки Var(M_k)/||E[a^L]||^2:')
    s=['{}:{:.2e}'.format(k,float(varM[k])/ref) for k in range(L+1)]
    for i in range(0,L+1,8): log('     '+'  '.join(s[i:i+8]))
    log('  C) внесок щабля ||Delta_k||^2/ref та його ШУМ Var(Delta_k)/ref:')
    s=['{}: вн={:.2e} шум={:.2e}'.format(k,float((ED[k]*ED[k]).sum())/ref,float(varD[k])/ref)
       for k in range(1,L+1)]
    for i in range(0,L,4): log('     '+'  '.join(s[i:i+4]))
    tel=float((varD[1:].clamp(min=0).sqrt()).sum())**2
    log('  D) КРИТЕРІЙ: (sum sqrt(Var Delta))^2 = {:.3e}   Var(M_L) = {:.3e}   -> телескоп {}'
        .format(tel/ref,float(varM[L])/ref,
                'ВИГРАЄ x{:.2f}'.format(float(varM[L])/tel) if tel<float(varM[L]) else
                'ПРОГРАЄ x{:.2f}'.format(tel/float(varM[L]))))
    # скільки внеску дає перший щабель (його ми знаємо ТОЧНО і безкоштовно)
    log('     внесок щабля 1 (знаємо аналітично) = {:.3e} від відповіді; його шум = {:.2e}'
        .format(float((ED[1]*ED[1]).sum())/ref,float(varD[1])/ref))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
