"""ЧИ ЗОСЕРЕДЖЕНА ПОХИБКА ЗАМИКАННЯ НА ЖИВІЙ МЕНШОСТІ? (С47, продовження прибитих затворів)

ЛОГІКА: для ПРИБИТОГО нейрона relu(h)=h (або 0) тотожно ⟹ середнє проходить ТОЧНО,
форма розподілу байдужа ⟹ негаусовість НЕ МОЖЕ нашкодити. Значить похибка замикання
мусить сидіти на ЖИВИХ.

ВИМІР (ізолюємо ОДИН крок замикання, без накопичення):
  для кожного нейрона кожного шару беремо ІСТИННІ mu_i, sd_i розподілу h_i і рахуємо
     гаусів прогноз:  E_g[relu] = mu*Phi(mu/sd) + sd*phi(mu/sd)
     істину:          E_t[relu] = середнє relu(h_i) по семплах
  похибка замикання нейрона = |E_g - E_t|.  Це РІВНО ефект негаусовості, нічого іншого.

ЧИТАННЯ: сортуємо нейрони за «живістю» l_i = min(p_i, 1-p_i) і дивимось, яка частка
СУМАРНОЇ похибки припадає на живу меншість. Якщо 3% нейронів несуть 90% похибки —
ефективна розмірність задачі замикання падає в рази.
"""
import torch, time, numpy as np, math
LOG='/mnt/win_d/llm/White-Box/closure_where.log'; T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
SQ2PI=math.sqrt(2*math.pi)
def Phi(z): return 0.5*(1+torch.erf(z/math.sqrt(2)))
def phi(z): return torch.exp(-0.5*z*z)/SQ2PI

def run(net):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    N=400_000; CH=25_000
    S1=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]   # sum h
    S2=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]   # sum h^2
    SR=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]   # sum relu(h)
    SG=[torch.zeros(D,device=dev,dtype=torch.float64) for _ in range(L)]   # sum gate
    n=0
    for c in range(N//CH):
        g=torch.Generator(device=dev).manual_seed(2468+c)
        a=torch.randn(CH,D,device=dev,generator=g)
        for li in range(L):
            h=(a@W[li]).double()
            S1[li]+=h.sum(0); S2[li]+=(h*h).sum(0)
            r=torch.relu(h); SR[li]+=r.sum(0); SG[li]+=(h>0).double().sum(0)
            a=r.float()
        n+=CH
    log('  шар | жива меншість (min(p,1-p)>0.05) | ЇЇ частка сумарної похибки замикання | конц.')
    for li in (1,2,4,8,16,24,30,31):
        mu=S1[li]/n; sd=(S2[li]/n-mu*mu).clamp(min=1e-30).sqrt(); p=SG[li]/n
        z=mu/sd
        Eg=mu*Phi(z)+sd*phi(z)                     # гаусів прогноз
        Et=SR[li]/n                                # істина
        err=(Eg-Et).abs()
        live=torch.minimum(p,1-p)
        sel=live>0.05
        f_neu=float(sel.double().mean())
        f_err=float(err[sel].sum()/err.sum())
        log('   {:3d} |            {:.3f}             |               {:.3f}                | ×{:.1f}'
            .format(li,f_neu,f_err,f_err/max(f_neu,1e-9)))
    # деталізація по останньому шару: децилі за живістю
    li=L-1
    mu=S1[li]/n; sd=(S2[li]/n-mu*mu).clamp(min=1e-30).sqrt(); p=SG[li]/n
    z=mu/sd; err=(mu*Phi(z)+sd*phi(z)-SR[li]/n).abs()
    live=torch.minimum(p,1-p); idx=torch.argsort(live)
    tot=float(err.sum()); parts=[]
    for b in range(10):
        s=idx[b*D//10:(b+1)*D//10]; parts.append(float(err[s].sum())/tot)
    log('  шар 31, децилі за ЖИВІСТЮ (від найприбитіших до найживіших), частка похибки:')
    log('     '+'  '.join('{:.3f}'.format(x) for x in parts))
    log('     верхній дециль (10% найживіших) несе {:.1%} усієї похибки замикання'.format(parts[-1]))
    log('     нижня половина (найприбитіші 50%) несе {:.1%}'.format(sum(parts[:5])))

for net in (0,1):
    log('==== НЕТ {} ===='.format(net)); run(net)
log('done.')
