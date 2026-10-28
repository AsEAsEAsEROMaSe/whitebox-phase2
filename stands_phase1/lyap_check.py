"""КАВЕРЗА до lyap.py: чи стиснення СПРАВЖНЄ, чи артефакт масштабу стану?

Якщо ||K|| саме по собі спадає уздовж глибини, то множники <1 нічого не означають --
збурення просто масштабується разом зі станом. Критична ініціалізація 2/256 МАЄ
зберігати масштаб, але це треба ПЕРЕВІРИТИ, а не припустити.

Плюс: перевірка кількісної згоди.  Якщо крок стискає з множником c, а похибка
замикання вкидається на КОЖНОМУ шарі, то накопичене = геометрична сума 1/(1-c).
compound.py виміряв підсилення x6.1 / 6.9 / 11.2.  Чи це 1/(1-c) при c~0.85-0.91?
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/lyap_check.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m)
    print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=20
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi); DT=torch.float64
relu=lambda x: torch.clamp(x,min=0.0)
from numpy.polynomial.hermite import hermgauss
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=DT,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=DT,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))

def step(m,K,W):
    mu=W.t()@m; C=W.t()@(K-torch.outer(m,m))@W
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14))
    t=mu/sd; Ph=Phi(t); ph=phi(t); m2=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999)
    sq=torch.sqrt((1-rho*rho).clamp(min=1e-14)); sdsq=sd[None,:]*sq
    Kg=torch.zeros(D,D,dtype=DT,device=dev)
    for a in range(H):
        Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
        Kg=Kg+wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
    dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
    Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg)
    return m2,0.5*(Kg+Kg.t())

for net in [0,1,2]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float64),device=dev,dtype=DT) for i in range(L)]
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev)
    nk=[]; tr=[]
    for ell in range(L):
        m,K=step(m,K,W[ell])
        nk.append(float(K.norm())); tr.append(float(torch.diagonal(K).sum()))
    nk=np.array(nk); tr=np.array(tr)
    log('')
    log('#### НЕТ {} ####'.format(net))
    log('  ||K|| по шарах 0,7,15,23,31 : ' + ' '.join('{:.4f}'.format(nk[i]) for i in (0,7,15,23,31)))
    log('  Tr K  по шарах 0,7,15,23,31 : ' + ' '.join('{:.2f}'.format(tr[i]) for i in (0,7,15,23,31)))
    # власний множник масштабу стану за крок
    sc=np.exp(np.diff(np.log(nk)).mean())
    log('  МАСШТАБ стану: множник ||K|| за крок = {:.4f}   (за 32 шари {:.3f}x)'.format(
        sc, nk[-1]/nk[0]))
    log('  -> якщо це ~1.00, а множники Ляпунова 0.88 -- стиснення СПРАВЖНЄ')

log('')
log('=== ГЕОМЕТРИЧНА СУМА: чи пояснює вона підсилення x6-11 з compound.py ? ===')
log('   похибка вкидається на КОЖНОМУ шарі й стискається з множником c:')
log('   накопичене / одношагове  =  (1-c^32)/(1-c)')
for c in (0.80,0.84,0.85,0.87,0.88,0.90,0.92,0.94,0.96):
    log('     c={:.2f} -> {:6.2f}'.format(c,(1-c**32)/(1-c)))
log('   ВИМІРЯНО (compound.py): 6.1 / 6.9 / 11.2')
log('done.')
