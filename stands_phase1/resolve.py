"""РОЗВ'ЯЗАННЯ СУПЕРЕЧКИ між lyap.py і gdecay.py.

lyap.py  : множники за шар 0.87-0.88 (стиснення), 40 проб, QR по 32 кроках
gdecay.py: ||G_s|| ПЛАСКА по глибині, множник 1.004 (нема стиснення)

Обидва не можуть бути правдою в наївному прочитанні. Дві гіпотези:
  (A) lyap не ЗБІГСЯ -- 40 випадкових проб у 65792 вимірах за 32 кроки не встигають
      вирівнятись на провідну моду, і 0.88 занижене
  (B) стиснення СПРАВЖНЄ у типовому напрямку, але Tr(K^L) зчитує саме ту моду,
      яка НЕ стискається -- тоді ||dK|| згасає, а |dTr| ні

ТЕСТ, що розрізняє: вкидаємо збурення на шарі s, женемо ВПЕРЕД до L і міряємо ОКРЕМО
   ||dK_L|| / ||dK_s||     (доля норми)
   |dTr_L|  / ||dK_s||     (доля того, що ми реально зчитуємо)
Якщо норма згасає, а слід ні -> (B). Якщо обидва пласкі -> (A).
Плюс: беремо не лише випадковий напрямок, а й НАПРЯМОК ІСТИННОЇ ПОХИБКИ ЗАМИКАННЯ.
"""
import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/resolve.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=20; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
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

def true_state(W,N=600_000,CH=60_000):
    s1=[torch.zeros(D,dtype=torch.float64,device=dev) for _ in range(L)]
    SK=[torch.zeros(D,D,dtype=torch.float64,device=dev) for _ in range(L)]
    n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(31000+c))
        x=torch.cat([z,-z],0).double()
        for li in range(L):
            x=relu(x@W[li]); s1[li]+=x.sum(0); SK[li]+=x.t()@x
        n+=x.shape[0]
    return [s/n for s in s1],[s/n for s in SK]

SEL=[0,4,8,12,16,20,24,28,30]
for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float64),device=dev,dtype=DT) for i in range(L)]
    log(''); log('######## НЕТ {} ########'.format(net))
    mt,KT=true_state(W)
    # траєкторія NNGP + істинна одношагова похибка замикання на кожному шарі
    ms=[];Ks=[];DK=[]
    m=torch.zeros(D,dtype=DT,device=dev); K=torch.eye(D,dtype=DT,device=dev)
    for l in range(L):
        ms.append(m); Ks.append(K)
        m,K=step(m,K,W[l])
    for l in range(L):
        mp,Kp=step(mt[l-1] if l>0 else torch.zeros(D,dtype=DT,device=dev),
                   KT[l-1] if l>0 else torch.eye(D,dtype=DT,device=dev), W[l])
        DK.append(0.5*((KT[l]-Kp)+(KT[l]-Kp).t()))     # істинна похибка замикання на шарі l

    g=torch.Generator(device=dev).manual_seed(9000+net)
    def push(s,dK0):
        """вкинути dK0 у стан ПІСЛЯ шару s, гнати вперед до L; повернути (||dK_L||, |dTr_L|)"""
        dm=torch.zeros(D,dtype=DT,device=dev); dK=dK0.clone()
        mm=Ks and ms[s]; KK=Ks[s]
        mm,KK=step(ms[s],Ks[s],W[s])
        for l in range(s+1,L):
            (_,_),(dm,dK)=torch.func.jvp(lambda a,b: step(a,b,W[l]),(mm,KK),(dm,dK))
            mm,KK=step(mm,KK,W[l])
        return float(dK.norm()), abs(float(torch.diagonal(dK).sum()))

    log('  s   (L-s)   ВИПАДКОВЕ збурення            ІСТИННА похибка замикання')
    log('              ||dK_L||/||dK_s||  |dTr|/n    ||dK_L||/||dK_s||  |dTr|/n')
    for s in SEL:
        R=torch.randn(D,D,dtype=DT,device=dev,generator=g); R=0.5*(R+R.t()); R=R/R.norm()
        nr,tr_=push(s,R)
        Dn=DK[s]/DK[s].norm()
        nd,td=push(s,Dn)
        log('  {:2d}   {:2d}      {:8.4f}          {:8.4f}      {:8.4f}          {:8.4f}'.format(
            s,L-s,nr,tr_,nd,td))
    log('  ЧИТАННЯ: якщо ||dK_L|| падає з (L-s), а |dTr| ні -> Tr зчитує НЕстискану моду (гіп.B)')
    log('           якщо обидва пласкі -> lyap.py не збігся (гіп.A)')
log('done.')
