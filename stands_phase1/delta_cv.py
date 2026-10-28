"""ЧИ Є ЧЛЕН 1-го ПОРЯДКУ ХОРОШОЮ CV? (єдина щілина, що лишилась від δ-розкладу)

ЛОГІКА. Тотожність вартості якоря вимагає: якір ТОЧНИЙ і БЕЗКОШТОВНИЙ.
Член 1-го порядку дає рідкісну конфігурацію:
  - його СЕРЕДНЄ = 2.6e-5 від ||E[a^L]||  (delta1.py)
  - отже навіть 100% похибка в аналітичній формулі -> зсув 2.6e-5 = НІЩО
  - а ПОСЕМПЛОВО він може мати велику дисперсію й корелювати з a^L
⟹ якщо R² помітний, маємо CV з фактично точним і безкоштовним якорем.

ОЗНАКА (посемплово, з ОДНОГО форварду):
  t_l = 2^{-(L-1)} · (u^l ⊙ δ_l) · Q_l,   u^l = x·W_1···W_l,  Q_l = W_{l+1}···W_L

⚠️ held-out ОБОВ'ЯЗКОВО; R² від ПОСЕМПЛОВОЇ дисперсії.
⚠️ ознаки НЕ зберігаємо (5.3 ГБ у fp64) — моменти накопичуємо ПОТОКОВО, два проходи.
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/delta_cv.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[0,2,4,8,12,16,20,24,27,29,30,31]
K=len(TAPS)
N=200_000; CH=20_000; NCH=N//CH; HALF=NCH//2

for net in [0,1]:
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    Wd=[w.double() for w in W]
    Q=[None]*L; Q[L-1]=torch.eye(D,dtype=torch.float64,device=dev)
    for li in range(L-2,-1,-1): Q[li]=Wd[li+1]@Q[li+1]
    Qf=[q.float() for q in Q]
    pref=2.0**(-(L-1))
    log(''); log('######## НЕТ {}  (N={}, held-out) ########'.format(net,N))

    def chunk(c):
        g=torch.Generator(device=dev).manual_seed(6600+net*17+c)
        z=torch.randn(CH//2,D,device=dev,generator=g)
        x=torch.cat([z,-z],0)
        a=x; u=x; ts=[]
        for li in range(L):
            h=a@W[li]; u=u@W[li]
            gt=(h>0).to(h.dtype)
            if li in TAPS: ts.append(pref*((u*(gt-0.5))@Qf[li]))
            a=h*gt
        return a.double(), torch.stack(ts,1).double()      # (n,D), (n,K,D)

    # ---- прохід 1: моменти на ПЕРШІЙ половині ----
    sA=torch.zeros(D,dtype=torch.float64,device=dev)
    sF=torch.zeros(K,D,dtype=torch.float64,device=dev)
    sFF=torch.zeros(D,K,K,dtype=torch.float64,device=dev)
    sFA=torch.zeros(D,K,dtype=torch.float64,device=dev)
    sAA=torch.zeros(D,dtype=torch.float64,device=dev); n1=0
    for c in range(HALF):
        a,f=chunk(c)
        sA+=a.sum(0); sF+=f.sum((0,)); sAA+=(a*a).sum(0)
        sFF+=torch.einsum('nkj,nlj->jkl',f,f)
        sFA+=torch.einsum('nkj,nj->jk',f,a)
        n1+=a.shape[0]
    mA=sA/n1; mF=sF/n1
    S=sFF/n1-torch.einsum('kj,lj->jkl',mF,mF)
    b=sFA/n1-(mF.t()*mA[:,None])
    # ⚠️ ознаки мають шалений динамічний діапазон (2^-31 × ||ΠW||~2^15) -> без
    # нормування linalg.solve вибухає (R² ~ -1e270). Нормуємо на std і ридж ВІДНОСНИЙ.
    sd=torch.diagonal(S,dim1=1,dim2=2).clamp(min=1e-300).sqrt()                  # (D,K)
    Sn=S/(sd[:,:,None]*sd[:,None,:])
    ridge=1e-6*torch.eye(K,dtype=torch.float64,device=dev)[None]
    beta=torch.linalg.solve((Sn+ridge).cpu(),(b/sd).cpu().unsqueeze(-1)).squeeze(-1).to(dev)/sd
    vA1=(sAA/n1-mA*mA).clamp(min=0)
    alive=vA1>1e-6*float(vA1.max())        # 1e-9 пускало майже-мертві -> вибух R²
    log('  ||E[a^L]|| = {:.5f}   живих {}'.format(float(mA.norm()),int(alive.sum())))
    log('  масштаб: std(a) сер {:.4e}'.format(float(vA1.clamp(min=0).sqrt().mean())))

    # ---- прохід 2: залишкова дисперсія на ДРУГІЙ половині ----
    def r2_of(bt, idx):
        s1=torch.zeros(D,dtype=torch.float64,device=dev)
        s2=torch.zeros(D,dtype=torch.float64,device=dev)
        sa=torch.zeros(D,dtype=torch.float64,device=dev)
        sa2=torch.zeros(D,dtype=torch.float64,device=dev); n2=0
        for c in range(HALF,NCH):
            a,f=chunk(c)
            p=torch.einsum('nkj,jk->nj',f[:,idx]-mF[idx][None],bt)
            ac=a-mA[None]                      # ⚠️ ЦЕНТРУВАТИ перед квадратом:
            r=ac-p                             # E[a²]−E[a]² дає катастрофічне скорочення
            s1+=r.sum(0); s2+=(r*r).sum(0)
            sa+=ac.sum(0); sa2+=(ac*ac).sum(0); n2+=a.shape[0]
        vr=(s2/n2-(s1/n2)**2).clamp(min=0)
        va=(sa2/n2-(sa/n2)**2).clamp(min=0)
        return (1-vr/va.clamp(min=1e-30))
    r2=r2_of(beta,list(range(K)))
    log('  HELD-OUT R² (усі {} ознак t_l): сер {:+.5f}  медіана {:+.5f}  макс {:+.5f}'.format(
        K,float(r2[alive].mean()),float(r2[alive].median()),float(r2[alive].max())))
    for nm,idx in (('останні 4 шари',list(range(K-4,K))),('перші 4 шари',list(range(4)))):
        Ss=S[:,idx][:,:,idx]; bs=b[:,idx]; sds=sd[:,idx]
        Sns=Ss/(sds[:,:,None]*sds[:,None,:])
        rg=1e-6*torch.eye(len(idx),dtype=torch.float64,device=dev)[None]
        bt=torch.linalg.solve((Sns+rg).cpu(),(bs/sds).cpu().unsqueeze(-1)).squeeze(-1).to(dev)/sds
        rr=r2_of(bt,idx)
        log('    лише {:15s}: R² сер {:+.5f}'.format(nm,float(rr[alive].mean())))
log('')
log("ОРІЄНТИР: 8 слідів Tr(K^l) ~0.70; вся ерміт-сім'я по входу 0.010; затвори шару31 0.444.")
log('done.')
