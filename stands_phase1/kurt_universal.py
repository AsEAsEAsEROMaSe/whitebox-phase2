"""ЧИ УНІВЕРСАЛЬНИЙ КУРТОЗИС? (можливий БЕЗКОШТОВНИЙ точний якір)

closure_bench.py: на кроці 1→2 поправка Еджворта з ОРАКУЛЬНИМИ κ₃,κ₄ дає 64× і б'є v2 у 8×.
Впало в очі: γ₂ сер = +0.0392 / +0.0391 / +0.0379 на ТРЬОХ різних нетах — збіг до 3-ї цифри.
І 0.039 × 256 = 10.0  ->  схоже на ТОЧНИЙ ЗАКОН γ₂ = c/d, універсальну константу.

Якщо так — κ₄ = γ₂·σ⁴ БЕЗКОШТОВНИЙ (σ знаємо точно), а це рівно той дефіцит,
у який упирається весь проєкт ([[finding-anchor-cost-identity]]).

ПЕРЕВІРЯЄМО:
  1. РОЗКИД γ₂ по компонентах (а не лише середнє) — чи сталий насправді
  2. що дає κ₄ ОКРЕМО, без κ₃
  3. чи вистачить УНІВЕРСАЛЬНОЇ константи замість покомпонентного оракула
  4. чи тримається закон при іншій ширині d (128, 192, 256) — підріз ваг
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/kurt_universal.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))

def bench(W0,W1,D,seed,NT=8_000_000,CH=200_000):
    C1h=W0.t()@W0
    s1=torch.sqrt(torch.diagonal(C1h).clamp(min=1e-14))
    m1=s1/SQ2PI
    rho=(C1h/torch.outer(s1,s1)).clamp(-0.999999,0.999999)
    ang=torch.acos(rho)
    E2=(torch.outer(s1,s1)/(2*math.pi))*(torch.sqrt((1-rho*rho).clamp(min=0))+(math.pi-ang)*rho)
    E2=E2-torch.diag(torch.diagonal(E2))+torch.diag(s1*s1*0.5)
    C1=E2-torch.outer(m1,m1)
    mu=W1.t()@m1
    var=torch.diagonal(W1.t()@C1@W1).clamp(min=1e-14)
    sig=var.sqrt(); t=mu/sig
    nn=mu*Phi(t)+sig*phi(t)
    s_a=torch.zeros(D,dtype=DT,device=dev)
    c1_=torch.zeros(D,dtype=DT,device=dev); c2_=torch.zeros(D,dtype=DT,device=dev)
    c3_=torch.zeros(D,dtype=DT,device=dev); c4_=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(seed+c)
        z=torch.randn(CH//2,D,device=dev,generator=g,dtype=DT); x=torch.cat([z,-z],0)
        h2=relu(x@W0)@W1; d=h2-mu[None]
        s_a+=relu(h2).sum(0)
        c1_+=d.sum(0); c2_+=(d*d).sum(0); c3_+=(d**3).sum(0); c4_+=(d**4).sum(0)
        n+=x.shape[0]
    TH=s_a/n
    M1=c1_/n; M2=c2_/n-M1*M1; M3=c3_/n-3*M1*(c2_/n)+2*M1**3
    M4=c4_/n-4*M1*(c3_/n)+6*M1*M1*(c2_/n)-3*M1**4
    k3=M3; k4=M4-3*M2*M2
    return TH,nn,mu,sig,var,t,k3,k4,M2

for net in [0,1]:
    for D in (256,192,128):
        W0=torch.tensor(dat['w_%d_0'%net].astype(np.float32),device=dev).double()[:D,:D]*math.sqrt(256/D)
        W1=torch.tensor(dat['w_%d_1'%net].astype(np.float32),device=dev).double()[:D,:D]*math.sqrt(256/D)
        TH,nn,mu,sig,var,t,k3,k4,M2=bench(W0,W1,D,7000+net*101+D)
        g1=k3/M2.pow(1.5); g2=k4/(M2*M2)
        p=float((TH*TH).mean())
        rel=lambda e: float(((e-TH)**2).mean())/p
        d3=(k3/6.0)*(-t*phi(t)/var)
        d4=(k4/24.0)*((t*t-1)*phi(t)/(sig*var))
        g2u=float(g2.mean())
        d4u=((g2u*M2*M2)/24.0)*((t*t-1)*phi(t)/(sig*var))    # УНІВЕРСАЛЬНА константа
        log('')
        log('#### НЕТ {}  ШИРИНА d={} ####'.format(net,D))
        log('  γ₂: сер {:+.5f}   ст.відх {:.5f}   мін {:+.4f}  макс {:+.4f}   γ₂·d = {:.3f}'.format(
            g2u,float(g2.std()),float(g2.min()),float(g2.max()),g2u*D))
        log('  γ₁: сер {:+.5f}   ст.відх {:.5f}   (|γ₁| сер {:.5f})'.format(
            float(g1.mean()),float(g1.std()),float(g1.abs().mean())))
        log('  гаус             {:.4e}'.format(rel(nn)))
        log('  + κ₄ ОКРЕМО      {:.4e}   ({:.1f}×)'.format(rel(nn+d4),rel(nn)/max(rel(nn+d4),1e-300)))
        log('  + κ₄ УНІВЕРС.    {:.4e}   ({:.1f}×)  <- БЕЗ оракула на κ₄'.format(
            rel(nn+d4u),rel(nn)/max(rel(nn+d4u),1e-300)))
        log('  + κ₃ ОКРЕМО      {:.4e}   ({:.1f}×)'.format(rel(nn+d3),rel(nn)/max(rel(nn+d3),1e-300)))
        log('  + κ₃+κ₄ оракул   {:.4e}   ({:.1f}×)'.format(rel(nn+d3+d4),rel(nn)/max(rel(nn+d3+d4),1e-300)))
log('done.')
