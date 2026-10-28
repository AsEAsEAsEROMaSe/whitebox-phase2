"""ЧИ Є ДЕШЕВИЙ κ₃? (вирішальний тест)

kurt_universal.py: на кроці замикання ГОЛОВНИЙ член — це СКІС κ₃ (7-11×), а не κ₄ (1.3×).
κ₃ покомпонентний (γ₁ сер=0, розкид ±0.036) -> універсальна константа НЕ поможе.

ПОВНИЙ κ₃(h²_j) = Σ_{i,i',i''} W_ij W_i'j W_i''j · κ₃(a¹_i,a¹_i',a¹_i'')
   -> d³ трійок × d виходів = d⁴ на шар = ВИРОК за бюджетом (пам'ять: n⁴=вирок)

ДЕШЕВЕ НАБЛИЖЕННЯ (нехтуємо кореляціями a¹, вони ~1/√d≈0.06):
   κ₃(h²_j) ≈ Σ_i W_ij³ · κ₃(a¹_i),   κ₃(relu(g)) = c₃·s³ — ЗАКРИТА ФОРМУЛА
   -> d² на шар = БЕЗКОШТОВНО (один matmul W³ на вектор)

ТЕ САМЕ для κ₄:  κ₄(h²_j) ≈ Σ_i W_ij⁴ · κ₄(a¹_i)

ПИТАННЯ: скільки з 7× (κ₃) і 60× (κ₃+κ₄) виживе без кореляцій?
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/k3_cheap.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))

# --- кумулянти relu(g), g~N(0,1): коефіцієнти c₃,c₄ (квадратура, щоб без помилок алгебри) ---
from numpy.polynomial.hermite import hermgauss
xq,wq=hermgauss(120)          # ⚠️ 400 дає NaN-ваги (переповнення 1/(fm*fm)) — відома пастка
zq=torch.tensor(xq*SQ2,dtype=DT); wq=torch.tensor(wq/math.sqrt(math.pi),dtype=DT)
r=torch.clamp(zq,min=0.0)
mm=[float((wq*r**k).sum()) for k in range(1,5)]
c3=mm[2]-3*mm[0]*mm[1]+2*mm[0]**3
c4=mm[3]-4*mm[0]*mm[2]-3*mm[1]**2+12*mm[0]**2*mm[1]-6*mm[0]**4
log('коефіцієнти для relu(g), g~N(0,1):  κ₃ = {:.6f}·s³   κ₄ = {:.6f}·s⁴'.format(c3,c4))

for net in [0,1,2]:
    W0=torch.tensor(dat['w_%d_0'%net].astype(np.float32),device=dev).double()
    W1=torch.tensor(dat['w_%d_1'%net].astype(np.float32),device=dev).double()
    C1h=W0.t()@W0
    s1=torch.sqrt(torch.diagonal(C1h).clamp(min=1e-14))
    m1=s1/SQ2PI
    rho=(C1h/torch.outer(s1,s1)).clamp(-0.999999,0.999999); ang=torch.acos(rho)
    E2=(torch.outer(s1,s1)/(2*math.pi))*(torch.sqrt((1-rho*rho).clamp(min=0))+(math.pi-ang)*rho)
    E2=E2-torch.diag(torch.diagonal(E2))+torch.diag(s1*s1*0.5)
    C1=E2-torch.outer(m1,m1)
    mu=W1.t()@m1
    var=torch.diagonal(W1.t()@C1@W1).clamp(min=1e-14); sig=var.sqrt(); t=mu/sig
    nn=mu*Phi(t)+sig*phi(t)

    # ДЕШЕВІ κ₃,κ₄ — БЕЗ жодного семпла, лише закриті формули + два матмули
    k3c=(W1**3).t()@(c3*s1**3)
    k4c=(W1**4).t()@(c4*s1**4)

    # істина + оракульні кумулянти
    NT=8_000_000; CH=200_000
    sa=torch.zeros(D,dtype=DT,device=dev)
    a1_=torch.zeros(D,dtype=DT,device=dev); a2_=torch.zeros(D,dtype=DT,device=dev)
    a3_=torch.zeros(D,dtype=DT,device=dev); a4_=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(1500+net*7+c)
        z=torch.randn(CH//2,D,device=dev,generator=g,dtype=DT); x=torch.cat([z,-z],0)
        h2=relu(x@W0)@W1; d=h2-mu[None]
        sa+=relu(h2).sum(0)
        a1_+=d.sum(0); a2_+=(d*d).sum(0); a3_+=(d**3).sum(0); a4_+=(d**4).sum(0)
        n+=x.shape[0]
    TH=sa/n
    M1=a1_/n; M2=a2_/n-M1*M1; M3=a3_/n-3*M1*(a2_/n)+2*M1**3
    M4=a4_/n-4*M1*(a3_/n)+6*M1*M1*(a2_/n)-3*M1**4
    k3o=M3; k4o=M4-3*M2*M2
    p=float((TH*TH).mean()); rel=lambda e: float(((e-TH)**2).mean())/p
    A3=lambda k3: (k3/6.0)*(-t*phi(t)/var)
    A4=lambda k4: (k4/24.0)*((t*t-1)*phi(t)/(sig*var))
    cs3=float((k3c@k3o)/(k3c.norm()*k3o.norm()))
    cs4=float((k4c@k4o)/(k4c.norm()*k4o.norm()))
    log('')
    log('#### НЕТ {} ####'.format(net))
    log('  ДЕШЕВИЙ проти ОРАКУЛА:  cos(κ₃)={:+.4f}  ||відн||={:.3f}   cos(κ₄)={:+.4f}  ||відн||={:.3f}'.format(
        cs3,float(k3c.norm()/k3o.norm()),cs4,float(k4c.norm()/k4o.norm())))
    log('  гаус                       {:.4e}'.format(rel(nn)))
    log('  + κ₃ ДЕШЕВИЙ               {:.4e}   ({:.1f}×)'.format(
        rel(nn+A3(k3c)),rel(nn)/max(rel(nn+A3(k3c)),1e-300)))
    log('  + κ₃ оракул                {:.4e}   ({:.1f}×)'.format(
        rel(nn+A3(k3o)),rel(nn)/max(rel(nn+A3(k3o)),1e-300)))
    log('  + κ₃+κ₄ ДЕШЕВІ             {:.4e}   ({:.1f}×)  <- БЕЗ СЕМПЛІВ, d² на шар'.format(
        rel(nn+A3(k3c)+A4(k4c)),rel(nn)/max(rel(nn+A3(k3c)+A4(k4c)),1e-300)))
    log('  + κ₃+κ₄ оракул             {:.4e}   ({:.1f}×)'.format(
        rel(nn+A3(k3o)+A4(k4o)),rel(nn)/max(rel(nn+A3(k3o)+A4(k4o)),1e-300)))
    log('  ОРІЄНТИР v2 при k=2: ~7.2e-7')
log('done.')
