"""РІВЕНЬ d³: врахувати ПАРИ індексів (між «незалежність» d² і «повна згортка» d⁴).

k3_cheap.py: наближення незалежності дає 3.5× замість оракульних 60×.
   cos(κ₃ дешевий, оракул)=0.78, норма 0.66  -> втрачені саме КОРЕЛЯЦІЇ

БЮДЖЕТ (уточнено!):  d⁴ × 32 шари = 1.4e11 -> C/B=0.51 ВИРОК
                     d³ × 32 шари = 5.4e8  -> C/B=0.002 БЕЗКОШТОВНО
⟹ проміжний рівень d³ (пари) вкладається із запасом ×50.

κ₃(h_j) = Σ_i W³ᵢⱼ κ₃(aᵢ)                                 [d², уже маємо]
        + 3 Σ_{i≠i'} W²ᵢⱼ Wᵢ'ⱼ κ₃(aᵢ,aᵢ,aᵢ')              [d³ через 2 матмули]
        + 6 Σ_{i<i'<i''} ...                               [d⁴, відкидаємо]

κ₃(aᵢ,aᵢ,aᵢ') для зрізаних гаусіан = функція ЛИШЕ ρᵢᵢ' (після нормування на sᵢ²sᵢ')
   -> табулюємо f(ρ) 2D-квадратурою на сітці, далі інтерполяція. O(d²) звернень.
Те саме для κ₄ з парами (домінантний доданок 4·W³W + 3·W²W²).
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/k3_pairs.log'
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
from numpy.polynomial.hermite import hermgauss

# ---- 1D моменти relu(g) ----
xq,wq=hermgauss(120)
zq=torch.tensor(xq*SQ2,dtype=DT,device=dev); wq1=torch.tensor(wq/math.sqrt(math.pi),dtype=DT,device=dev)
rq=torch.clamp(zq,min=0.0)
mm=[float((wq1*rq**k).sum()) for k in range(1,5)]
c3=mm[2]-3*mm[0]*mm[1]+2*mm[0]**3
c4=mm[3]-4*mm[0]*mm[2]-3*mm[1]**2+12*mm[0]**2*mm[1]-6*mm[0]**4
log('relu(g) g~N(0,1):  m1={:.6f} m2={:.6f} κ₃={:.6f} κ₄={:.6f}'.format(mm[0],mm[1],c3,c4))

# ---- ТАБЛИЦЯ мішаного кумулянта κ₃(r,r,r') як функції ρ (2D квадратура) ----
NG=64
xg,wg=hermgauss(NG)
u=torch.tensor(xg*SQ2,dtype=DT,device=dev); w2=torch.tensor(wg/math.sqrt(math.pi),dtype=DT,device=dev)
RHO=torch.linspace(-0.995,0.995,401,dtype=DT,device=dev)
def mixed(rho):
    """κ₃(r₁,r₁,r₂) для стандартних корельованих гаусіан (r=relu)"""
    a=u[:,None,None]; b=u[None,:,None]; rh=rho[None,None,:]
    g1=a; g2=rh*a+torch.sqrt((1-rh*rh).clamp(min=0))*b
    r1=torch.clamp(g1,min=0.0); r2=torch.clamp(g2,min=0.0)
    W=w2[:,None,None]*w2[None,:,None]
    E112=(W*r1*r1*r2).sum((0,1)); E11=(W*r1*r1).sum((0,1))
    E12=(W*r1*r2).sum((0,1)); E1=mm[0]; E2=mm[0]
    return E112-E11*E2-2*E12*E1+2*E1*E1*E2
TAB3=mixed(RHO)
def interp(tab,rho):
    i=((rho+0.995)/(1.99)*(len(RHO)-1)).clamp(0,len(RHO)-1-1e-6)
    lo=i.floor().long(); fr=i-lo
    return tab[lo]*(1-fr)+tab[(lo+1).clamp(max=len(RHO)-1)]*fr

for net in [0,1,2]:
    W0=torch.tensor(dat['w_%d_0'%net].astype(np.float32),device=dev).double()
    W1=torch.tensor(dat['w_%d_1'%net].astype(np.float32),device=dev).double()
    C1h=W0.t()@W0
    s1=torch.sqrt(torch.diagonal(C1h).clamp(min=1e-14)); m1=s1/SQ2PI
    rho=(C1h/torch.outer(s1,s1)).clamp(-0.995,0.995); ang=torch.acos(rho)
    E2=(torch.outer(s1,s1)/(2*math.pi))*(torch.sqrt((1-rho*rho).clamp(min=0))+(math.pi-ang)*rho)
    E2=E2-torch.diag(torch.diagonal(E2))+torch.diag(s1*s1*0.5)
    C1=E2-torch.outer(m1,m1)
    mu=W1.t()@m1; var=torch.diagonal(W1.t()@C1@W1).clamp(min=1e-14)
    sig=var.sqrt(); t=mu/sig; nn=mu*Phi(t)+sig*phi(t)

    # d²: незалежність
    k3_d2=(W1**3).t()@(c3*s1**3)
    k4_d2=(W1**4).t()@(c4*s1**4)
    # d³: пари.  B[i,i'] = κ₃(aᵢ,aᵢ,aᵢ') = f(ρ)·sᵢ²·sᵢ'
    B=interp(TAB3,rho)*(s1[:,None]**2)*s1[None,:]
    B=B-torch.diag(torch.diagonal(B))                      # i≠i'
    k3_d3=k3_d2+3.0*torch.einsum('ij,ik,jk->k',B,W1**2,W1)

    # істина + оракул
    NT=8_000_000; CH=200_000
    sa=torch.zeros(D,dtype=DT,device=dev)
    q1=torch.zeros(D,dtype=DT,device=dev); q2=torch.zeros(D,dtype=DT,device=dev)
    q3=torch.zeros(D,dtype=DT,device=dev); q4=torch.zeros(D,dtype=DT,device=dev); n=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(2500+net*7+c)
        z=torch.randn(CH//2,D,device=dev,generator=g,dtype=DT); x=torch.cat([z,-z],0)
        h2=relu(x@W0)@W1; d=h2-mu[None]
        sa+=relu(h2).sum(0)
        q1+=d.sum(0); q2+=(d*d).sum(0); q3+=(d**3).sum(0); q4+=(d**4).sum(0)
        n+=x.shape[0]
    TH=sa/n
    M1=q1/n; M2=q2/n-M1*M1; M3=q3/n-3*M1*(q2/n)+2*M1**3
    M4=q4/n-4*M1*(q3/n)+6*M1*M1*(q2/n)-3*M1**4
    k3o=M3; k4o=M4-3*M2*M2
    p=float((TH*TH).mean()); rel=lambda e: float(((e-TH)**2).mean())/p
    A3=lambda k: (k/6.0)*(-t*phi(t)/var)
    A4=lambda k: (k/24.0)*((t*t-1)*phi(t)/(sig*var))
    cs=lambda a_,b_: float((a_@b_)/(a_.norm()*b_.norm()))
    log('')
    log('#### НЕТ {} ####'.format(net))
    log('  κ₃: cos(d²,оракул)={:+.4f} норма {:.3f}  |  cos(d³,оракул)={:+.4f} норма {:.3f}'.format(
        cs(k3_d2,k3o),float(k3_d2.norm()/k3o.norm()),cs(k3_d3,k3o),float(k3_d3.norm()/k3o.norm())))
    log('  гаус                    {:.4e}'.format(rel(nn)))
    log('  + κ₃ d² + κ₄ d²         {:.4e}   ({:.1f}×)'.format(
        rel(nn+A3(k3_d2)+A4(k4_d2)),rel(nn)/max(rel(nn+A3(k3_d2)+A4(k4_d2)),1e-300)))
    log('  + κ₃ d³ + κ₄ d²         {:.4e}   ({:.1f}×)  <- ПАРИ, бюджет C/B=0.002'.format(
        rel(nn+A3(k3_d3)+A4(k4_d2)),rel(nn)/max(rel(nn+A3(k3_d3)+A4(k4_d2)),1e-300)))
    log('  + κ₃ d³ + κ₄ оракул     {:.4e}   ({:.1f}×)'.format(
        rel(nn+A3(k3_d3)+A4(k4o)),rel(nn)/max(rel(nn+A3(k3_d3)+A4(k4o)),1e-300)))
    log('  + κ₃+κ₄ оракул (стеля)  {:.4e}   ({:.1f}×)'.format(
        rel(nn+A3(k3o)+A4(k4o)),rel(nn)/max(rel(nn+A3(k3o)+A4(k4o)),1e-300)))
    log('  ОРІЄНТИР v2 при k=2: ~7.2e-7')
log('done.')
