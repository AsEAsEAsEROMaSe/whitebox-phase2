"""ЕТАП 1 ЗБІРКИ: виправлений крок замикання — СЕРЕДНЄ *І* КОВАРІАЦІЯ. Валідація проти MC.

Поправка Еджворта до 2D-величини K_ij = E[relu(h_i)relu(h_j)]:
   E[f(h)] = E[f(g)] + (1/6)Σ κ_abc E[∂³f] + ...
   f = r(h_i)r(h_j),  r=relu:
      ∂³/∂h_i³        = δ'(h_i) r(h_j)
      ∂³/∂h_i²∂h_j    = δ(h_i) H(h_j)         (H = сходинка)
      ∂³/∂h_i∂h_j²    = H(h_i) δ(h_j)
      ∂³/∂h_j³        = r(h_i) δ'(h_j)
   ΔK_ij = (1/6)[ κ_iii·B_ij + 3κ_iij·A_ij + 3κ_ijj·A_ji + κ_jjj·B_ji ]

ЗАКРИТІ ФОРМУЛИ (двовимірна гаусіана, t=μ/σ, s=σ_j√(1−ρ²), m0=μ_j−ρσ_j t_i):
   A_ij = E[δ(g_i)H(g_j)]  = φ(t_i)/σ_i · Φ((t_j−ρt_i)/√(1−ρ²))
   B_ij = E[δ'(g_i)r(g_j)] = −[ t_iφ(t_i)/σ_i² · (m0Φ(m0/s)+sφ(m0/s))
                                + φ(t_i)/σ_i · Φ(m0/s) · ρσ_j/σ_i ]
Кумулянти h з кумулянтів a попереднього шару (наближення незалежності, d³ матмулами):
   κ_iii = Σ_a W_ai³ κ₃(a_a)          κ_iij = Σ_a W_ai² W_aj κ₃(a_a)
   -> M3 = (W∘W)ᵀ diag(κ₃) W   дає ВСЕ, разом з діагоналлю κ_iii
"""
import torch, math, time, numpy as np
LOG='/mnt/win_d/llm/White-Box/edge_step.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; DT=torch.float64
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
phi=lambda t: torch.exp(-0.5*t*t)/SQ2PI
Phi=lambda t: 0.5*(1+torch.erf(t/SQ2))
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
from numpy.polynomial.hermite import hermgauss
HQ=64
xq,wq=hermgauss(HQ)
zq=torch.tensor(xq*SQ2,dtype=DT,device=dev); wq1=torch.tensor(wq/math.sqrt(math.pi),dtype=DT,device=dev)

def cum_relu(mu,sd):
    """κ₃,κ₄ від relu(g), g~N(mu,sd) — покомпонентно, 1D квадратура"""
    y=relu(mu[None,:]+sd[None,:]*zq[:,None])           # (HQ,D)
    w=wq1[:,None]
    M1=(w*y).sum(0); M2=(w*y*y).sum(0); M3=(w*y**3).sum(0); M4=(w*y**4).sum(0)
    c2=M2-M1*M1
    c3=M3-3*M1*M2+2*M1**3
    c4=M4-4*M1*M3-3*M2*M2+12*M1*M1*M2-6*M1**4
    return c3,c4

def gauss_step(m,K,W):
    """звичайний NNGP-крок; повертає також μ,σ,ρ для поправок"""
    mu=W.t()@m; C=W.t()@(K-torch.outer(m,m))@W
    sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-14)); t=mu/sd
    Ph=Phi(t); ph=phi(t)
    mn=mu*Ph+sd*ph
    rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999)
    sq=torch.sqrt((1-rho*rho).clamp(min=1e-14)); sdsq=sd[None,:]*sq
    Kg=torch.zeros(D,D,dtype=DT,device=dev)
    for a in range(HQ):
        Ri=relu(mu+sd*zq[a]); gjb=mu[None,:]+sd[None,:]*(rho*zq[a])
        Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*zq[:,None,None])
        Kg=Kg+wq1[a]*Ri[:,None]*(wq1[:,None,None]*Rj).sum(0)
    dg=(mu*mu+sd*sd)*Ph+mu*sd*ph
    Kg=Kg-torch.diag(torch.diagonal(Kg))+torch.diag(dg)
    return mn,0.5*(Kg+Kg.t()),mu,sd,t,rho

def edge_terms(mu,sd,t,rho):
    """A_ij = E[δ(g_i)H(g_j)],  B_ij = E[δ'(g_i)r(g_j)]  — закриті формули"""
    sq=torch.sqrt((1-rho*rho).clamp(min=1e-12))
    A=(phi(t)/sd)[:,None]*Phi((t[None,:]-rho*t[:,None])/sq)
    m0=mu[None,:]-rho*sd[None,:]*t[:,None]
    s=sd[None,:]*sq
    Er=m0*Phi(m0/s)+s*phi(m0/s)
    B=-((t*phi(t)/(sd*sd))[:,None]*Er + (phi(t)/sd)[:,None]*Phi(m0/s)*(rho*sd[None,:]/sd[:,None]))
    return A,B

def dK3(kap3_h,A,B):
    """ΔK_ij = (1/6)[κ_iii B_ij + 3κ_iij A_ij + 3κ_ijj A_ji + κ_jjj B_ji]"""
    kiii=torch.diagonal(kap3_h)
    Kc=(kiii[:,None]*B + 3.0*kap3_h*A + 3.0*kap3_h.t()*A.t() + kiii[None,:]*B.t())/6.0
    return 0.5*(Kc+Kc.t())

# ================= ВАЛІДАЦІЯ на кроці 1->2 =================
for net in [0,1]:
    W0=torch.tensor(dat['w_%d_0'%net].astype(np.float32),device=dev).double()
    W1=torch.tensor(dat['w_%d_1'%net].astype(np.float32),device=dev).double()
    log(''); log('######## НЕТ {} — валідація кроку 1→2 ########'.format(net))
    # стан після шару 0 (ТОЧНИЙ: вхід рівно гаусів)
    m0_=torch.zeros(D,dtype=DT,device=dev); K0=torch.eye(D,dtype=DT,device=dev)
    m1,K1,mu1,sd1,t1,rho1=gauss_step(m0_,K0,W0)
    c3a,c4a=cum_relu(mu1,sd1)                     # кумулянти a¹
    # крок 1->2
    m2g,K2g,mu2,sd2,t2,rho2=gauss_step(m1,K1,W1)
    M3=(W1*W1).t()@torch.diag(c3a)@W1             # κ_iij (діагональ = κ_iii)
    A,B=edge_terms(mu2,sd2,t2,rho2)
    K2c=K2g+dK3(M3,A,B)
    # ---- MC-істина для K² і m² ----
    NT=6_000_000; CH=200_000
    sA=torch.zeros(D,dtype=DT,device=dev); sK=torch.zeros(D,D,dtype=DT,device=dev); n=0
    for c in range(NT//CH):
        g=torch.Generator(device=dev).manual_seed(4400+net*9+c)
        z=torch.randn(CH//2,D,device=dev,generator=g,dtype=DT); x=torch.cat([z,-z],0)
        a2=relu(relu(x@W0)@W1)
        sA+=a2.sum(0); sK+=a2.t()@a2; n+=x.shape[0]
    mT=sA/n; KT=sK/n
    e=lambda M: float((M-KT).norm()/KT.norm())
    log('  ПЕРЕВІРКА ПОПРАВКИ КОВАРІАЦІЇ:')
    log('    ||K_гаус − K_іст||/||K_іст||       = {:.4e}'.format(e(K2g)))
    log('    ||K_виправл − K_іст||/||K_іст||    = {:.4e}   ({:.2f}× краще)'.format(
        e(K2c), e(K2g)/max(e(K2c),1e-300)))
    log('    (діагональ) сліди: гаус {:.4f}  виправл {:.4f}  істина {:.4f}'.format(
        float(torch.diagonal(K2g).sum()),float(torch.diagonal(K2c).sum()),
        float(torch.diagonal(KT).sum())))
    em=lambda v: float((v-mT).norm()/mT.norm())
    k3h=torch.diagonal(M3); k4h=((W1**4).t()@c4a)
    var2=sd2*sd2
    d3=(k3h/6.0)*(-t2*phi(t2)/var2)
    d4=(k4h/24.0)*((t2*t2-1)*phi(t2)/(sd2*var2))
    log('  ПЕРЕВІРКА ПОПРАВКИ СЕРЕДНЬОГО:')
    log('    ||m_гаус − m_іст||/||m||           = {:.4e}'.format(em(m2g)))
    log('    + κ₃                               = {:.4e}   ({:.2f}×)'.format(
        em(m2g+d3), em(m2g)/max(em(m2g+d3),1e-300)))
    log('    + κ₃ + κ₄                          = {:.4e}   ({:.2f}×)'.format(
        em(m2g+d3+d4), em(m2g)/max(em(m2g+d3+d4),1e-300)))
log('done.')
