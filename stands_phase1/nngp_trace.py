import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/nngp_trace.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=24
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)
d=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
W=[torch.tensor(d['w_0_%d'%i].astype(np.float32),device=dev) for i in range(L)]
TAPS=[3,7,11,15,19,23,27,30]

# ---------- NNGP propagation, collect Tr(K) at taps ----------
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))
def nngp():
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); tr={}
    for ell in range(L):
        mu=W[ell].t()@m; C=W[ell].t()@(K-torch.outer(m,m))@W[ell]
        sd=torch.sqrt(torch.diagonal(C).clamp(min=1e-12)); t=mu/sd; Ph=Phi(t); ph=phi(t)
        m=mu*Ph+sd*ph
        rho=(C/torch.outer(sd,sd)).clamp(-0.999999,0.999999); sq=torch.sqrt(1-rho*rho)
        Kg=torch.zeros(D,D,device=dev); sdsq=sd[None,:]*sq
        for a in range(H):
            Ri=relu(mu+sd*xn[a]); gjb=mu[None,:]+sd[None,:]*(rho*xn[a])
            Rj=relu(gjb[None,:,:]+sdsq[None,:,:]*xn[:,None,None])
            Kg+=wn[a]*Ri[:,None]*(wn[:,None,None]*Rj).sum(0)
        Kg[range(D),range(D)]=(mu*mu+sd*sd)*Ph+mu*sd*ph      # exact diagonal
        K=Kg
        if ell in TAPS: tr[ell]=float(torch.diagonal(K).sum())
    return tr
log('NNGP propagation ...'); TRN=nngp(); log('NNGP traces done')

# ---------- truth: E[||a^l||^2] and regression of Y on the 8 norms ----------
def run(w):
    def fwd(x):
        out=[]
        for li in range(L):
            x=relu(x@W[li])
            if li in TAPS: out.append((x*x).sum(1))
        return x,out
    ap,na=fwd(w); am,nb=fwd(-w)
    return 0.5*(ap+am), torch.stack([0.5*(a+b) for a,b in zip(na,nb)],1)

k=len(TAPS)
A=torch.zeros(k,k,dtype=torch.float64,device=dev); BY=torch.zeros(k,D,dtype=torch.float64,device=dev)
SX=torch.zeros(k,dtype=torch.float64,device=dev)
SY=torch.zeros(D,dtype=torch.float64,device=dev); SY2=torch.zeros(D,dtype=torch.float64,device=dev)
STRUE=torch.zeros(k,dtype=torch.float64,device=dev); N=0
for c in range(60):
    w=torch.randn(10000,D,device=dev,generator=torch.Generator(device=dev).manual_seed(4000+c))
    Y,T=run(w); Td=T.double(); Yd=Y.double()
    A+=Td.t()@Td; BY+=Td.t()@Yd; SX+=Td.sum(0); SY+=Yd.sum(0); SY2+=(Yd*Yd).sum(0)
    STRUE+=Td.sum(0); N+=w.shape[0]
Etrue=(STRUE/N).cpu()
A=A.cpu();BY=BY.cpu();SX=SX.cpu();SY=SY.cpu();SY2=SY2.cpu()
SStot=SY2-SY*SY/N; alive=SStot>1e-9*float(SStot.max())
Ac=A-torch.outer(SX,SX)/N; Bc=BY-torch.outer(SX,SY)/N
beta=torch.linalg.solve(Ac+1e-6*torch.eye(k,dtype=torch.float64),Bc)     # (k,D)
SSres=SStot-(beta*Bc).sum(0); R2=float((1-SSres/SStot)[alive].mean())

log('=== NNGP Tr(K) accuracy vs truth, and the bias it would cause ===')
log('layer |   true E||a||^2   |   NNGP Tr(K)     | rel.err')
delta=[]
for i,ell in enumerate(TAPS):
    tv=float(Etrue[i]); nv=TRN[ell]; delta.append(nv-tv)
    log('  {:2d}  | {:16.6f} | {:16.6f} | {:+.3e}'.format(ell,tv,nv,(nv-tv)/tv))
delta=torch.tensor(delta,dtype=torch.float64)
bias=(beta*delta[:,None]).sum(0)                      # (D,) bias per output component
mse_bias=float((bias[alive]**2).mean())
log('')
log('LINEAR CV on the 8 deep norms:  held-in R^2 = {:.4f}  -> variance would drop to x{:.3f}'.format(R2,1-R2))
log('v2 MSE 3.09e-7  ->  variance part {:.3e}'.format(3.09e-7*(1-R2)))
log('BIAS from using NNGP traces as the known means: MSE_bias = {:.4e}'.format(mse_bias))
log('total = {:.4e}   vs v2 {:.4e}   -> {}'.format(3.09e-7*(1-R2)+mse_bias,3.09e-7,
    'WIN' if 3.09e-7*(1-R2)+mse_bias<3.09e-7 else 'LOSE'))
log('done.')
