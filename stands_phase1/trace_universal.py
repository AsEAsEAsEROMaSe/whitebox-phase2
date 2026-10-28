import torch, math, time, numpy as np
from numpy.polynomial.hermite import hermgauss
LOG='/mnt/win_d/llm/White-Box/trace_universal.log'
T0=time.time()
def log(m):
    line='[{:6.1f}s] {}'.format(time.time()-T0,m); print(line,flush=True); open(LOG,'a').write(line+'\n')
open(LOG,'w').close()
dev='cuda:4'; D=256; L=32; H=24
SQ2=math.sqrt(2.0); SQ2PI=math.sqrt(2*math.pi)
relu=lambda x: torch.clamp(x,min=0.0)
xh,wh=hermgauss(H)
xn=torch.tensor(xh*SQ2,dtype=torch.float32,device=dev); wn=torch.tensor(wh/math.sqrt(math.pi),dtype=torch.float32,device=dev)
dat=np.load('/mnt/win_d/llm/White-Box/netbank/bm16_w.npz')
TAPS=[3,7,11,15,19,23,27,30]
def phi(t): return torch.exp(-0.5*t*t)/SQ2PI
def Phi(t): return 0.5*(1+torch.erf(t/SQ2))

def nngp_traces(W):
    m=torch.zeros(D,device=dev); K=torch.eye(D,device=dev); tr=[]
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
        Kg[range(D),range(D)]=(mu*mu+sd*sd)*Ph+mu*sd*ph
        K=Kg
        if ell in TAPS: tr.append(float(torch.diagonal(K).sum()))
    return np.array(tr)

def true_traces(W,N=1_200_000,CH=60_000):
    acc=np.zeros(len(TAPS)); n=0
    for c in range(N//CH):
        z=torch.randn(CH//2,D,device=dev,generator=torch.Generator(device=dev).manual_seed(8000+c))
        x=torch.cat([z,-z],0); i=0; out=[]
        for li in range(L):
            x=relu(x@W[li])
            if li in TAPS: out.append(float((x*x).sum(1).double().sum()))
        acc+=np.array(out); n+=x.shape[0]
    return acc/n

NN=16
rel=np.zeros((NN,len(TAPS)))
for net in range(NN):
    W=[torch.tensor(dat['w_%d_%d'%(net,i)].astype(np.float32),device=dev) for i in range(L)]
    tn=nngp_traces(W); tt=true_traces(W)
    rel[net]=(tn-tt)/tt
    log('net {:2d}: rel.err per tap = {}'.format(net,' '.join('{:+.2e}'.format(v) for v in rel[net])))

log('')
log('=== IS THE NNGP TRACE ERROR UNIVERSAL ACROSS NETS? ===')
log(' tap |    mean rel.err |   std across nets | std/|mean|  -> calibratable?')
for i,ell in enumerate(TAPS):
    mu=rel[:,i].mean(); sd=rel[:,i].std()
    log('  {:2d} | {:+14.4e} | {:16.4e} | {:9.3f}'.format(ell,mu,sd,sd/abs(mu) if mu!=0 else float('nan')))
log('')
resid=rel-rel.mean(0,keepdims=True)
log('If we subtract the per-layer MEAN (offline calibration on a bank):')
log('  raw   RMS rel.err = {:.4e}'.format(np.sqrt((rel**2).mean())))
log('  after calibration = {:.4e}   -> improvement {:.2f}x'.format(
    np.sqrt((resid**2).mean()), np.sqrt((rel**2).mean())/np.sqrt((resid**2).mean())))
log('  NEEDED improvement to beat v2: 7.6x')
log('done.')
